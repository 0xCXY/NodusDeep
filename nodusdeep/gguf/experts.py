"""Expert-addressed access to MoE weights inside a GGUF file.

In the GGUF layout a MoE projection stores every expert of one layer in a single
tensor::

    blk.{layer}.ffn_gate_exps.weight   shape [n_experts, d_model, d_ff]

Experts are stacked along the outermost axis, so expert ``e`` owns a contiguous
byte range starting at ``base + e * bytes_per_expert``. That single fact is what
makes distributed MoE practical: a node can be handed expert indices rather than
a model, and read only those experts out of a file it may not even fully hold.

This module resolves those ranges and decodes them. It never loads a tensor it
was not asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import GgmlType
from .reader import GgufError, GgufFile, TensorInfo

#: Projection suffixes that make up a routed expert's feed-forward block.
EXPERT_PROJECTIONS = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")


@dataclass(frozen=True)
class ExpertSlice:
    """Byte range of one expert within one projection tensor."""

    layer: int
    projection: str
    expert: int
    offset: int
    """Absolute file offset of the expert's first byte."""
    nbytes: int
    """Bytes occupied by this expert (all rows)."""
    shape: tuple[int, int]
    """Per-expert shape as ``(out_features, in_features)``."""
    element_type: GgmlType

    @property
    def name(self) -> str:
        """Human-readable identifier, e.g. ``blk.3.ffn_gate_exps.weight[7]``."""
        return f"blk.{self.layer}.{self.projection}.weight[{self.expert}]"


class ExpertIndex:
    """Locates experts inside a GGUF file without loading them.

    Parameters:
        gguf: A parsed file.
        n_experts: Expected expert count per layer. Verified per tensor, because a
            mismatch means the caller's config does not match the checkpoint and
            every offset derived from it would be wrong.
    """

    def __init__(self, gguf: GgufFile, *, n_experts: int) -> None:
        if n_experts <= 0:
            raise ValueError("n_experts must be positive")
        self.gguf = gguf
        self.n_experts = n_experts

    # -- discovery --------------------------------------------------------

    def layers(self) -> list[int]:
        """Layer indices that contain routed experts, ascending."""
        found: set[int] = set()
        for name in self.gguf.tensors:
            parsed = _parse_expert_tensor(name)
            if parsed is not None:
                layer, projection = parsed
                if projection == "ffn_gate_exps":
                    found.add(layer)
        return sorted(found)

    def projections(self, layer: int) -> list[str]:
        """Expert projections present for ``layer``, in canonical order."""
        present = []
        for projection in EXPERT_PROJECTIONS:
            if f"blk.{layer}.{projection}.weight" in self.gguf.tensors:
                present.append(projection)
        return present

    # -- resolution -------------------------------------------------------

    def tensor_info(self, layer: int, projection: str) -> TensorInfo:
        """Directory entry for one layer's expert stack."""
        name = f"blk.{layer}.{projection}.weight"
        info = self.gguf.tensors.get(name)
        if info is None:
            raise KeyError(f"{self.gguf.path}: no tensor {name!r}")
        if len(info.shape) != 3:
            raise GgufError(f"{name}: expected rank-3 [n_experts, out, in], got shape {info.shape}")
        if info.shape[0] != self.n_experts:
            raise GgufError(
                f"{name}: has {info.shape[0]} experts, index configured for {self.n_experts}"
            )
        return info

    def slice(self, layer: int, projection: str, expert: int) -> ExpertSlice:
        """Resolve the byte range of ``expert`` in ``layer``/``projection``."""
        info = self.tensor_info(layer, projection)
        if not 0 <= expert < self.n_experts:
            raise IndexError(f"expert {expert} out of range for {self.n_experts} experts")
        per_expert = _elements_per_expert(info)
        # Rows of one expert are contiguous, so the byte width is the row width
        # times the row count, not the tensor width divided by n_experts.
        bytes_per_expert = info.row_nbytes() * per_expert[0]
        return ExpertSlice(
            layer=layer,
            projection=projection,
            expert=expert,
            offset=info.absolute_offset(self.gguf.data_start) + expert * bytes_per_expert,
            nbytes=bytes_per_expert,
            shape=(per_expert[0], per_expert[1]),
            element_type=info.element_type,
        )

    def slices(
        self, layer: int, projections: tuple[str, ...] | None = None
    ) -> dict[str, list[ExpertSlice]]:
        """Resolve every expert for the requested projections of ``layer``."""
        wanted = projections or tuple(self.projections(layer))
        return {
            projection: [self.slice(layer, projection, e) for e in range(self.n_experts)]
            for projection in wanted
        }

    # -- reading ----------------------------------------------------------

    def read_expert(self, layer: int, projection: str, expert: int) -> np.ndarray:
        """Read and dequantize one expert to a dense ``float32`` matrix."""
        from .quant import dequantize

        spec = self.slice(layer, projection, expert)
        raw = self.gguf.read_range(spec.offset, spec.nbytes)
        return dequantize(raw, spec.element_type, spec.shape)

    def read_experts(
        self,
        layer: int,
        projection: str,
        experts: list[int],
    ) -> dict[int, np.ndarray]:
        """Read several experts of one projection, keyed by expert index."""
        return {e: self.read_expert(layer, projection, e) for e in experts}


def _parse_expert_tensor(name: str) -> tuple[int, str] | None:
    """Return ``(layer, projection)`` for an expert-stack tensor name, else ``None``.

    Only names of the form ``blk.{int}.{projection}.weight`` match. A name such as
    ``blk.3.attn_q.weight`` parses to no expert stack, and expert indices embedded
    in other names (some architectures use ``experts.3.w1``) are not layer numbers.
    """
    parts = name.split(".")
    if len(parts) != 4 or parts[0] != "blk" or parts[3] != "weight":
        return None
    layer_text, projection = parts[1], parts[2]
    if projection not in EXPERT_PROJECTIONS:
        return None
    try:
        return int(layer_text), projection
    except ValueError:
        return None


def _elements_per_expert(info: TensorInfo) -> tuple[int, int]:
    """Per-expert ``(out_features, in_features)`` from a rank-3 expert stack."""
    _, out_features, in_features = info.shape
    return out_features, in_features


def discover_expert_tensors(gguf: GgufFile) -> dict[int, list[str]]:
    """Map each layer index to the expert projections it declares."""
    found: dict[int, list[str]] = {}
    for name in gguf.tensors:
        parsed = _parse_expert_tensor(name)
        if parsed is None:
            continue
        layer, projection = parsed
        projections = found.setdefault(layer, [])
        if projection not in projections:
            projections.append(projection)
    return {
        layer: [p for p in EXPERT_PROJECTIONS if p in projections]
        for layer, projections in sorted(found.items())
    }


def open_expert_index(path: str | Path, *, n_experts: int) -> ExpertIndex:
    """Parse ``path`` and return an :class:`ExpertIndex` over it."""
    return ExpertIndex(GgufFile.open(path), n_experts=n_experts)
