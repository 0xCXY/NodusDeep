"""Minimal GGUF writer, for building fixtures and emitting shards.

This writes the container only: header, metadata, tensor directory, data section.
It exists so tests can construct files with known contents and verify that the
reader lands on exactly the right bytes -- an end-to-end check that would be
impossible against a real checkpoint whose ground truth we cannot inspect.

It is also the first half of shard export: given a source file and a set of expert
indices, the same writer emits a container holding only those experts.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .constants import ALIGNMENT, MAGIC, SUPPORTED_VERSION, GgmlType, GgufType


@dataclass
class OutTensor:
    """A tensor to be written."""

    name: str
    data: bytes
    """Packed element bytes."""
    shape: tuple[int, ...]
    """Logical shape, outermost first."""
    element_type: GgmlType


@dataclass
class GgufWriter:
    """Accumulates metadata and tensors, then writes one file."""

    metadata: dict[str, tuple[GgufType, object]] = field(default_factory=dict)
    tensors: list[OutTensor] = field(default_factory=list)

    def add_meta(self, key: str, value_type: GgufType, value: object) -> GgufWriter:
        """Record a metadata entry."""
        self.metadata[key] = (value_type, value)
        return self

    def add_string(self, key: str, value: str) -> GgufWriter:
        """Record a string metadata entry."""
        return self.add_meta(key, GgufType.STRING, value)

    def add_uint32(self, key: str, value: int) -> GgufWriter:
        """Record a uint32 metadata entry."""
        return self.add_meta(key, GgufType.UINT32, value)

    def add_int32(self, key: str, value: int) -> GgufWriter:
        """Record an int32 metadata entry."""
        return self.add_meta(key, GgufType.INT32, value)

    def add_float32(self, key: str, value: float) -> GgufWriter:
        """Record a float32 metadata entry."""
        return self.add_meta(key, GgufType.FLOAT32, value)

    def add_string_array(self, key: str, values: list[str]) -> GgufWriter:
        """Record a string-array metadata entry."""
        return self.add_meta(key, GgufType.ARRAY, (GgufType.STRING, values))

    def add_tensor(
        self,
        name: str,
        data: bytes,
        shape: tuple[int, ...],
        element_type: GgmlType,
    ) -> GgufWriter:
        """Queue a tensor for writing."""
        if name in {t.name for t in self.tensors}:
            raise ValueError(f"duplicate tensor name {name!r}")
        self.tensors.append(OutTensor(name, data, shape, element_type))
        return self

    def add_f32_tensor(self, name: str, array: np.ndarray) -> GgufWriter:
        """Queue a float32 tensor from an array."""
        array = np.ascontiguousarray(array, dtype="<f4")
        return self.add_tensor(name, array.tobytes(), array.shape, GgmlType.F32)

    # -- serialization ----------------------------------------------------

    def to_bytes(self) -> bytes:
        """Serialize to a complete GGUF file image."""
        header = bytearray()
        header += MAGIC
        header += struct.pack("<I", SUPPORTED_VERSION)
        header += struct.pack("<Q", len(self.tensors))
        header += struct.pack("<Q", len(self.metadata))

        for key, (value_type, value) in self.metadata.items():
            header += _encode_string(key)
            header += struct.pack("<I", int(value_type))
            header += _encode_value(value_type, value)

        # Tensor offsets are relative to the data section, which starts after the
        # directory is padded to the alignment boundary.
        offsets: list[int] = []
        cursor = 0
        for tensor in self.tensors:
            offsets.append(cursor)
            cursor = _align_up(cursor + len(tensor.data), ALIGNMENT)

        for tensor, offset in zip(self.tensors, offsets, strict=True):
            header += _encode_string(tensor.name)
            header += struct.pack("<I", len(tensor.shape))
            for dim in reversed(tensor.shape):
                header += struct.pack("<Q", dim)
            header += struct.pack("<I", int(tensor.element_type))
            header += struct.pack("<Q", offset)

        data_start = _align_up(len(header), ALIGNMENT)
        # Pad from the end of the header to the data section. ``body`` is indexed
        # from the end of the header, so a tensor at data-section offset ``o``
        # lives at body index ``data_start - len(header) + o``. Mixing that with
        # absolute file offsets shifts every tensor by the header length.
        body = bytearray(b"\x00" * (data_start - len(header)))
        body_origin = data_start - len(header)

        for tensor, offset in zip(self.tensors, offsets, strict=True):
            target = body_origin + offset
            if len(body) < target:
                body += b"\x00" * (target - len(body))
            body += tensor.data

        tail = _align_up(len(header) + len(body), ALIGNMENT) - (len(header) + len(body))
        if tail:
            body += b"\x00" * tail

        return bytes(header) + bytes(body)

    def write(self, path: str | Path) -> None:
        """Serialize to ``path``."""
        Path(path).write_bytes(self.to_bytes())


def _encode_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _encode_value(value_type: GgufType, value: object) -> bytes:
    if value_type is GgufType.STRING:
        return _encode_string(str(value))
    if value_type is GgufType.ARRAY:
        item_type, items = value  # type: ignore[misc]
        out = struct.pack("<I", int(item_type))
        out += struct.pack("<Q", len(items))  # type: ignore[arg-type]
        for item in items:  # type: ignore[union-attr]
            out += _encode_value(item_type, item)
        return out
    fmt = {
        GgufType.UINT8: "<B",
        GgufType.INT8: "<b",
        GgufType.UINT16: "<H",
        GgufType.INT16: "<h",
        GgufType.UINT32: "<I",
        GgufType.INT32: "<i",
        GgufType.FLOAT32: "<f",
        GgufType.BOOL: "<?",
        GgufType.UINT64: "<Q",
        GgufType.INT64: "<q",
        GgufType.FLOAT64: "<d",
    }[value_type]
    return struct.pack(fmt, value)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment
