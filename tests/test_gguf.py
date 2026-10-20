"""Tests for GGUF parsing, expert addressing and dequantization.

The central property under test is that reading one expert returns exactly that
expert's bytes -- not its neighbour's, not a range shifted by one row. A file with
distinguishable per-expert content makes any off-by-one immediately visible.
"""

from __future__ import annotations

import itertools
import struct

import numpy as np
import pytest

from nodusdeep.gguf import (
    ExpertIndex,
    GgmlType,
    GgufError,
    GgufFile,
    dequantize,
    discover_expert_tensors,
    row_nbytes,
    tensor_nbytes,
    type_layout,
)
from nodusdeep.gguf.writer import GgufWriter

# -- fixtures -------------------------------------------------------------


def _pack_q8_0(values: np.ndarray) -> bytes:
    """Encode one block of 32 float32 values as Q8_0."""
    values = np.asarray(values, dtype=np.float32)
    peak = float(np.abs(values).max())
    # An all-zero block has no scale to derive; encoding it as d=0 with q=0
    # decodes back to zero, which is the correct round-trip.
    if peak == 0.0:
        return np.float16(0.0).tobytes() + np.zeros(32, dtype=np.int8).tobytes()
    scale = float(np.float16(peak / 127.0))
    qs = np.clip(np.round(values / scale), -127, 127).astype(np.int8)
    return np.float16(scale).tobytes() + qs.tobytes()


def _pack_f32(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(values, dtype="<f4").tobytes()


def build_moe_file(
    path,
    *,
    n_layers: int = 2,
    n_experts: int = 4,
    d_model: int = 8,
    d_ff: int = 8,
    element_type: GgmlType = GgmlType.F32,
    arch: str = "testmoe",
):
    """Write a synthetic MoE GGUF with uniquely identifiable expert contents.

    Each expert's values encode ``(layer, projection_index, expert)`` so a test
    can assert exactly which expert came back.
    """
    writer = GgufWriter()
    writer.add_string("general.architecture", arch)
    writer.add_uint32(f"{arch}.block_count", n_layers)
    writer.add_uint32(f"{arch}.expert_count", n_experts)

    projections = ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
    expected: dict[tuple[int, str, int], np.ndarray] = {}

    for layer in range(n_layers):
        for p_index, projection in enumerate(projections):
            stack = np.zeros((n_experts, d_ff, d_model), dtype=np.float32)
            for expert in range(n_experts):
                # A distinct constant per expert: catches shifts by any amount.
                stack[expert, :, :] = layer * 1000 + p_index * 100 + expert
                expected[(layer, projection, expert)] = stack[expert].copy()

            if element_type is GgmlType.F32:
                data = _pack_f32(stack)
                shape = stack.shape
            elif element_type is GgmlType.Q8_0:
                rows = []
                for expert in range(n_experts):
                    for row in range(d_ff):
                        rows.append(_pack_q8_0(stack[expert, row, :]))
                data = b"".join(rows)
                shape = stack.shape
            else:
                raise AssertionError(f"unsupported fixture type {element_type}")

            writer.add_tensor(f"blk.{layer}.{projection}.weight", data, shape, element_type)

    writer.write(path)
    return expected


# -- container ------------------------------------------------------------


def test_roundtrip_metadata_and_directory(tmp_path):
    path = tmp_path / "m.gguf"
    build_moe_file(path)

    gguf = GgufFile.open(path)
    assert gguf.version == 3
    assert gguf.arch() == "testmoe"
    assert gguf.int_meta("block_count") == 2
    assert gguf.int_meta("expert_count") == 4
    assert gguf.data_start % 32 == 0
    assert len(gguf.tensors) == 6


def test_rejects_non_gguf(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(b"NOPE" + b"\x00" * 64)
    with pytest.raises(GgufError, match="not a GGUF"):
        GgufFile.open(path)


def test_rejects_wrong_version(tmp_path):
    path = tmp_path / "v2.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<I", 2) + b"\x00" * 32)
    with pytest.raises(GgufError, match="version 2"):
        GgufFile.open(path)


def test_rejects_truncated_file(tmp_path):
    path = tmp_path / "short.gguf"
    build_moe_file(path)
    data = path.read_bytes()
    path.write_bytes(data[:40])
    with pytest.raises(GgufError, match="unexpected end of file"):
        GgufFile.open(path)


def test_tensor_shapes_are_row_major(tmp_path):
    path = tmp_path / "m.gguf"
    build_moe_file(path, n_experts=3, d_ff=4, d_model=16)
    gguf = GgufFile.open(path)
    info = gguf.tensor("blk.0.ffn_gate_exps.weight")
    # Writers store dimensions innermost-first; readers must reverse them.
    assert info.shape == (3, 4, 16)
    assert info.n_elements == 3 * 4 * 16


# -- expert addressing ----------------------------------------------------


def test_expert_slices_are_contiguous_and_disjoint(tmp_path):
    path = tmp_path / "m.gguf"
    build_moe_file(path, n_layers=2, n_experts=5, d_ff=8, d_model=8)
    index = ExpertIndex(GgufFile.open(path), n_experts=5)

    for layer in range(2):
        for projection in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            slices = [index.slice(layer, projection, e) for e in range(5)]
            for earlier, later in itertools.pairwise(slices):
                # Contiguous means the next expert starts exactly where the
                # previous one ends: no padding, no overlap.
                assert later.offset == earlier.offset + earlier.nbytes
            assert slices[0].nbytes == 8 * 8 * 4


def test_reads_the_requested_expert_not_a_neighbour(tmp_path):
    path = tmp_path / "m.gguf"
    expected = build_moe_file(path, n_layers=2, n_experts=4, d_ff=8, d_model=8)
    index = ExpertIndex(GgufFile.open(path), n_experts=4)

    for layer in range(2):
        for projection in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            for expert in range(4):
                got = index.read_expert(layer, projection, expert)
                want = expected[(layer, projection, expert)]
                assert np.array_equal(got, want), f"wrong expert: {layer}/{projection}/{expert}"


def test_reads_quantized_expert_with_correct_values(tmp_path):
    path = tmp_path / "q.gguf"
    build_moe_file(path, n_layers=1, n_experts=3, d_ff=4, d_model=32, element_type=GgmlType.Q8_0)
    index = ExpertIndex(GgufFile.open(path), n_experts=3)

    for expert in range(3):
        got = index.read_expert(0, "ffn_gate_exps", expert)
        # Every value in this expert was stored as the same integer, so the block
        # scale is exact up to float16 -- 127 * fp16(1/127) lands within 1e-3 of 1.
        # The point of the test is that the right expert's bytes were addressed and
        # decoded, which a uniform value still proves: a wrong expert yields a
        # different constant, not a different rounding.
        want = np.full((4, 32), float(expert), dtype=np.float32)
        assert got.shape == (4, 32)
        assert np.allclose(got, want, atol=1e-3), f"expert {expert} decoded wrong"


def test_expert_index_rejects_layer_count_mismatch(tmp_path):
    path = tmp_path / "m.gguf"
    build_moe_file(path, n_experts=4)
    index = ExpertIndex(GgufFile.open(path), n_experts=5)
    with pytest.raises(GgufError, match="has 4 experts"):
        index.slice(0, "ffn_gate_exps", 0)


def test_expert_index_rejects_out_of_range(tmp_path):
    path = tmp_path / "m.gguf"
    build_moe_file(path, n_experts=4)
    index = ExpertIndex(GgufFile.open(path), n_experts=4)
    with pytest.raises(IndexError, match="out of range"):
        index.slice(0, "ffn_gate_exps", 4)
    with pytest.raises(IndexError, match="out of range"):
        index.slice(0, "ffn_gate_exps", -1)


def test_layers_and_projections_discovered(tmp_path):
    path = tmp_path / "m.gguf"
    build_moe_file(path, n_layers=3, n_experts=2)
    gguf = GgufFile.open(path)
    index = ExpertIndex(gguf, n_experts=2)
    assert index.layers() == [0, 1, 2]
    assert index.projections(0) == ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
    assert discover_expert_tensors(gguf) == {
        0: ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"],
        1: ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"],
        2: ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"],
    }


def test_missing_projection_reported(tmp_path):
    path = tmp_path / "partial.gguf"
    writer = GgufWriter()
    writer.add_string("general.architecture", "testmoe")
    writer.add_f32_tensor("blk.0.ffn_gate_exps.weight", np.zeros((2, 4, 8), dtype=np.float32))
    writer.write(path)

    index = ExpertIndex(GgufFile.open(path), n_experts=2)
    assert index.projections(0) == ["ffn_gate_exps"]
    with pytest.raises(KeyError, match="ffn_up_exps"):
        index.slice(0, "ffn_up_exps", 0)


# -- layout arithmetic ----------------------------------------------------


@pytest.mark.parametrize(
    "element_type,expected",
    [
        (GgmlType.F32, (1, 4)),
        (GgmlType.F16, (1, 2)),
        (GgmlType.Q8_0, (32, 34)),
        (GgmlType.Q4_K, (256, 144)),
        (GgmlType.Q6_K, (256, 210)),
    ],
)
def test_type_layouts(element_type, expected):
    assert type_layout(element_type) == expected


def test_row_nbytes_rejects_unaligned_row():
    with pytest.raises(ValueError, match="not a multiple"):
        row_nbytes(GgmlType.Q4_K, 100)


def test_tensor_nbytes_matches_expert_stack(tmp_path):
    path = tmp_path / "m.gguf"
    build_moe_file(path, n_experts=3, d_ff=16, d_model=32)
    gguf = GgufFile.open(path)
    info = gguf.tensor("blk.0.ffn_gate_exps.weight")
    assert info.nbytes == 3 * 16 * 32 * 4
    assert info.nbytes == tensor_nbytes(GgmlType.F32, info.shape)


def test_unsupported_element_type_refused():
    from nodusdeep.gguf.constants import tensor_nbytes as tnb

    with pytest.raises(NotImplementedError, match="not supported"):
        tnb(GgmlType.IQ1_S, (1, 256))


# -- dequantization -------------------------------------------------------


def test_dequantize_f32_identity():
    values = np.arange(32, dtype=np.float32)
    got = dequantize(values.tobytes(), GgmlType.F32, (32,))
    assert np.array_equal(got, values)


def test_dequantize_f16_roundtrip():
    values = np.array([[0.5, -1.25], [2.0, 3.5]], dtype=np.float16)
    got = dequantize(values.tobytes(), GgmlType.F16, (2, 2))
    assert got.shape == (2, 2)
    assert np.allclose(got, values.astype(np.float32))


def test_dequantize_bf16_roundtrip():
    values = np.array([1.0, -2.0, 0.5, 4.0], dtype=np.float32)
    bf16 = (values.view(np.uint32) >> 16).astype(np.uint16)
    got = dequantize(bf16.tobytes(), GgmlType.BF16, (4,))
    assert np.allclose(got, values)


def test_dequantize_q8_0_block():
    values = np.array([(-1) ** i * i for i in range(32)], dtype=np.float32)
    packed = _pack_q8_0(values)
    got = dequantize(packed, GgmlType.Q8_0, (32,))
    scale = np.float32(np.float16(np.abs(values).max() / 127.0))
    want = np.clip(np.round(values / scale), -127, 127) * scale
    assert np.allclose(got, want, atol=1e-5)


def test_dequantize_q4_0_interleaving():
    """Low nibbles fill the first half of the block, high nibbles the second."""
    scale = np.float16(0.5)
    packed = bytes([0x21, 0x43] + [0x00] * 14)  # only the first two bytes set
    raw = scale.tobytes() + packed
    got = dequantize(raw, GgmlType.Q4_0, (32,))

    # Byte 0x21 -> low nibble 1, high nibble 2. Byte 0x43 -> low 3, high 4.
    assert got[0] == pytest.approx((1 - 8) * 0.5)
    assert got[1] == pytest.approx((3 - 8) * 0.5)
    assert got[16] == pytest.approx((2 - 8) * 0.5)
    assert got[17] == pytest.approx((4 - 8) * 0.5)
    # Untouched bytes decode to minus eight times the scale, not to zero.
    assert got[2] == pytest.approx(-8 * 0.5)


def test_dequantize_rejects_wrong_length():
    with pytest.raises(ValueError, match="expected"):
        dequantize(b"\x00" * 10, GgmlType.Q8_0, (32,))


def test_dequantize_rejects_unaligned_shape():
    with pytest.raises(ValueError, match="not a multiple"):
        dequantize(b"\x00" * 34, GgmlType.Q8_0, (31,))


def test_supported_types_listed():
    from nodusdeep.gguf.quant import supported_types

    names = supported_types()
    assert "Q4_K" in names
    assert "Q6_K" in names
    assert "IQ1_S" not in names
