"""GGUF format constants.

GGUF is a single-file container for quantized model weights: a header, a
key-value metadata table, a tensor directory, and then the raw tensor data.
Every value a node needs to read a slice of the file -- tensor offsets, shapes,
quantization types -- lives in the directory, so a node can map the file and
touch only the byte ranges it was assigned.

Reference: the gguf spec published with ggml.
"""

from __future__ import annotations

import enum

MAGIC = b"GGUF"

#: The only version this reader accepts. v1 and v2 differ in length encoding and
#: are not produced by current tooling, so supporting them would be dead code.
SUPPORTED_VERSION = 3

#: GGUF pads the metadata section so the tensor data begins on this boundary.
ALIGNMENT = 32


class GgufType(enum.IntEnum):
    """Metadata value types, as encoded in the key-value table."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GgmlType(enum.IntEnum):
    """Tensor element types.

    ``block_size`` elements share one ``type_size``-byte block for the quantized
    formats. Dense formats have ``block_size == 1`` and ``type_size`` equal to the
    element width.
    """

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    # 4, 5 are retired legacy formats; not emitted by current quantizers.
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    Q4_0_4_4 = 31
    Q4_0_4_8 = 32
    Q4_0_8_8 = 33
    TQ1_0 = 34
    TQ2_0 = 35
    IQ4_NL_4_4 = 36
    IQ4_NL_4_8 = 37
    IQ4_NL_8_8 = 38


#: ``(block_size, type_size)`` per element type. Types absent from this table are
#: recognised by name but not decodable, and the reader refuses them explicitly
#: rather than guessing a stride and silently reading garbage.
TYPE_LAYOUT: dict[GgmlType, tuple[int, int]] = {
    GgmlType.F32: (1, 4),
    GgmlType.F16: (1, 2),
    GgmlType.BF16: (1, 2),
    GgmlType.F64: (1, 8),
    GgmlType.I8: (1, 1),
    GgmlType.I16: (1, 2),
    GgmlType.I32: (1, 4),
    GgmlType.I64: (1, 8),
    GgmlType.Q4_0: (32, 18),
    GgmlType.Q4_1: (32, 20),
    GgmlType.Q5_0: (32, 22),
    GgmlType.Q5_1: (32, 24),
    GgmlType.Q8_0: (32, 34),
    GgmlType.Q8_1: (32, 36),
    GgmlType.Q2_K: (256, 84),
    GgmlType.Q3_K: (256, 110),
    GgmlType.Q4_K: (256, 144),
    GgmlType.Q5_K: (256, 176),
    GgmlType.Q6_K: (256, 210),
    GgmlType.Q8_K: (256, 292),
    GgmlType.IQ4_NL: (32, 18),
    GgmlType.IQ4_XS: (256, 136),
}


def type_layout(element_type: GgmlType) -> tuple[int, int]:
    """Return ``(block_size, type_size)`` for an element type.

    Raises:
        NotImplementedError: the type is recognised but has no decoder here.
    """
    try:
        return TYPE_LAYOUT[element_type]
    except KeyError:
        raise NotImplementedError(
            f"element type {element_type.name} is not supported; "
            f"supported types: {', '.join(sorted(t.name for t in TYPE_LAYOUT))}"
        ) from None


def row_nbytes(element_type: GgmlType, row_elements: int) -> int:
    """Bytes occupied by one row of ``row_elements`` elements.

    Quantized rows are a whole number of blocks; a row that is not block-aligned
    is malformed, and we say so instead of rounding.
    """
    block_size, type_size = type_layout(element_type)
    if row_elements % block_size:
        raise ValueError(
            f"row of {row_elements} elements is not a multiple of the "
            f"{element_type.name} block size {block_size}"
        )
    return (row_elements // block_size) * type_size


def tensor_nbytes(element_type: GgmlType, shape: tuple[int, ...]) -> int:
    """Total bytes occupied by a tensor of ``shape`` (row-major, last axis fastest)."""
    if not shape:
        raise ValueError("tensor shape must have at least one dimension")
    count = 1
    for dim in shape:
        if dim <= 0:
            raise ValueError(f"tensor shape has non-positive dimension: {shape}")
        count *= dim
    block_size, type_size = type_layout(element_type)
    per_block = shape[-1]
    if per_block % block_size:
        raise ValueError(
            f"innermost dimension {per_block} is not a multiple of the "
            f"{element_type.name} block size {block_size}"
        )
    return (count // block_size) * type_size
