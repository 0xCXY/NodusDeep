"""Dequantization of GGML block formats to dense float32.

Quantized weights are kept in their packed form everywhere in the system and are
decoded here, at the point of computation, into a freshly allocated buffer.
Expanding weights to float32 at load time would multiply memory by 2-4x and defeat
the reason a node holds shards instead of a model.

Every layout below follows the ggml block definitions. A block whose size does not
divide the input is a corrupt read, not something to round around.

Attention to layout detail matters more here than anywhere else in the codebase:
a wrong nibble order or scale index still produces plausible-looking floats, and a
model fed plausible-looking wrong weights fails in ways no test of the *plumbing*
will catch. Each decoder therefore mirrors the reference loop structure rather
than being rewritten into something vectorized-looking but subtly different.
"""

from __future__ import annotations

import numpy as np

from .constants import TYPE_LAYOUT, GgmlType, type_layout


def dequantize(raw: bytes, element_type: GgmlType, shape: tuple[int, ...]) -> np.ndarray:
    """Decode ``raw`` into a ``float32`` array of ``shape``.

    Args:
        raw: Packed bytes, a whole number of blocks.
        element_type: Element format.
        shape: Logical shape of the result.

    Returns:
        A dense ``float32`` array with the requested shape.

    Raises:
        ValueError: byte count does not match the shape, or the row width is not
            block-aligned.
        NotImplementedError: the format has no decoder here.
    """
    n_elements = 1
    for dim in shape:
        n_elements *= dim

    flat = _decode(raw, element_type, n_elements)
    return flat.reshape(shape)


def _decode(raw: bytes, element_type: GgmlType, n_elements: int) -> np.ndarray:
    """Decode to a flat float32 vector of exactly ``n_elements`` entries."""
    block_size, type_size = type_layout(element_type)
    if n_elements % block_size:
        raise ValueError(
            f"{element_type.name}: {n_elements} elements is not a multiple of "
            f"the block size {block_size}"
        )
    n_blocks = n_elements // block_size
    expected = n_blocks * type_size
    if len(raw) != expected:
        raise ValueError(
            f"{element_type.name}: got {len(raw)} bytes, expected {expected} "
            f"for {n_elements} elements"
        )

    if element_type is GgmlType.F32:
        return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
    if element_type is GgmlType.F16:
        return np.frombuffer(raw, dtype="<f2").astype(np.float32)
    if element_type is GgmlType.BF16:
        return _dequant_bf16(raw)

    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(n_blocks, type_size)

    if element_type is GgmlType.Q8_0:
        return _dequant_q8_0(blocks)
    if element_type is GgmlType.Q4_0:
        return _dequant_q4_0(blocks)
    if element_type is GgmlType.Q4_K:
        return _dequant_q4_k(blocks)
    if element_type is GgmlType.Q5_K:
        return _dequant_q5_k(blocks)
    if element_type is GgmlType.Q6_K:
        return _dequant_q6_k(blocks)

    raise NotImplementedError(f"no decoder for {element_type.name}")


# -- primitive readers ----------------------------------------------------


def _half_field(blocks: np.ndarray, offset: int) -> np.ndarray:
    """Read a little-endian float16 scalar at ``offset`` from every block.

    Returns shape ``(n_blocks,)`` as float32.
    """
    raw = np.ascontiguousarray(blocks[:, offset : offset + 2])
    return raw.view("<f2").reshape(-1).astype(np.float32)


def _dequant_bf16(raw: bytes) -> np.ndarray:
    """BF16 is the top 16 bits of an IEEE float32."""
    halves = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (halves << 16).view(np.float32).copy()


# -- block decoders -------------------------------------------------------


def _dequant_q8_0(blocks: np.ndarray) -> np.ndarray:
    """32 elements: f16 scale ``d``, then 32 signed bytes multiplied by ``d``."""
    d = _half_field(blocks, 0)
    qs = blocks[:, 2:34].view(np.int8).astype(np.float32)
    return (qs * d[:, None]).reshape(-1)


def _dequant_q4_0(blocks: np.ndarray) -> np.ndarray:
    """32 elements: f16 scale ``d``, then 16 bytes of nibbles offset by -8.

    Low nibbles fill the first half of the block, high nibbles the second half.
    The halves are not interleaved element-by-element.
    """
    d = _half_field(blocks, 0)
    packed = blocks[:, 2:18]
    out = np.empty((blocks.shape[0], 32), dtype=np.float32)
    out[:, 0:16] = (packed & 0x0F).astype(np.float32) - 8.0
    out[:, 16:32] = ((packed >> 4) & 0x0F).astype(np.float32) - 8.0
    return (out * d[:, None]).reshape(-1)


def _scale_min_k4(index: int, scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack 6-bit scale/min pair ``index`` (0..7) from a 12-byte K-quant scale array.

    The first four slots hold 6-bit values directly; the last four split their
    low nibble from a byte's high nibble to reach the sixth bit.

    Returns:
        ``(scale, min)`` as float32, each shaped ``(n_blocks,)``.
    """
    if index < 4:
        scale = (scales[:, index] & 0x3F).astype(np.float32)
        minimum = (scales[:, index + 4] & 0x3F).astype(np.float32)
    else:
        scale = ((scales[:, index + 4] & 0x0F) | ((scales[:, index - 4] >> 6) << 4)).astype(
            np.float32
        )
        minimum = ((scales[:, index + 4] >> 4) | ((scales[:, index] >> 6) << 4)).astype(np.float32)
    return scale, minimum


def _dequant_q4_k(blocks: np.ndarray) -> np.ndarray:
    """256 elements per block.

    Layout: f16 ``d`` (0), f16 ``dmin`` (2), 12 scale bytes (4), 128 nibble
    bytes (16).
    """
    d = _half_field(blocks, 0)
    dmin = _half_field(blocks, 2)
    scales = blocks[:, 4:16]
    packed = blocks[:, 16:144]

    out = np.empty((blocks.shape[0], 256), dtype=np.float32)
    for group in range(4):
        scale_lo, min_lo = _scale_min_k4(2 * group, scales)
        scale_hi, min_hi = _scale_min_k4(2 * group + 1, scales)
        chunk = packed[:, group * 32 : (group + 1) * 32]

        d_lo = (d * scale_lo)[:, None]
        m_lo = (dmin * min_lo)[:, None]
        d_hi = (d * scale_hi)[:, None]
        m_hi = (dmin * min_hi)[:, None]

        base = group * 64
        out[:, base : base + 32] = d_lo * (chunk & 0x0F).astype(np.float32) - m_lo
        out[:, base + 32 : base + 64] = d_hi * ((chunk >> 4) & 0x0F).astype(np.float32) - m_hi
    return out.reshape(-1)


def _dequant_q5_k(blocks: np.ndarray) -> np.ndarray:
    """256 elements per block.

    Layout: f16 ``d`` (0), f16 ``dmin`` (2), 12 scale bytes (4), 32 high-bit
    bytes (16), 128 low-nibble bytes (48).

    The fifth bit of element ``l`` in group ``g`` is bit ``2g`` of ``qh[l]`` for
    the low half and bit ``2g+1`` for the high half. ``qh`` is shared across all
    four groups rather than sliced per group.
    """
    d = _half_field(blocks, 0)
    dmin = _half_field(blocks, 2)
    scales = blocks[:, 4:16]
    qh = blocks[:, 16:48]
    ql = blocks[:, 48:176]

    out = np.empty((blocks.shape[0], 256), dtype=np.float32)
    for group in range(4):
        scale_lo, min_lo = _scale_min_k4(2 * group, scales)
        scale_hi, min_hi = _scale_min_k4(2 * group + 1, scales)
        low_mask = 1 << (2 * group)
        high_mask = 1 << (2 * group + 1)

        chunk = ql[:, group * 32 : (group + 1) * 32]
        low5 = (chunk & 0x0F).astype(np.float32) + ((qh & low_mask) != 0).astype(np.float32) * 16.0
        high5 = ((chunk >> 4) & 0x0F).astype(np.float32) + ((qh & high_mask) != 0).astype(
            np.float32
        ) * 16.0

        d_lo = (d * scale_lo)[:, None]
        m_lo = (dmin * min_lo)[:, None]
        d_hi = (d * scale_hi)[:, None]
        m_hi = (dmin * min_hi)[:, None]

        base = group * 64
        out[:, base : base + 32] = d_lo * low5 - m_lo
        out[:, base + 32 : base + 64] = d_hi * high5 - m_hi
    return out.reshape(-1)


def _dequant_q6_k(blocks: np.ndarray) -> np.ndarray:
    """256 elements per block.

    Layout: 128 low-nibble bytes (0), 64 high-bit bytes (128), 16 int8 scales
    (192), f16 ``d`` (208).

    Values are 6-bit, stored biased by 32. The block splits into two halves of
    128 outputs, each half taking its own 64 low-nibble bytes, 32 high-bit bytes
    and 8 scales.
    """
    d = _half_field(blocks, 208)
    ql = blocks[:, 0:128]
    qh = blocks[:, 128:192]
    scales = blocks[:, 192:208].view(np.int8).astype(np.float32)

    out = np.empty((blocks.shape[0], 256), dtype=np.float32)
    # Within a half, output position l and l+64 use the same low-nibble byte but
    # different 2-bit fields of the same high-bit byte.
    is_index = (np.arange(32) // 16).astype(np.int64)

    for half in range(2):
        ql_half = ql[:, half * 64 : (half + 1) * 64]
        qh_half = qh[:, half * 32 : (half + 1) * 32]
        sc_half = scales[:, half * 8 : (half + 1) * 8]

        ql_lo = ql_half[:, 0:32]
        ql_hi = ql_half[:, 32:64]

        q1 = ((ql_lo & 0x0F) | (((qh_half >> 0) & 0x03) << 4)).astype(np.int16).astype(
            np.float32
        ) - 32.0
        q2 = ((ql_hi & 0x0F) | (((qh_half >> 2) & 0x03) << 4)).astype(np.int16).astype(
            np.float32
        ) - 32.0
        q3 = (((ql_lo >> 4) & 0x0F) | (((qh_half >> 4) & 0x03) << 4)).astype(np.int16).astype(
            np.float32
        ) - 32.0
        q4 = (((ql_hi >> 4) & 0x0F) | (((qh_half >> 6) & 0x03) << 4)).astype(np.int16).astype(
            np.float32
        ) - 32.0

        d_b = d[:, None]
        s1 = d_b * sc_half[:, is_index]
        s2 = d_b * sc_half[:, is_index + 2]
        s3 = d_b * sc_half[:, is_index + 4]
        s4 = d_b * sc_half[:, is_index + 6]

        base = half * 128
        out[:, base : base + 32] = s1 * q1
        out[:, base + 32 : base + 64] = s2 * q2
        out[:, base + 64 : base + 96] = s3 * q3
        out[:, base + 96 : base + 128] = s4 * q4
    return out.reshape(-1)


_DECODABLE = frozenset(
    {
        GgmlType.F32,
        GgmlType.F16,
        GgmlType.BF16,
        GgmlType.Q8_0,
        GgmlType.Q4_0,
        GgmlType.Q4_K,
        GgmlType.Q5_K,
        GgmlType.Q6_K,
    }
)


def supported_types() -> list[str]:
    """Names of element types this module can decode."""
    return sorted(t.name for t in _DECODABLE if t in TYPE_LAYOUT)


def is_decodable(element_type: GgmlType) -> bool:
    """Whether :func:`dequantize` handles ``element_type``."""
    return element_type in _DECODABLE
