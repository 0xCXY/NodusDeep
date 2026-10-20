"""GGUF container reader.

Parses the header, metadata table and tensor directory of a GGUF v3 file. Only
the directory is read eagerly; tensor data is never touched until something asks
for a byte range, which is what makes byte-offset shard loading cheap.

A malformed file raises :class:`GgufError` with the offset it failed at. Silent
recovery is not attempted: a wrong offset produces plausible-looking garbage, and
garbage weights are indistinguishable from a bad model after the fact.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

from .constants import ALIGNMENT, MAGIC, SUPPORTED_VERSION, GgmlType, GgufType, tensor_nbytes

#: Fixed-width scalar decoders for the metadata table.
_SCALAR_FORMAT: dict[GgufType, str] = {
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
}

_SCALAR_SIZE = {t: struct.calcsize(f) for t, f in _SCALAR_FORMAT.items()}


class GgufError(ValueError):
    """A GGUF file is malformed or uses an unsupported feature."""


@dataclass(frozen=True)
class TensorInfo:
    """One entry in the tensor directory.

    ``offset`` is relative to the start of the tensor data section, per the spec;
    :meth:`absolute_offset` adds the section base.
    """

    name: str
    shape: tuple[int, ...]
    element_type: GgmlType
    offset: int

    @property
    def nbytes(self) -> int:
        """Bytes this tensor occupies."""
        return tensor_nbytes(self.element_type, self.shape)

    @property
    def n_elements(self) -> int:
        """Element count across all dimensions."""
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    def absolute_offset(self, data_start: int) -> int:
        """File offset of this tensor's first byte."""
        return data_start + self.offset

    def row_nbytes(self) -> int:
        """Bytes in one row (the innermost dimension)."""
        from .constants import row_nbytes

        return row_nbytes(self.element_type, self.shape[-1])


@dataclass
class GgufFile:
    """A parsed GGUF file.

    The file handle is opened on demand by :meth:`read_range`, so instances are
    cheap to create and safe to keep around.
    """

    path: Path
    version: int
    metadata: dict[str, object]
    tensors: dict[str, TensorInfo] = field(default_factory=dict)
    data_start: int = 0

    # -- construction -----------------------------------------------------

    @classmethod
    def open(cls, path: str | Path) -> GgufFile:
        """Parse the header and directory of ``path``."""
        path = Path(path)
        with path.open("rb") as fh:
            reader = _Cursor(fh.read(), path)
            magic = reader.take(4)
            if magic != MAGIC:
                raise GgufError(f"{path}: not a GGUF file (magic {magic!r})")

            version = reader.u32()
            if version != SUPPORTED_VERSION:
                raise GgufError(
                    f"{path}: GGUF version {version} is not supported "
                    f"(only v{SUPPORTED_VERSION} is)"
                )

            n_tensors = reader.u64()
            n_kv = reader.u64()

            metadata: dict[str, object] = {}
            for _ in range(n_kv):
                key = reader.string()
                metadata[key] = reader.value(reader.u32())

            tensors: dict[str, TensorInfo] = {}
            for _ in range(n_tensors):
                name = reader.string()
                n_dims = reader.u32()
                if n_dims == 0 or n_dims > 8:
                    raise GgufError(f"{path}: tensor {name!r} has {n_dims} dimensions")
                # Dimensions are stored innermost-first, so reading them in order
                # and then reversing gives row-major shape.
                dims = [reader.u64() for _ in range(n_dims)]
                raw_type = reader.u32()
                try:
                    element_type = GgmlType(raw_type)
                except ValueError:
                    raise GgufError(
                        f"{path}: tensor {name!r} has unknown element type {raw_type}"
                    ) from None
                offset = reader.u64()
                shape = tuple(reversed(dims))
                if name in tensors:
                    raise GgufError(f"{path}: duplicate tensor name {name!r}")
                tensors[name] = TensorInfo(name, shape, element_type, offset)

            data_start = _align_up(reader.pos, ALIGNMENT)

        return cls(
            path=path,
            version=version,
            metadata=metadata,
            tensors=tensors,
            data_start=data_start,
        )

    # -- data access ------------------------------------------------------

    def read_range(self, offset: int, length: int) -> bytes:
        """Read ``length`` bytes at absolute file ``offset``."""
        if offset < 0 or length < 0:
            raise ValueError("offset and length must be non-negative")
        with self.path.open("rb") as fh:
            fh.seek(offset)
            data = fh.read(length)
        if len(data) != length:
            raise GgufError(
                f"{self.path}: wanted {length} bytes at {offset}, file ended after {len(data)}"
            )
        return data

    def read_tensor(self, name: str) -> bytes:
        """Read a tensor's complete raw bytes."""
        info = self.tensor(name)
        return self.read_range(info.absolute_offset(self.data_start), info.nbytes)

    def tensor(self, name: str) -> TensorInfo:
        """Look up a tensor by name."""
        try:
            return self.tensors[name]
        except KeyError:
            raise KeyError(f"{self.path}: no tensor named {name!r}") from None

    # -- metadata helpers -------------------------------------------------

    def get(self, key: str, default: object = None) -> object:
        """Read a metadata value."""
        return self.metadata.get(key, default)

    def arch(self) -> str:
        """The model architecture string, e.g. ``llama`` or ``deepseek2``."""
        value = self.metadata.get("general.architecture")
        if not isinstance(value, str):
            raise GgufError(f"{self.path}: missing general.architecture")
        return value

    def key(self, name: str) -> str:
        """Architecture-namespaced metadata key, e.g. ``llama.block_count``."""
        return f"{self.arch()}.{name}"

    def int_meta(self, name: str, default: int | None = None) -> int:
        """Read an integer metadata field, namespaced by architecture."""
        value = self.metadata.get(self.key(name), default)
        if not isinstance(value, int):
            raise GgufError(
                f"{self.path}: metadata {self.key(name)!r} is missing or not an integer"
            )
        return value

    def str_meta(self, name: str) -> str | None:
        """Read a string metadata field, namespaced by architecture."""
        value = self.metadata.get(self.key(name))
        return value if isinstance(value, str) else None


class _Cursor:
    """Sequential reader over an in-memory buffer with offset reporting."""

    __slots__ = ("buf", "path", "pos")

    def __init__(self, buf: bytes, path: Path) -> None:
        self.buf = buf
        self.pos = 0
        self.path = path

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.buf):
            raise GgufError(
                f"{self.path}: unexpected end of file at offset {self.pos} "
                f"(wanted {n} bytes, {len(self.buf) - self.pos} available)"
            )
        start = self.pos
        self.pos += n
        return self.buf[start : self.pos]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        length = self.u64()
        try:
            return self.take(length).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GgufError(f"{self.path}: invalid UTF-8 in string at offset {self.pos}") from exc

    def value(self, type_id: int) -> object:
        """Decode one metadata value of ``type_id``."""
        try:
            value_type = GgufType(type_id)
        except ValueError:
            raise GgufError(
                f"{self.path}: unknown metadata type {type_id} at offset {self.pos}"
            ) from None

        if value_type is GgufType.STRING:
            return self.string()
        if value_type is GgufType.ARRAY:
            item_type = self.u32()
            count = self.u64()
            # Guard against a corrupt count allocating gigabytes before failing.
            if count > len(self.buf):
                raise GgufError(
                    f"{self.path}: array length {count} exceeds file size at offset {self.pos}"
                )
            return [self.value(item_type) for _ in range(count)]
        return struct.unpack(_SCALAR_FORMAT[value_type], self.take(_SCALAR_SIZE[value_type]))[0]


def _align_up(value: int, alignment: int) -> int:
    """Round ``value`` up to the next multiple of ``alignment``."""
    return (value + alignment - 1) // alignment * alignment
