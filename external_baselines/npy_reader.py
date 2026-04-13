from __future__ import annotations

import ast
import struct
from pathlib import Path
from typing import Any, Sequence


_DTYPE_MAP = {
    "<i4": ("<i", 4),
    "<i8": ("<q", 8),
    "<u4": ("<I", 4),
    "<u8": ("<Q", 8),
    "|u1": ("B", 1),
    "<u1": ("B", 1),
}


def _prod(values: Sequence[int]) -> int:
    total = 1
    for value in values:
        total *= value
    return total


def _reshape(values: list[int], shape: Sequence[int]) -> Any:
    if not shape:
        return values[0]
    if len(shape) == 1:
        return values
    stride = _prod(shape[1:])
    return [
        _reshape(values[index * stride : (index + 1) * stride], shape[1:])
        for index in range(shape[0])
    ]


def load_npy(path: str | Path) -> Any:
    """
    Minimal `.npy` reader for the integer arrays used by the Sudoku datasets.

    This keeps the external-baseline smoke harness independent of NumPy, which
    is useful on machines where third-party Python packages are not yet
    available.
    """

    path = Path(path)
    with path.open("rb") as handle:
        magic = handle.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"{path} is not a valid .npy file")

        major = handle.read(1)[0]
        minor = handle.read(1)[0]
        if major == 1:
            header_len = struct.unpack("<H", handle.read(2))[0]
        elif major in (2, 3):
            header_len = struct.unpack("<I", handle.read(4))[0]
        else:
            raise ValueError(f"Unsupported .npy version {major}.{minor} in {path}")

        header = ast.literal_eval(handle.read(header_len).decode("latin1"))
        descr = header["descr"]
        if descr not in _DTYPE_MAP:
            raise ValueError(f"Unsupported dtype {descr!r} in {path}")
        if header["fortran_order"]:
            raise ValueError(f"Fortran-order arrays are not supported in {path}")

        shape = header["shape"]
        if isinstance(shape, int):
            shape = (shape,)

        fmt, itemsize = _DTYPE_MAP[descr]
        expected_items = _prod(shape)
        raw = handle.read()
        expected_bytes = expected_items * itemsize
        if len(raw) < expected_bytes:
            raise ValueError(
                f"{path} is truncated: expected {expected_bytes} bytes, got {len(raw)}"
            )

        values = [item[0] for item in struct.iter_unpack(fmt, raw[:expected_bytes])]
        return _reshape(values, shape)
