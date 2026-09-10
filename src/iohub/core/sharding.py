"""Resolve a shard geometry from the caller's intent.

A zarr v3 shard must span a whole number of chunks, so choosing one by hand
means computing a per-axis chunk multiple for every output shape. This module
lets the caller state the intent instead -- a target file size (``"2GB"``) or a
spatial extent (``"XYZ"``) -- and derives a shard shape that is chunk-aligned
and no larger than the array.

Two properties are deliberate, and only hold for the derived forms:

- **Channels are never grown.** Sharding along the channel axis breaks
  per-channel writes (see ``iohub.ngff.utils.process_single_position``), so the
  channel axis always stays at one chunk.
- **Spatial axes grow before time.** A torn shard costs everything it spans,
  so a size target fills X, then Y, then Z before it spends any of the budget
  on timepoints.
"""

from __future__ import annotations

import logging
import math
import re
import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import DTypeLike

_logger = logging.getLogger(__name__)

#: What ``resolve_shards`` accepts: an explicit shard shape, a spatial extent
#: keyword, a target size string, or None for no sharding.
type ShardsLike = tuple[int, ...] | str | None

#: Spatial extent keywords, mapped to the number of trailing axes they cover.
#: Any letter order is accepted, so ``"ZYX"`` and ``"XYZ"`` are the same.
_EXTENT_KEYWORDS: dict[frozenset[str], int] = {
    frozenset("XY"): 2,
    frozenset("XYZ"): 3,
}

#: Byte-size suffixes, both SI (``GB``) and binary (``GiB``).
_SIZE_UNITS: dict[str, int] = {
    "B": 1,
    "KB": 10**3,
    "MB": 10**6,
    "GB": 10**9,
    "TB": 10**12,
    "KIB": 2**10,
    "MIB": 2**20,
    "GIB": 2**30,
    "TIB": 2**40,
}

_SIZE_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]I?B|B)?\s*$", re.IGNORECASE)

#: Axis names treated as channel and time, compared case-insensitively.
_CHANNEL_NAMES = frozenset({"c", "channel"})
_TIME_NAMES = frozenset({"t", "time"})

#: Axis names assumed when the caller does not pass any, aligned to the end of
#: the shape. iohub arrays are TCZYX, so a 5D array without names is read as
#: TCZYX and a lower-rank one as its trailing axes.
_DEFAULT_AXIS_NAMES: tuple[str, ...] = ("t", "c", "z", "y", "x")


def parse_shard_size(target: str) -> int:
    """Parse a target size string into a number of bytes.

    Accepts SI (``"2GB"`` = 2e9) and binary (``"2GiB"`` = 2**31) suffixes, and
    a bare number as bytes. Case- and whitespace-insensitive.

    Parameters
    ----------
    target : str
        Size string, e.g. ``"2GB"``, ``"512 MiB"``, ``"1.5gb"``, ``"4096"``.

    Returns
    -------
    int
        Size in bytes.

    Raises
    ------
    ValueError
        If the string is not a number with an optional known suffix,
        or if it resolves to zero bytes.
    """
    match = _SIZE_PATTERN.match(target)
    if match is None:
        raise ValueError(
            f"Cannot parse shard size {target!r}. Expected a number with an optional "
            f"unit, e.g. '2GB', '512MiB', or '1048576'."
        )
    value, unit = match.group(1), (match.group(2) or "B").upper()
    nbytes = int(float(value) * _SIZE_UNITS[unit])
    if nbytes < 1:
        raise ValueError(f"Shard size {target!r} rounds to {nbytes} bytes, which cannot hold a chunk.")
    return nbytes


def resolve_shards(
    shards: ShardsLike,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: DTypeLike,
    dimension_names: Sequence[str] | None = None,
) -> tuple[int, ...] | None:
    """Resolve a shard specification into a chunk-aligned shard shape.

    Parameters
    ----------
    shards : tuple[int, ...] or str or None
        The intent to resolve:

        - ``None``: no sharding; returns None.
        - tuple of int: an explicit shard shape in array elements. Entries are
          rounded up to whole chunks, with a warning if that changes them.
          Unlike the derived forms below, an explicit shape is taken as given
          and may exceed the array.
        - ``"XY"`` or ``"XYZ"``: one shard per plane or per volume. The
          trailing 2 or 3 axes cover the whole array; every other axis stays
          at one chunk.
        - a size string such as ``"2GB"``: the largest chunk-aligned shard
          whose nominal (uncompressed) size does not exceed the target. Axes
          grow innermost-first -- X, then Y, then Z, then time -- and the
          channel axis is never grown. Compression is not accounted for, so
          the files on disk are typically smaller than the target.
    shape : tuple[int, ...]
        Array shape.
    chunks : tuple[int, ...]
        Chunk shape, same length as ``shape``.
    dtype : DTypeLike
        Array data type, used to convert a size target into element counts.
    dimension_names : sequence of str, optional
        Axis names, used to find the channel and time axes when growing into a
        size target. When None, iohub's TCZYX order is assumed, aligned to the
        end of the shape.

    Returns
    -------
    tuple[int, ...] or None
        Shard shape, or None when ``shards`` is None.

    Raises
    ------
    ValueError
        If lengths do not match, an explicit entry is not positive, an extent
        keyword needs more axes than the array has, or a string is neither a
        known keyword nor a parsable size.

    Examples
    --------
    One shard per ZYX volume, rounded up to whole chunks:

    >>> resolve_shards(
    ...     "XYZ",
    ...     shape=(5, 6, 86, 1664, 1193),
    ...     chunks=(1, 1, 16, 256, 256),
    ...     dtype="uint16",
    ... )
    (1, 1, 96, 1792, 1280)

    A ~2 GB target on the same array, growing time only once the spatial axes
    are full:

    >>> resolve_shards(
    ...     "2GB",
    ...     shape=(5, 6, 86, 1664, 1193),
    ...     chunks=(1, 1, 16, 256, 256),
    ...     dtype="uint16",
    ...     dimension_names=["t", "c", "z", "y", "x"],
    ... )
    (4, 1, 96, 1792, 1280)
    """
    if shards is None:
        return None
    if len(chunks) != len(shape):
        raise ValueError(f"Chunk shape {tuple(chunks)} does not match shape length {len(shape)}.")
    if any(c < 1 for c in chunks):
        raise ValueError(f"Chunk shape {tuple(chunks)} must be positive along every axis.")

    if not isinstance(shards, str):
        try:
            entries = tuple(int(s) for s in shards)
        except TypeError as err:
            raise TypeError(
                f"shards must be a shard shape, a spatial extent, or a size string, "
                f"got {type(shards).__name__}: {shards!r}."
            ) from err
        return _explicit_shards(entries, shape, chunks)

    keyword = frozenset(shards.upper())
    if keyword in _EXTENT_KEYWORDS:
        return _extent_shards(_EXTENT_KEYWORDS[keyword], shards, shape, chunks)
    try:
        target_bytes = parse_shard_size(shards)
    except ValueError as err:
        raise ValueError(
            f"Cannot interpret shards={shards!r}. Expected a shard shape, a spatial extent "
            f"({' or '.join(sorted(''.join(sorted(k)) for k in _EXTENT_KEYWORDS))}), "
            f"or a target size such as '2GB'."
        ) from err
    return _size_target_shards(target_bytes, shape, chunks, dtype, dimension_names)


def resolve_shard_shape(
    shards: ShardsLike,
    shards_ratio: tuple[int, ...] | None,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: DTypeLike,
    dimension_names: Sequence[str] | None = None,
    stacklevel: int = 3,
) -> tuple[int, ...] | None:
    """Resolve ``shards``, or the deprecated ``shards_ratio``, to a shard shape.

    The array-creation API accepts both; this folds the legacy per-axis chunk
    multiplier into the same result as ``shards`` and warns about it. Passing
    both is an error, since they would each ask for a different geometry.

    Parameters
    ----------
    shards : tuple[int, ...] or str or None
        Shard intent, see :func:`resolve_shards`.
    shards_ratio : tuple[int, ...], optional
        Deprecated per-axis chunk multiplier (``shards = chunks * ratio``).
    shape, chunks, dtype, dimension_names
        See :func:`resolve_shards`.
    stacklevel : int, optional
        ``stacklevel`` for the deprecation warning, counted from this
        function. Defaults to 3, which points at the caller of the public
        array-creation method.

    Returns
    -------
    tuple[int, ...] or None
        Shard shape, or None when neither argument requests sharding.
    """
    if shards_ratio:
        if shards is not None:
            raise ValueError(
                "Pass either shards or shards_ratio, not both. shards_ratio is deprecated; "
                "prefer shards, which also accepts a size target like '2GB' or an extent like 'XYZ'."
            )
        warnings.warn(
            "shards_ratio is deprecated and will be removed in a future release. Pass shards instead: "
            "an explicit shard shape, a spatial extent ('XY' or 'XYZ'), or a size target like '2GB'.",
            DeprecationWarning,
            stacklevel=stacklevel,
        )
        if len(shards_ratio) != len(shape):
            raise ValueError(f"Sharding ratio length {len(shards_ratio)} does not match shape length {len(shape)}.")
        if len(chunks) != len(shape):
            raise ValueError(f"Chunk shape {tuple(chunks)} does not match shape length {len(shape)}.")
        shards = tuple(c * r for c, r in zip(chunks, shards_ratio, strict=True))
    return resolve_shards(
        shards,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        dimension_names=dimension_names,
    )


def _full_chunks_extent(shape: tuple[int, ...], chunks: tuple[int, ...]) -> tuple[int, ...]:
    """Per-axis array size rounded up to a whole number of chunks.

    This is the largest shard shape worth writing: anything beyond it only
    pads the final shard of each axis with fill value.
    """
    return tuple(math.ceil(d / c) * c for d, c in zip(shape, chunks, strict=True))


def _explicit_shards(
    shards: tuple[int, ...],
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
) -> tuple[int, ...]:
    """Round an explicit shard shape up to whole chunks.

    An explicit shape is not capped at the array extent: a shard larger than
    the array only leaves inner chunks unwritten, and silently shrinking what
    the caller asked for would hide the mismatch.
    """
    if len(shards) != len(shape):
        raise ValueError(f"Shard shape length {len(shards)} does not match shape length {len(shape)}.")
    if any(s < 1 for s in shards):
        raise ValueError(f"Shard shape {shards} must be positive along every axis.")
    resolved = tuple(math.ceil(s / c) * c for s, c in zip(shards, chunks, strict=True))
    if resolved != shards:
        _logger.warning(f"Shard shape {shards} is not a whole number of {chunks} chunks; using {resolved}.")
    return resolved


def _extent_shards(
    n_axes: int,
    keyword: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
) -> tuple[int, ...]:
    """One shard per spatial extent: the trailing ``n_axes`` cover the array."""
    if len(shape) < n_axes:
        raise ValueError(f"shards={keyword!r} needs at least {n_axes} dimensions, but shape {shape} has {len(shape)}.")
    extent = _full_chunks_extent(shape, chunks)
    return tuple(chunks[:-n_axes]) + extent[-n_axes:]


def _growth_order(
    ndim: int,
    dimension_names: Sequence[str] | None,
) -> list[int]:
    """Axis indices in the order a size target grows them.

    Spatial axes come first, innermost (most contiguous on disk) first, so the
    budget buys a contiguous region before it buys timepoints. Time axes come
    last, and channel axes are left out entirely -- writers address single
    channels, so a shard spanning channels cannot be written a channel at a
    time.
    """
    if dimension_names is None:
        names = list(_DEFAULT_AXIS_NAMES[-ndim:] if ndim <= len(_DEFAULT_AXIS_NAMES) else _DEFAULT_AXIS_NAMES)
    else:
        names = [str(n).lower() for n in dimension_names][:ndim]
    # Unnamed leading axes are treated as spatial: growing them is wrong only
    # if they are really channels, and a caller with channels has names.
    names = [""] * (ndim - len(names)) + names
    spatial, time = [], []
    for axis in reversed(range(ndim)):
        if names[axis] in _CHANNEL_NAMES:
            continue
        if names[axis] in _TIME_NAMES:
            time.append(axis)
        else:
            spatial.append(axis)
    return spatial + time


def _size_target_shards(
    target_bytes: int,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: DTypeLike,
    dimension_names: Sequence[str] | None,
) -> tuple[int, ...]:
    """Largest chunk-aligned shard whose nominal size fits the byte target."""
    itemsize = np.dtype(dtype).itemsize
    extent = _full_chunks_extent(shape, chunks)
    chunk_counts = tuple(e // c for e, c in zip(extent, chunks, strict=True))

    shards = list(chunks)
    nbytes = math.prod(chunks) * itemsize
    if nbytes > target_bytes:
        _logger.warning(
            f"A single {chunks} chunk of {dtype} is {nbytes} bytes, over the {target_bytes} byte shard target; "
            f"using one chunk per shard. Pass smaller chunks for smaller shards."
        )
        return tuple(shards)

    # Growth is greedy, and greedy is exact here: an axis is only partially
    # filled once the remaining budget is under one more multiple of it, and
    # every later axis is at least that large.
    for axis in _growth_order(len(shape), dimension_names):
        multiple = min(chunk_counts[axis], target_bytes // nbytes)
        if multiple <= 1:
            continue
        shards[axis] = chunks[axis] * multiple
        nbytes *= multiple
    return tuple(shards)
