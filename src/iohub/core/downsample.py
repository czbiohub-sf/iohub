"""Pure numpy downsample utilities — no backend coupling."""

from __future__ import annotations

import itertools

import numpy as np


def downsample_block(data: np.ndarray, factors: list[int], method: str) -> np.ndarray:
    """Downsample an N-D numpy block by integer factors.

    Axes whose size is not an exact multiple of the corresponding factor
    are edge-padded up to the next multiple before pooling, so the output
    shape is the ceiling division ``ceil(s / f)``. The trailing partial
    block on such an axis is averaged together with the boundary value,
    which matches the ceiling-division shape that ``initialize_pyramid``
    allocates for downstream pyramid levels.

    Parameters
    ----------
    data : numpy.ndarray
        Source block.
    factors : list of int
        Integer downsample factor per axis. Use 1 for axes that should pass
        through unchanged.
    method : str
        Aggregation method: ``"mean"``, ``"median"``, ``"min"``, ``"max"``,
        ``"mode"``, or ``"stride"``.

    Returns
    -------
    numpy.ndarray
        Downsampled block with shape ``tuple(ceil(s / f) for s, f in zip(data.shape, factors))``.
    """
    if method == "stride":
        return data[tuple(slice(None, None, f) for f in factors)]

    pad_widths = tuple((0, (-s) % f) for s, f in zip(data.shape, factors, strict=False))
    if any(after > 0 for _, after in pad_widths):
        data = np.pad(data, pad_widths, mode="edge")

    new_shape = []
    reduce_axes = []
    for i, (s, f) in enumerate(zip(data.shape, factors, strict=False)):
        new_shape.extend([s // f, f])
        reduce_axes.append(2 * i + 1)
    data = data.reshape(new_shape)

    reduce_axes = tuple(reduce_axes)
    if method == "mean":
        return data.mean(axis=reduce_axes).astype(data.dtype)
    elif method == "median":
        return np.median(data, axis=reduce_axes).astype(data.dtype)
    elif method == "min":
        return data.min(axis=reduce_axes)
    elif method == "max":
        return data.max(axis=reduce_axes)
    elif method == "mode":
        from scipy import stats

        orig_shape = tuple(s // f for s, f in zip(data.shape[::2], factors, strict=False))
        flat = data.reshape(*orig_shape, -1)
        return stats.mode(flat, axis=-1, keepdims=False).mode.astype(data.dtype)
    else:
        raise ValueError(f"Unknown downsample method: {method!r}")


def target_region_to_source(
    target_region: tuple[slice, ...], factors: list[int], source_shape: tuple[int, ...]
) -> tuple[slice, ...]:
    """Map a target-space region back to the corresponding source region."""
    source_slices = []
    for sl, f, src_dim in zip(target_region, factors, source_shape, strict=False):
        start = sl.start * f
        stop = min(sl.stop * f, src_dim)
        source_slices.append(slice(start, stop))
    return tuple(source_slices)


def iter_work_regions(shape: tuple[int, ...], step_shape: tuple[int, ...]) -> list[tuple[slice, ...]]:
    """Return shard/chunk-aligned regions covering the full array shape.

    Parameters
    ----------
    shape : tuple[int, ...]
        Total array shape.
    step_shape : tuple[int, ...]
        Step size per dimension (shard shape or chunk shape).

    Returns
    -------
    list[tuple[slice, ...]]
        One region (tuple of slices) per tile, covering the full array.
    """
    dim_ranges = []
    for total, step in zip(shape, step_shape, strict=False):
        dim_ranges.append([slice(s, min(s + step, total)) for s in range(0, total, step)])
    return [tuple(r) for r in itertools.product(*dim_ranges)]
