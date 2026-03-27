"""Pure numpy downsample utilities — no backend coupling."""

from __future__ import annotations

import itertools

import numpy as np


def downsample_block(data: np.ndarray, factors: list[int], method: str) -> np.ndarray:
    """Downsample an N-D numpy block by integer factors."""
    trim_slices = tuple(slice(0, (s // f) * f) for s, f in zip(data.shape, factors))
    data = data[trim_slices]

    if method == "stride":
        return data[tuple(slice(None, None, f) for f in factors)]

    new_shape = []
    reduce_axes = []
    for i, (s, f) in enumerate(zip(data.shape, factors)):
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

        orig_shape = tuple(s // f for s, f in zip(data.shape[::2], factors))
        flat = data.reshape(*orig_shape, -1)
        return stats.mode(flat, axis=-1, keepdims=False).mode.astype(data.dtype)
    else:
        raise ValueError(f"Unknown downsample method: {method!r}")


def target_region_to_source(
    target_region: tuple[slice, ...], factors: list[int], source_shape: tuple[int, ...]
) -> tuple[slice, ...]:
    """Map a target-space region back to the corresponding source region."""
    source_slices = []
    for sl, f, src_dim in zip(target_region, factors, source_shape):
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
    for total, step in zip(shape, step_shape):
        dim_ranges.append([slice(s, min(s + step, total)) for s in range(0, total, step)])
    return [tuple(r) for r in itertools.product(*dim_ranges)]
