"""Pure numpy downsample utilities — no backend coupling."""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray


def downsample_block(data: NDArray, factors: list[int], method: str) -> NDArray:
    """Downsample an N-D numpy block by integer factors.

    Axes whose size is not an exact multiple of the corresponding factor
    are edge-padded up to the next multiple before pooling, so the output
    shape is the ceiling division ``ceil(s / f)``. The trailing partial
    block on such an axis is averaged together with the boundary value,
    which matches the ceiling-division shape that ``initialize_pyramid``
    allocates for downstream pyramid levels.

    Parameters
    ----------
    data : NDArray
        Source block.
    factors : list of int
        Integer downsample factor per axis. Use 1 for axes that should pass
        through unchanged.
    method : str
        Aggregation method: ``"mean"``, ``"median"``, ``"min"``, ``"max"``,
        ``"mode"``, or ``"stride"``.

    Returns
    -------
    NDArray
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

        # After the reshape above the even axes hold the pooled shape and the
        # odd ones the pooling windows. Move the window axes to the end before
        # flattening them, so each output element sees its own window rather
        # than a memory-order mix of neighbours.
        pooled_axes = tuple(range(0, data.ndim, 2))
        windows = np.transpose(data, pooled_axes + reduce_axes)
        flat = windows.reshape(*windows.shape[: len(pooled_axes)], -1)
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


def split_region(region: tuple[slice, ...], factors: list[int], step_shape: tuple[int, ...]) -> list[tuple[slice, ...]]:
    """Split a target region into sub-blocks that downsample independently.

    An axis that is not downsampled (factor 1, e.g. T and C) pools nothing
    across its indices, so the region splits along it at ``step_shape``
    (the target's chunk shape) without changing the result. A downsampled
    axis keeps the region's full extent, so its edge padding is the same as
    for the whole region.

    Parameters
    ----------
    region : tuple[slice, ...]
        Target region, with explicit start and stop on every axis.
    factors : list[int]
        Downsample factor per axis.
    step_shape : tuple[int, ...]
        Sub-block step per axis; only used on axes with factor 1.

    Returns
    -------
    list[tuple[slice, ...]]
        Sub-blocks tiling ``region``, in C order.
    """
    dim_ranges = []
    for sl, f, step in zip(region, factors, step_shape, strict=True):
        if f == 1:
            dim_ranges.append([slice(s, min(s + step, sl.stop)) for s in range(sl.start, sl.stop, step)])
        else:
            dim_ranges.append([sl])
    return [tuple(r) for r in itertools.product(*dim_ranges)]


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
