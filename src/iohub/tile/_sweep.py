"""Shared sweep-line decomposition for mosaic assembly.

Partitions a set of overlapping rectangular regions into a non-overlapping
cell grid via sweep-line decomposition, then assembles them into a single
dask array with ``da.block()``.

Used by both ``_composite_fovs`` (FOV stitching) and ``_blend_tiles``
(tile blending) — the only difference is the overlap handler callback.

Supports N-D tiling dimensions (e.g. YX or ZYX). Non-tiled leading
dimensions (e.g. T, C) are passed through unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Protocol, runtime_checkable

import dask
import dask.array as da
import numpy as np
import xarray as xr


@dataclass(frozen=True, slots=True)
class _CellInfo:
    """Pixel-space bounds of a single cell in the sweep-line grid."""

    bounds: dict[str, tuple[int, int]]


@runtime_checkable
class OverlapHandler(Protocol):
    """Resolve overlap where 2+ regions cover the same cell.

    Implementations receive the cropped sub-regions, the indices of the
    contributing regions (into the original list), and the cell's
    pixel-space bounds. They must return the blended/composited result
    as a numpy array with the same shape as each sub-region.
    """

    def __call__(
        self,
        cell_slices: list[xr.DataArray],
        contributing: list[int],
        info: _CellInfo,
    ) -> np.ndarray: ...


def _sweep_line_assemble(
    regions: list[xr.DataArray],
    bboxes: list[dict[str, tuple[int, int]]],
    overlap_fn: OverlapHandler,
    tile_dims: tuple[str, ...],
) -> tuple[da.Array, dict[str, tuple[int, int]]]:
    """N-D sweep-line decomposition + ``da.block()`` assembly.

    Parameters
    ----------
    regions : list[xr.DataArray]
        Input data arrays (FOVs or processed tiles).
    bboxes : list[dict[str, tuple[int, int]]]
        Pixel-space bounding boxes per region, keyed by dim name.
        e.g. ``{"y": (0, 256), "x": (0, 512)}`` or
        ``{"z": (0, 32), "y": (0, 256), "x": (0, 512)}``.
    overlap_fn : OverlapHandler
        Handler for cells with 2+ contributors.
    tile_dims : tuple[str, ...]
        Ordered tiling dimensions, e.g. ``("y", "x")`` or ``("z", "y", "x")``.

    Returns
    -------
    tuple[da.Array, dict[str, tuple[int, int]]]
        ``(mosaic_dask, global_bounds)`` where global_bounds maps each
        tiled dim to ``(min, max)``.
    """
    # Global bounds per tiled dimension
    global_bounds: dict[str, tuple[int, int]] = {}
    for dim in tile_dims:
        dim_min = min(b[dim][0] for b in bboxes)
        dim_max = max(b[dim][1] for b in bboxes)
        global_bounds[dim] = (dim_min, dim_max)

    # Compute unique edges per tiled dimension
    edges_per_dim: dict[str, list[int]] = {}
    for dim in tile_dims:
        edges = sorted({coord for b in bboxes for coord in (b[dim][0], b[dim][1])})
        edges_per_dim[dim] = edges

    # Number of cells per dimension
    n_cells_per_dim = {dim: len(edges) - 1 for dim, edges in edges_per_dim.items()}

    # Leading dims: all dims from the first region that are NOT tiled
    first = regions[0]
    leading_dims = [d for d in first.dims if d not in tile_dims]
    leading_shape = tuple(first.sizes[d] for d in leading_dims)
    dtype = first.dtype

    # Build N-D block grid
    # We iterate over all cell indices (cartesian product of per-dim cell ranges)
    dim_cell_ranges = [range(n_cells_per_dim[d]) for d in tile_dims]

    # We'll build a flat dict mapping cell index tuple -> da.Array block,
    # then reshape into nested lists for da.block()
    blocks: dict[tuple[int, ...], da.Array] = {}

    for cell_idx in product(*dim_cell_ranges):
        # Cell bounds for each tiled dim
        cell_bounds: dict[str, tuple[int, int]] = {}
        cell_spatial_shape: list[int] = []
        for i, dim in enumerate(tile_dims):
            start = edges_per_dim[dim][cell_idx[i]]
            end = edges_per_dim[dim][cell_idx[i] + 1]
            cell_bounds[dim] = (start, end)
            cell_spatial_shape.append(end - start)

        # Which regions fully cover this cell?
        contributing: list[int] = []
        for idx, bbox in enumerate(bboxes):
            covers = all(
                bbox[dim][0] <= cell_bounds[dim][0] and bbox[dim][1] >= cell_bounds[dim][1] for dim in tile_dims
            )
            if covers:
                contributing.append(idx)

        cell_shape = leading_shape + tuple(cell_spatial_shape)

        if len(contributing) == 0:
            block = da.full(cell_shape, np.nan, dtype=dtype)

        elif len(contributing) == 1:
            region = regions[contributing[0]]
            local_slices = {
                dim: slice(
                    cell_bounds[dim][0] - bboxes[contributing[0]][dim][0],
                    cell_bounds[dim][1] - bboxes[contributing[0]][dim][0],
                )
                for dim in tile_dims
            }
            block = region.isel(**local_slices).data

        else:
            cell_slices: list[xr.DataArray] = []
            for idx in contributing:
                region = regions[idx]
                local_slices = {
                    dim: slice(
                        cell_bounds[dim][0] - bboxes[idx][dim][0],
                        cell_bounds[dim][1] - bboxes[idx][dim][0],
                    )
                    for dim in tile_dims
                }
                cell_slices.append(region.isel(**local_slices))

            info = _CellInfo(bounds=cell_bounds)
            result = dask.delayed(overlap_fn)(cell_slices, contributing, info)
            block = da.from_delayed(result, shape=cell_shape, dtype=dtype)

        blocks[cell_idx] = block

    # Reshape flat dict into nested list structure for da.block()
    grid_shape = tuple(n_cells_per_dim[d] for d in tile_dims)
    mosaic = _build_nested_block(blocks, grid_shape, depth=0)
    mosaic = da.block(mosaic)

    return mosaic, global_bounds


def _build_nested_block(
    blocks: dict[tuple[int, ...], da.Array],
    grid_shape: tuple[int, ...],
    depth: int,
    prefix: tuple[int, ...] = (),
):
    """Recursively build nested list structure for da.block() from flat dict."""
    if depth == len(grid_shape) - 1:
        # Innermost dimension: return a list of blocks
        return [blocks[prefix + (i,)] for i in range(grid_shape[depth])]
    else:
        return [_build_nested_block(blocks, grid_shape, depth + 1, prefix + (i,)) for i in range(grid_shape[depth])]
