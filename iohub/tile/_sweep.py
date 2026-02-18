"""Shared sweep-line decomposition for mosaic assembly.

Partitions a set of overlapping rectangular regions into a non-overlapping
cell grid via sweep-line decomposition, then assembles them into a single
dask array with ``da.block()``.

Used by both ``_composite_fovs`` (FOV stitching) and ``_blend_tiles``
(tile blending) — the only difference is the overlap handler callback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import dask
import dask.array as da
import numpy as np
import xarray as xr


@dataclass(frozen=True, slots=True)
class _CellInfo:
    """Pixel-space bounds of a single cell in the sweep-line grid."""

    y_start: int
    y_end: int
    x_start: int
    x_end: int


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
    bboxes: list[tuple[int, int, int, int]],
    overlap_fn: OverlapHandler,
) -> tuple[da.Array, tuple[int, int, int, int]]:
    """Sweep-line decomposition + ``da.block()`` assembly.

    Parameters
    ----------
    regions : list[xr.DataArray]
        Input data arrays (FOVs or processed tiles).
    bboxes : list[tuple[int, int, int, int]]
        Pixel-space bounding boxes ``(y_start, y_end, x_start, x_end)``,
        one per region.
    overlap_fn : OverlapHandler
        Handler for cells with 2+ contributors. Receives the cell slices,
        contributing indices, and cell bounds. Must return an np.ndarray.

    Returns
    -------
    tuple[da.Array, tuple[int, int, int, int]]
        ``(mosaic_dask, (global_y_min, global_y_max, global_x_min, global_x_max))``.
        Callers wrap the dask array in ``xr.DataArray`` with their own coords.
    """
    # Global bounds
    global_y_min = min(b[0] for b in bboxes)
    global_y_max = max(b[1] for b in bboxes)
    global_x_min = min(b[2] for b in bboxes)
    global_x_max = max(b[3] for b in bboxes)

    # Sweep-line: unique Y/X edges → cell grid
    y_edges = sorted({coord for b in bboxes for coord in (b[0], b[1])})
    x_edges = sorted({coord for b in bboxes for coord in (b[2], b[3])})

    n_rows = len(y_edges) - 1
    n_cols = len(x_edges) - 1

    # Leading dims from first region (assumed identical across all)
    first = regions[0]
    T = first.sizes["t"]
    C = first.sizes["c"]
    Z = first.sizes["z"]
    dtype = first.dtype

    # Build block grid
    block_grid: list[list[da.Array]] = []
    for i in range(n_rows):
        block_row: list[da.Array] = []
        cell_h = y_edges[i + 1] - y_edges[i]

        for j in range(n_cols):
            cell_w = x_edges[j + 1] - x_edges[j]

            # Which regions fully cover this cell?
            contributing: list[int] = []
            for idx, (y0, y1, x0, x1) in enumerate(bboxes):
                if y0 <= y_edges[i] and y1 >= y_edges[i + 1] and x0 <= x_edges[j] and x1 >= x_edges[j + 1]:
                    contributing.append(idx)

            cell_shape = (T, C, Z, cell_h, cell_w)

            if len(contributing) == 0:
                # Gap: no coverage
                block = da.full(cell_shape, np.nan, dtype=dtype)

            elif len(contributing) == 1:
                # Interior: direct slice from single region (lazy)
                region = regions[contributing[0]]
                ry0 = bboxes[contributing[0]][0]
                rx0 = bboxes[contributing[0]][2]
                local_y = slice(y_edges[i] - ry0, y_edges[i + 1] - ry0)
                local_x = slice(x_edges[j] - rx0, x_edges[j + 1] - rx0)
                block = region.isel(y=local_y, x=local_x).data

            else:
                # Overlap: delegate to caller's callback
                cell_slices: list[xr.DataArray] = []
                for idx in contributing:
                    region = regions[idx]
                    ry0 = bboxes[idx][0]
                    rx0 = bboxes[idx][2]
                    local_y = slice(y_edges[i] - ry0, y_edges[i + 1] - ry0)
                    local_x = slice(x_edges[j] - rx0, x_edges[j + 1] - rx0)
                    cell_slices.append(region.isel(y=local_y, x=local_x))

                info = _CellInfo(
                    y_start=y_edges[i],
                    y_end=y_edges[i + 1],
                    x_start=x_edges[j],
                    x_end=x_edges[j + 1],
                )
                # NOTE: dask.delayed captures cell_slices as opaque objects —
                # dask cannot track their inner dask-array dependencies.
                # Works with local/threaded scheduler; for dask.distributed
                # replace with da.map_blocks or explicit graph wiring.
                result = dask.delayed(overlap_fn)(cell_slices, contributing, info)
                block = da.from_delayed(result, shape=cell_shape, dtype=dtype)

            block_row.append(block)
        block_grid.append(block_row)

    mosaic = da.block(block_grid)
    bounds = (global_y_min, global_y_max, global_x_min, global_x_max)
    return mosaic, bounds
