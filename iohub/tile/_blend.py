"""Sweep-line tile blending into a single xr.DataArray.

Thin wrapper around :func:`_sweep_line_assemble` that applies
weighted blending in overlap regions using :class:`Blender` kernels.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from iohub.tile._blenders import Blender
from iohub.tile._slicer import Slicer, TileSpec
from iohub.tile._sweep import _CellInfo, _sweep_line_assemble


def _weighted_blend(
    regions: list[xr.DataArray],
    weights: list[np.ndarray],
) -> np.ndarray:
    """Blend regions with weights via weighted mean.

    Parameters
    ----------
    regions : list[xr.DataArray]
        Tile sub-regions, all the same shape (..., Y, X).
    weights : list[np.ndarray]
        2D (Y, X) weight kernels, one per region.

    Returns
    -------
    np.ndarray
        Blended array with the same shape as each region.
    """
    stacked = np.stack([r.values for r in regions], axis=0)  # (N, T, C, Z, Y, X)
    weight_stack = np.stack(weights, axis=0)  # (N, Y, X)
    # Reshape for broadcasting: (N, 1, 1, 1, Y, X)
    ndim_extra = stacked.ndim - weight_stack.ndim
    for _ in range(ndim_extra):
        weight_stack = np.expand_dims(weight_stack, axis=1)
    numerator = (stacked * weight_stack).sum(axis=0)
    denominator = weight_stack.sum(axis=0)
    return numerator / denominator


def _blend_tiles(
    tiles: list[xr.DataArray],
    tile_specs: list[TileSpec],
    blender: Blender,
    slicer: Slicer,
) -> xr.DataArray:
    """Blend overlapping processed tiles into a single mosaic.

    Uses sweep-line decomposition + weighted reduction (same pattern
    as ``_composite_fovs``). Returns a lazy dask-backed xr.DataArray.

    Parameters
    ----------
    tiles : list[xr.DataArray]
        Processed tile data arrays, one per tile_spec.
    tile_specs : list[TileSpec]
        TileSpec objects from the Slicer (pixel-space positions).
    blender : Blender
        Blending strategy providing weight kernels.
    slicer : Slicer
        The Slicer that produced the tile_specs (for overlap info).

    Returns
    -------
    xr.DataArray
        Dask-backed mosaic with coordinates from the original data.
    """
    if len(tiles) != len(tile_specs):
        raise ValueError(f"tiles ({len(tiles)}) and tile_specs ({len(tile_specs)}) must have the same length")

    if len(tiles) == 1:
        return tiles[0]

    overlap = slicer.overlap
    data = slicer.data

    # Pixel-space bounding boxes from TileSpecs
    tile_bboxes = [(s.y_slice.start, s.y_slice.stop, s.x_slice.start, s.x_slice.stop) for s in tile_specs]

    # Weight cache: avoid recomputing kernels for same tile shape
    weight_cache: dict[tuple[int, int], np.ndarray] = {}

    def _get_weight(shape_yx: tuple[int, int]) -> np.ndarray:
        if shape_yx not in weight_cache:
            weight_cache[shape_yx] = blender.weights(shape_yx, overlap)
        return weight_cache[shape_yx]

    # Overlap callback: crop weight kernels and weighted-blend
    def _blend_overlap(
        cell_slices: list[xr.DataArray],
        contributing: list[int],
        info: _CellInfo,
    ) -> np.ndarray:
        cell_weights = []
        for idx in contributing:
            full_weight = _get_weight(tile_specs[idx].shape_yx)
            ty0, _, tx0, _ = tile_bboxes[idx]
            local_y = slice(info.y_start - ty0, info.y_end - ty0)
            local_x = slice(info.x_start - tx0, info.x_end - tx0)
            cell_weights.append(full_weight[local_y, local_x])
        return _weighted_blend(cell_slices, cell_weights)

    mosaic, (y_min, y_max, x_min, x_max) = _sweep_line_assemble(tiles, tile_bboxes, _blend_overlap)

    # Wrap in xarray with coordinates from original data
    mosaic_y = data.coords["y"].values[y_min:y_max]
    mosaic_x = data.coords["x"].values[x_min:x_max]

    return xr.DataArray(
        mosaic,
        dims=("t", "c", "z", "y", "x"),
        coords={
            "t": data.coords["t"],
            "c": data.coords["c"],
            "z": data.coords["z"],
            "y": ("y", mosaic_y, data.coords["y"].attrs),
            "x": ("x", mosaic_x, data.coords["x"].attrs),
        },
    )
