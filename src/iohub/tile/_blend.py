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
        Tile sub-regions, all the same shape (leading..., tiled...).
    weights : list[np.ndarray]
        N-D weight kernels over tiled dims, one per region.

    Returns
    -------
    np.ndarray
        Blended array with the same shape as each region.
    """
    stacked = np.stack([r.values for r in regions], axis=0)
    weight_stack = np.stack(weights, axis=0)
    # Reshape for broadcasting: add leading dim axes after the stack axis
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
    tile_dims = slicer.tile_dims

    # Pixel-space bounding boxes from TileSpecs as dict[str, (start, stop)]
    tile_bboxes: list[dict[str, tuple[int, int]]] = [
        {d: (s.slices[d].start, s.slices[d].stop) for d in tile_dims} for s in tile_specs
    ]

    # Weight cache: avoid recomputing kernels for same tile shape
    weight_cache: dict[tuple[int, ...], np.ndarray] = {}

    def _get_weight(shape: tuple[int, ...]) -> np.ndarray:
        if shape not in weight_cache:
            weight_cache[shape] = blender.weights(shape, overlap)
        return weight_cache[shape]

    # Overlap callback: crop weight kernels and weighted-blend
    def _blend_overlap(
        cell_slices: list[xr.DataArray],
        contributing: list[int],
        info: _CellInfo,
    ) -> np.ndarray:
        cell_weights = []
        for idx in contributing:
            full_weight = _get_weight(tile_specs[idx].tile_shape)
            # Crop weight to the cell's local region within this tile
            local_slices = tuple(
                slice(
                    info.bounds[d][0] - tile_bboxes[idx][d][0],
                    info.bounds[d][1] - tile_bboxes[idx][d][0],
                )
                for d in tile_dims
            )
            cell_weights.append(full_weight[local_slices])
        return _weighted_blend(cell_slices, cell_weights)

    mosaic, global_bounds = _sweep_line_assemble(tiles, tile_bboxes, _blend_overlap, tile_dims)

    # Wrap in xarray with coordinates from original data
    all_dims = tuple(str(d) for d in data.dims)
    coords = {}
    for d in all_dims:
        if d in global_bounds:
            dmin, dmax = global_bounds[d]
            coords[d] = (d, data.coords[d].values[dmin:dmax], data.coords[d].attrs)
        elif d in data.coords:
            coords[d] = data.coords[d]

    return xr.DataArray(
        mosaic,
        dims=all_dims,
        coords=coords,
    )
