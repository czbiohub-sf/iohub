"""Sweep-line FOV compositing into a single xr.DataArray.

Thin wrapper around :func:`_sweep_line_assemble` that handles
physical-to-pixel coordinate conversion and compositor dispatch.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from iohub.tile._compositors import CompositeContext, Compositor
from iohub.tile._sweep import _CellInfo, _sweep_line_assemble


def _pixel_spacing(coords: np.ndarray) -> float:
    """Infer pixel spacing from coordinate array."""
    if len(coords) > 1:
        return float(coords[1] - coords[0])
    return 1.0


def _composite_fovs(
    fov_xarrays: list[xr.DataArray],
    compositor: Compositor,
) -> xr.DataArray:
    """Composite N FOV xarrays into one mosaic xr.DataArray.

    Uses sweep-line decomposition to partition the mosaic into
    non-overlapping cells, composites overlaps via *compositor*,
    and assembles with ``dask.array.block()``.

    Compositing dimensions are determined automatically: Y and X are
    always composited; Z is included when FOVs have Z coordinates
    that vary across FOVs.

    Parameters
    ----------
    fov_xarrays : list[xr.DataArray]
        FOV data arrays, each with physical coordinates.
    compositor : Compositor
        Strategy for combining overlapping regions.

    Returns
    -------
    xr.DataArray
        Dask-backed mosaic with physical coordinates.
    """
    if len(fov_xarrays) == 1:
        return fov_xarrays[0]

    first = fov_xarrays[0]

    # Determine which spatial dims to composite over.
    # Y and X always; Z only if FOVs have varying Z coordinates.
    composite_dims: list[str] = []
    spacings: dict[str, float] = {}

    for dim in ("z", "y", "x"):
        if dim not in first.dims or dim not in first.coords:
            continue
        spacing = _pixel_spacing(first.coords[dim].values)
        if dim in ("y", "x"):
            # Always composite over Y and X
            composite_dims.append(dim)
            spacings[dim] = spacing
        elif dim == "z":
            # Only composite over Z if FOVs have different Z origins
            z_origins = {float(xa.coords["z"].values[0]) for xa in fov_xarrays}
            if len(z_origins) > 1:
                composite_dims.append(dim)
                spacings[dim] = spacing

    tile_dims = tuple(composite_dims)

    # Derive pixel-space bounding boxes from physical coords
    fov_bboxes: list[dict[str, tuple[int, int]]] = []
    for xa in fov_xarrays:
        bbox: dict[str, tuple[int, int]] = {}
        for dim in tile_dims:
            s = spacings[dim]
            start = round(float(xa.coords[dim].values[0]) / s)
            bbox[dim] = (start, start + xa.sizes[dim])
        fov_bboxes.append(bbox)

    # Overlap callback: build CompositeContext and delegate to compositor
    def _composite_overlap(
        cell_slices: list[xr.DataArray],
        contributing: list[int],
        info: _CellInfo,
    ) -> np.ndarray:
        ctx = CompositeContext(
            overlap_bounds=info.bounds,
            fov_bounds=[fov_bboxes[idx] for idx in contributing],
        )
        return compositor.composite(cell_slices, masks=None, metadata=ctx)

    mosaic, global_bounds = _sweep_line_assemble(fov_xarrays, fov_bboxes, _composite_overlap, tile_dims)

    # Wrap in xarray with physical coordinates
    all_dims = tuple(str(d) for d in first.dims)

    coords: dict = {}
    for d in all_dims:
        if d in global_bounds:
            dmin, dmax = global_bounds[d]
            s = spacings[d]
            coords[d] = (d, np.arange(dmin, dmax) * s, first.coords[d].attrs)
        elif d in first.coords:
            coords[d] = first.coords[d]

    return xr.DataArray(
        mosaic,
        dims=all_dims,
        coords=coords,
    )
