"""Sweep-line FOV compositing into a single xr.DataArray.

Thin wrapper around :func:`_sweep_line_assemble` that handles
physical-to-pixel coordinate conversion and compositor dispatch.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from iohub.tile._compositors import CompositeContext, Compositor
from iohub.tile._sweep import _CellInfo, _sweep_line_assemble


def _composite_fovs(
    fov_xarrays: list[xr.DataArray],
    compositor: Compositor,
) -> xr.DataArray:
    """Composite N FOV xarrays into one mosaic xr.DataArray.

    Uses sweep-line decomposition to partition the mosaic into
    non-overlapping cells, composites overlaps via *compositor*,
    and assembles with ``dask.array.block()``.

    Parameters
    ----------
    fov_xarrays : list[xr.DataArray]
        FOV data arrays, each with physical y/x coordinates.
    compositor : Compositor
        Strategy for combining overlapping regions.

    Returns
    -------
    xr.DataArray
        Dask-backed mosaic with physical coordinates.
    """
    if len(fov_xarrays) == 1:
        return fov_xarrays[0]

    # Infer pixel size from first FOV's coordinate spacing
    y0 = fov_xarrays[0].coords["y"].values
    x0 = fov_xarrays[0].coords["x"].values
    sy = float(y0[1] - y0[0]) if len(y0) > 1 else 1.0
    sx = float(x0[1] - x0[0]) if len(x0) > 1 else 1.0

    # Derive pixel-space bounding boxes from physical coords
    fov_bboxes: list[tuple[int, int, int, int]] = []
    for xa in fov_xarrays:
        y_start = round(float(xa.coords["y"].values[0]) / sy)
        x_start = round(float(xa.coords["x"].values[0]) / sx)
        fov_bboxes.append(
            (
                y_start,
                y_start + xa.sizes["y"],
                x_start,
                x_start + xa.sizes["x"],
            )
        )

    # Overlap callback: build CompositeContext and delegate to compositor
    def _composite_overlap(
        cell_slices: list[xr.DataArray],
        contributing: list[int],
        info: _CellInfo,
    ) -> np.ndarray:
        ctx = CompositeContext(
            overlap_bbox=np.array([[info.y_start, info.y_end], [info.x_start, info.x_end]]),
            fov_bboxes=[
                np.array(
                    [
                        [fov_bboxes[idx][0], fov_bboxes[idx][1]],
                        [fov_bboxes[idx][2], fov_bboxes[idx][3]],
                    ]
                )
                for idx in contributing
            ],
        )
        return compositor.composite(cell_slices, masks=None, metadata=ctx)

    mosaic, (y_min, y_max, x_min, x_max) = _sweep_line_assemble(fov_xarrays, fov_bboxes, _composite_overlap)

    # Wrap in xarray with physical coordinates
    mosaic_y = np.arange(y_min, y_max) * sy
    mosaic_x = np.arange(x_min, x_max) * sx

    return xr.DataArray(
        mosaic,
        dims=("t", "c", "z", "y", "x"),
        coords={
            "t": fov_xarrays[0].coords["t"],
            "c": fov_xarrays[0].coords["c"],
            "z": fov_xarrays[0].coords["z"],
            "y": ("y", mosaic_y, fov_xarrays[0].coords["y"].attrs),
            "x": ("x", mosaic_x, fov_xarrays[0].coords["x"].attrs),
        },
    )
