"""Compositor protocol and built-in implementations for FOV overlap compositing.

A Compositor controls how overlapping FOV regions are combined when
``_composite_fovs()`` builds the mosaic dask graph. Built-in strategies
are resolved by name; third-party strategies are discoverable via the
``iohub.compositors`` entrypoint group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import xarray as xr

from iohub.tile._registry import resolve_strategy


@dataclass
class CompositeContext:
    """Context passed to compositors about the overlap region."""

    overlap_bbox: np.ndarray
    """(2, 2) array: ``[[y_start, y_stop], [x_start, x_stop]]`` in pixel space."""

    fov_bboxes: list[np.ndarray]
    """Each contributing FOV's full bbox in pixel space."""


@runtime_checkable
class Compositor(Protocol):
    """Combine data from overlapping FOVs into a single output region."""

    def composite(
        self,
        regions: list[xr.DataArray],
        masks: list[np.ndarray] | None,
        metadata: CompositeContext,
    ) -> np.ndarray: ...


class MeanCompositor:
    """Simple average in overlap regions (default)."""

    def composite(
        self,
        regions: list[xr.DataArray],
        masks: list[np.ndarray] | None,
        metadata: CompositeContext,
    ) -> np.ndarray:
        stacked = xr.concat(regions, dim="__fov__")
        return stacked.mean(dim="__fov__").values


class MaxCompositor:
    """Maximum intensity projection across FOVs."""

    def composite(
        self,
        regions: list[xr.DataArray],
        masks: list[np.ndarray] | None,
        metadata: CompositeContext,
    ) -> np.ndarray:
        stacked = xr.concat(regions, dim="__fov__")
        return stacked.max(dim="__fov__").values


class FirstCompositor:
    """First FOV wins (no blending). Fastest."""

    def composite(
        self,
        regions: list[xr.DataArray],
        masks: list[np.ndarray] | None,
        metadata: CompositeContext,
    ) -> np.ndarray:
        return regions[0].values


_BUILTINS: dict[str, type] = {
    "mean": MeanCompositor,
    "max": MaxCompositor,
    "first": FirstCompositor,
}


def get_compositor(name: str | Compositor) -> Compositor:
    """Resolve a compositor by name or pass through an object.

    Checks built-in names first, then ``iohub.compositors`` entrypoints.
    """
    return resolve_strategy(name, _BUILTINS, "iohub.compositors", "compositor")
