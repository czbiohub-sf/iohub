"""LayoutResolver protocol and implementations for FOV translation sources.

A LayoutResolver adjusts FOV xarray coordinates based on external
translation data (stitching YAML, position naming convention, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import xarray as xr
import yaml


@runtime_checkable
class LayoutResolver(Protocol):
    """Adjust FOV coordinates based on external translation sources."""

    def resolve(
        self,
        fov_xarrays: list[xr.DataArray],
        position_paths: list[str],
    ) -> list[xr.DataArray]:
        """Return FOV xarrays with updated YX coordinates.

        Parameters
        ----------
        fov_xarrays : list[xr.DataArray]
            FOV data arrays with existing coordinates.
        position_paths : list[str]
            Position paths within the zarr store (e.g. ``"000000"``).
        """
        ...


class TransformResolver:
    """No-op resolver — trusts existing OME-NGFF coordinateTransformations.

    Use this when Position metadata already has correct translations
    (i.e. ``Position.to_xarray()`` already produces correct coordinates).
    """

    def resolve(
        self,
        fov_xarrays: list[xr.DataArray],
        position_paths: list[str],
    ) -> list[xr.DataArray]:
        return fov_xarrays


class StitchingYAMLResolver:
    """Reads ZYX pixel translations from a stitching YAML config.

    YAML format::

        total_translation:
          well/row/posname:
          - z_pixels
          - y_pixels
          - x_pixels

    The resolver matches position paths against YAML keys and updates
    the xarray y/x coordinates accordingly.

    Parameters
    ----------
    yaml_path : str | Path
        Path to the stitching YAML file.
    well_path : str | None
        Well prefix for matching YAML keys (e.g. ``"0/1"``).
        If None, tries to infer from YAML keys.
    """

    def __init__(self, yaml_path: str | Path, well_path: str | None = None):
        self._yaml_path = Path(yaml_path)
        self._well_path = well_path
        with open(self._yaml_path) as f:
            self._config = yaml.safe_load(f)
        self._translations = self._config["total_translation"]

    def resolve(
        self,
        fov_xarrays: list[xr.DataArray],
        position_paths: list[str],
    ) -> list[xr.DataArray]:
        result = []
        for xa, pos_path in zip(fov_xarrays, position_paths):
            # Build lookup key: well_path/pos_path or just pos_path
            key = f"{self._well_path}/{pos_path}" if self._well_path else pos_path

            if key not in self._translations:
                # Try matching by position name suffix
                key = self._find_matching_key(pos_path)

            if key is None:
                raise KeyError(
                    f"Position {pos_path!r} not found in stitching YAML. "
                    f"Available keys: {list(self._translations.keys())[:5]}..."
                )

            z_px, y_px, x_px = self._translations[key]

            # Infer pixel size from coordinate spacing
            y_coords = xa.coords["y"].values
            x_coords = xa.coords["x"].values
            sy = float(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 1.0
            sx = float(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1.0

            # Build new coordinates with translation applied
            new_y = np.arange(len(y_coords)) * sy + y_px * sy
            new_x = np.arange(len(x_coords)) * sx + x_px * sx

            xa = xa.assign_coords(
                y=("y", new_y, xa.coords["y"].attrs),
                x=("x", new_x, xa.coords["x"].attrs),
            )
            result.append(xa)

        return result

    def _find_matching_key(self, pos_path: str) -> str | None:
        """Find a YAML key ending with the position path."""
        for key in self._translations:
            if key.endswith(f"/{pos_path}") or key == pos_path:
                return key
        return None
