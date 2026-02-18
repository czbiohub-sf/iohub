"""Blender protocol and built-in implementations for tile overlap blending.

A Blender produces spatial weight kernels that control how overlapping tiles
are combined during reassembly. The weighted accumulation pattern
(from patchly's Aggregator):

    output[bbox] += tile_data * weight
    weight_map[bbox] += weight
    result = output / weight_map

Built-in strategies are resolved by name; third-party strategies are
discoverable via the ``iohub.blenders`` entrypoint group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from iohub.tile._registry import resolve_strategy
from iohub.tile._slicer import TileSpec


@dataclass
class BlendContext:
    """Tile metadata passed to blenders."""

    tile_spec: TileSpec
    """The tile being blended."""

    neighbors: list[int]
    """Tile IDs of neighboring (overlapping) tiles."""

    is_edge: bool
    """Whether this tile touches the mosaic border."""


@runtime_checkable
class Blender(Protocol):
    """Produce spatial weight kernels for tile overlap blending."""

    def weights(
        self,
        tile_shape: tuple[int, int],
        overlap: dict[str, int],
        metadata: BlendContext | None = None,
    ) -> np.ndarray:
        """Return a 2D (Y, X) weight array for the given tile shape.

        Parameters
        ----------
        tile_shape : tuple[int, int]
            (Y, X) size of the tile.
        overlap : dict[str, int]
            Overlap in pixels, e.g. ``{"y": 128, "x": 128}``.
        metadata : BlendContext | None
            Optional tile context (neighbors, edge status).

        Returns
        -------
        np.ndarray
            2D float64 weight array with shape ``tile_shape``.
        """
        ...


class UniformBlender:
    """Uniform weights — simple averaging in overlap regions."""

    def weights(
        self,
        tile_shape: tuple[int, int],
        overlap: dict[str, int],
        metadata: BlendContext | None = None,
    ) -> np.ndarray:
        return np.ones(tile_shape, dtype=np.float64)


class GaussianBlender:
    """Gaussian weight kernel via separable 1D gaussians.

    Center-weighted blending: tiles contribute most from their centers
    and least from their edges, producing smooth transitions in overlap
    regions. Default sigma is tile_size / 8 (per patchly convention).
    """

    def __init__(self, sigma_fraction: float = 1.0 / 8.0):
        self._sigma_fraction = sigma_fraction

    def weights(
        self,
        tile_shape: tuple[int, int],
        overlap: dict[str, int],
        metadata: BlendContext | None = None,
    ) -> np.ndarray:
        wy = self._gaussian_1d(tile_shape[0])
        wx = self._gaussian_1d(tile_shape[1])
        return np.outer(wy, wx)

    def _gaussian_1d(self, size: int) -> np.ndarray:
        """1D gaussian centered in the array."""
        sigma = size * self._sigma_fraction
        center = (size - 1) / 2.0
        x = np.arange(size, dtype=np.float64)
        g = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        return g


class DistanceBlender:
    """Euclidean distance transform weights with cosine ramp.

    Each pixel's weight is proportional to its distance from the nearest
    tile edge, with a cosine falloff for smooth transitions. Inspired by
    multiview-stitcher's EDT blending and Preibisch et al. (2009).

    Produces smoother transitions than Gaussian blending because the
    weight profile adapts to the tile shape rather than assuming a
    fixed bell curve.
    """

    def weights(
        self,
        tile_shape: tuple[int, int],
        overlap: dict[str, int],
        metadata: BlendContext | None = None,
    ) -> np.ndarray:
        # Separable 1D distance ramps with cosine profile.
        # Each pixel's weight = product of Y and X ramp values.
        # Ramp goes from ~0 at the edge to 1 at the center.
        wy = self._cosine_ramp_1d(tile_shape[0])
        wx = self._cosine_ramp_1d(tile_shape[1])
        return np.outer(wy, wx)

    @staticmethod
    def _cosine_ramp_1d(size: int) -> np.ndarray:
        """1D cosine ramp: small at edges, 1 at center."""
        if size <= 1:
            return np.ones(size, dtype=np.float64)
        # Distance from nearest edge, in [0.5, center] then normalized to (0, 1]
        # The 0.5 offset ensures edge pixels get nonzero weight
        dist = np.minimum(np.arange(size, dtype=np.float64), np.arange(size - 1, -1, -1, dtype=np.float64))
        dist = (dist + 0.5) / (size / 2.0)
        dist = np.clip(dist, 0.0, 1.0)
        # Cosine ramp
        return (1.0 - np.cos(np.pi * dist)) / 2.0


_BUILTINS: dict[str, type] = {
    "uniform": UniformBlender,
    "gaussian": GaussianBlender,
    "distance": DistanceBlender,
}


def get_blender(name: str | Blender) -> Blender:
    """Resolve a blender by name or pass through an object.

    Checks built-in names first, then ``iohub.blenders`` entrypoints.
    """
    return resolve_strategy(name, _BUILTINS, "iohub.blenders", "blender")
