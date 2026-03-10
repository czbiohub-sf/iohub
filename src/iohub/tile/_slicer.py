"""Tile generation for xr.DataArray with overlap support.

Produces TileSpec objects that lazily slice an xr.DataArray mosaic.
Inspired by xbatcher (issue #172 Slicer/Batcher decomposition) and
patchly's SamplingMode strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Iterator

import networkx as nx
import numpy as np
import xarray as xr

from iohub._experimental import experimental


class SamplingMode(Enum):
    """How to handle the last tile when the array doesn't divide evenly."""

    EDGE = "edge"
    """Align last tile to array edge (may increase overlap for last tile)."""

    SQUEEZE = "squeeze"
    """Redistribute overlap evenly across all tiles (default). From patchly."""

    CROP = "crop"
    """Discard tiles that extend beyond the array border."""


def _gen_slices_1d(
    dim_size: int,
    tile_size: int,
    overlap: int = 0,
    mode: SamplingMode = SamplingMode.SQUEEZE,
) -> list[int]:
    """Generate tile start positions for one dimension.

    Parameters
    ----------
    dim_size : int
        Size of the dimension in pixels.
    tile_size : int
        Size of each tile in pixels.
    overlap : int
        Overlap between adjacent tiles in pixels.
    mode : SamplingMode
        Border handling strategy.

    Returns
    -------
    list[int]
        Start positions for each tile.
    """
    if tile_size > dim_size:
        return [0]
    if overlap >= tile_size:
        raise ValueError(f"overlap ({overlap}) must be less than tile_size ({tile_size})")

    stride = tile_size - overlap
    positions = list(range(0, dim_size - tile_size + 1, stride))

    if mode == SamplingMode.SQUEEZE:
        last_end = positions[-1] + tile_size
        if last_end < dim_size:
            # Need one more tile; redistribute positions evenly
            n = len(positions) + 1
            max_start = dim_size - tile_size
            if n == 1:
                positions = [0]
            else:
                positions = [round(i * max_start / (n - 1)) for i in range(n)]
        elif last_end > dim_size and len(positions) > 1:
            # Squeeze existing positions so last tile fits exactly
            max_start = dim_size - tile_size
            n = len(positions)
            positions = [round(i * max_start / (n - 1)) for i in range(n)]

    elif mode == SamplingMode.EDGE:
        last_end = positions[-1] + tile_size
        if last_end < dim_size:
            positions.append(dim_size - tile_size)

    elif mode == SamplingMode.CROP:
        pass  # positions already only include full tiles

    return positions


def _ceil_to_multiple(value: int, multiple: int) -> int:
    """Round up to the nearest multiple."""
    return ((value + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class TileSpec:
    """Metadata for a single tile. Holds slices into the parent xr.DataArray.

    Use ``to_xarray()`` to get the tile data as a labeled xr.DataArray
    (dask-backed, lazy until ``.compute()``).
    """

    tile_id: int
    y_slice: slice
    x_slice: slice

    # Back-reference to the mosaic xarray, set by Slicer.
    # Excluded from repr/hash/eq (frozen dataclass compares by value fields only).
    _data: xr.DataArray

    def __init__(self, tile_id: int, y_slice: slice, x_slice: slice, data: xr.DataArray):
        # frozen dataclass workaround for non-comparable field
        object.__setattr__(self, "tile_id", tile_id)
        object.__setattr__(self, "y_slice", y_slice)
        object.__setattr__(self, "x_slice", x_slice)
        object.__setattr__(self, "_data", data)

    def to_xarray(self) -> xr.DataArray:
        """Slice the mosaic xarray for this tile.

        Returns a dask-backed DataArray with dims ("t", "c", "z", "y", "x")
        and physical coordinates (subset of the mosaic's global coords).
        """
        return self._data[..., self.y_slice, self.x_slice]

    @property
    def bbox(self) -> np.ndarray:
        """Bounding box as ``[[y_start, y_stop], [x_start, x_stop]]``."""
        return np.array(
            [
                [self.y_slice.start, self.y_slice.stop],
                [self.x_slice.start, self.x_slice.stop],
            ]
        )

    @property
    def shape_yx(self) -> tuple[int, int]:
        """Tile shape in (Y, X) pixels."""
        return (
            self.y_slice.stop - self.y_slice.start,
            self.x_slice.stop - self.x_slice.start,
        )

    def __repr__(self) -> str:
        return (
            f"TileSpec(tile_id={self.tile_id}, "
            f"y={self.y_slice.start}:{self.y_slice.stop}, "
            f"x={self.x_slice.start}:{self.x_slice.stop})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TileSpec):
            return NotImplemented
        return self.tile_id == other.tile_id and self.y_slice == other.y_slice and self.x_slice == other.x_slice

    def __hash__(self) -> int:
        return hash((self.tile_id, self.y_slice.start, self.y_slice.stop, self.x_slice.start, self.x_slice.stop))


@experimental
class Slicer:
    """Generate overlapping tiles from an xr.DataArray.

    Partitions the YX extent of the input DataArray into overlapping tiles.
    Iteration yields TileSpec objects (metadata-only). Use ``iter_xarrays()``
    to iterate over lazy xr.DataArrays instead.

    Parameters
    ----------
    data : xr.DataArray
        The mosaic to tile. Must have "y" and "x" dimensions.
    tile_size : dict[str, int]
        Tile size per dimension, e.g. ``{"y": 1024, "x": 1024}``.
    overlap : dict[str, int] | None
        Overlap between adjacent tiles, e.g. ``{"y": 128, "x": 128}``.
    mode : SamplingMode
        Border handling strategy. Default: SQUEEZE.
    align_to_chunks : bool
        If True, snap tile_size up to the nearest multiple of the data's
        zarr chunk size in Y and X. Prevents partial chunk reads.
    align_to_shards : bool
        If True and the zarr array has shards, align to shard boundaries
        instead of chunk boundaries. Takes precedence over align_to_chunks.

    Examples
    --------
    >>> pos = open_ome_zarr("deskewed.zarr", mode="r")
    >>> slicer = Slicer(pos.to_xarray(), tile_size={"y": 1024, "x": 1024})
    >>> len(slicer)
    15
    >>> for tile in slicer:
    ...     xa = tile.to_xarray()  # lazy xr.DataArray
    >>> for xa in slicer.iter_xarrays():
    ...     print(xa.shape)  # lazy xr.DataArray directly
    """

    def __init__(
        self,
        data: xr.DataArray,
        tile_size: dict[str, int],
        overlap: dict[str, int] | None = None,
        mode: SamplingMode = SamplingMode.SQUEEZE,
        align_to_chunks: bool = False,
        align_to_shards: bool = False,
    ):
        if "y" not in data.dims or "x" not in data.dims:
            raise ValueError(f"DataArray must have 'y' and 'x' dims, got {data.dims}")
        if "y" not in tile_size or "x" not in tile_size:
            raise ValueError(f"tile_size must specify 'y' and 'x', got {tile_size}")

        self._data = data
        self._original_tile_size = dict(tile_size)
        self._overlap = overlap or {}
        self._mode = mode
        self._align_to_chunks = align_to_chunks
        self._align_to_shards = align_to_shards

        # Chunk/shard alignment: snap tile_size up to nearest multiple
        tile_size = dict(tile_size)  # don't mutate caller's dict
        if (align_to_chunks or align_to_shards) and data.chunks is not None:
            y_idx = list(data.dims).index("y")
            x_idx = list(data.dims).index("x")
            y_chunk = data.chunks[y_idx][0]
            x_chunk = data.chunks[x_idx][0]
            tile_size["y"] = _ceil_to_multiple(tile_size["y"], y_chunk)
            tile_size["x"] = _ceil_to_multiple(tile_size["x"], x_chunk)

        self._tile_size = tile_size

        # Generate tile positions
        y_positions = _gen_slices_1d(
            dim_size=data.sizes["y"],
            tile_size=tile_size["y"],
            overlap=self._overlap.get("y", 0),
            mode=mode,
        )
        x_positions = _gen_slices_1d(
            dim_size=data.sizes["x"],
            tile_size=tile_size["x"],
            overlap=self._overlap.get("x", 0),
            mode=mode,
        )

        self._y_positions = y_positions
        self._x_positions = x_positions

        # Build TileSpecs from cartesian product of Y and X positions
        self._tiles: list[TileSpec] = []
        tile_id = 0
        for y_start, x_start in product(y_positions, x_positions):
            y_end = min(y_start + tile_size["y"], data.sizes["y"])
            x_end = min(x_start + tile_size["x"], data.sizes["x"])
            self._tiles.append(
                TileSpec(
                    tile_id=tile_id,
                    y_slice=slice(y_start, y_end),
                    x_slice=slice(x_start, x_end),
                    data=data,
                )
            )
            tile_id += 1

        self._graph: nx.Graph | None = None

    def __iter__(self) -> Iterator[TileSpec]:
        yield from self._tiles

    def iter_xarrays(self) -> Iterator[xr.DataArray]:
        """Iterate over tiles as lazy xr.DataArrays."""
        for tile in self._tiles:
            yield tile.to_xarray()

    def __len__(self) -> int:
        return len(self._tiles)

    def __getitem__(self, idx: int | slice) -> TileSpec | list[TileSpec]:
        return self._tiles[idx]

    def __repr__(self) -> str:
        ny = len(self._y_positions)
        nx_ = len(self._x_positions)
        size_str = f"tile_size={self._tile_size}"
        if self._tile_size != self._original_tile_size:
            size_str += f" (requested={self._original_tile_size})"
        return f"Slicer(tiles={len(self)}, grid={ny}x{nx_}, {size_str}, overlap={self._overlap})"

    @property
    def data(self) -> xr.DataArray:
        """The underlying mosaic xr.DataArray."""
        return self._data

    @property
    def overlap(self) -> dict[str, int]:
        """Overlap in pixels per dimension, e.g. ``{"y": 128, "x": 128}``."""
        return self._overlap

    @property
    def graph(self) -> nx.Graph:
        """Tile neighborhood graph. Nodes=tile_ids, edges=overlapping pairs.

        Built lazily on first access. Uses grid-based O(N) construction.
        """
        if self._graph is None:
            self._graph = self._build_neighborhood_graph()
        return self._graph

    @property
    def tile_grid_shape(self) -> tuple[int, int]:
        """Number of tiles in (Y, X)."""
        return (len(self._y_positions), len(self._x_positions))

    def _build_neighborhood_graph(self) -> nx.Graph:
        """Build tile neighborhood graph using grid-based O(N) lookup.

        Tiles are on a regular grid (from cartesian product of Y and X
        positions). Tile (r, c) is adjacent to (r+1, c) and (r, c+1).
        """
        G = nx.Graph()
        nx_ = len(self._x_positions)

        # Map grid (row, col) -> tile_id
        # Tile IDs are assigned in row-major order: id = row * nx + col
        for tile in self._tiles:
            G.add_node(tile.tile_id)

        for tile in self._tiles:
            row = tile.tile_id // nx_
            col = tile.tile_id % nx_

            # Right neighbor
            right_id = row * nx_ + (col + 1)
            if col + 1 < nx_ and self._overlap.get("x", 0) > 0:
                G.add_edge(tile.tile_id, right_id)

            # Down neighbor
            down_id = (row + 1) * nx_ + col
            if row + 1 < len(self._y_positions) and self._overlap.get("y", 0) > 0:
                G.add_edge(tile.tile_id, down_id)

        return G
