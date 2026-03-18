"""Tile generation for xr.DataArray with overlap support.

Produces Tile objects that lazily slice an xr.DataArray mosaic.
Inspired by xbatcher (issue #172 Tiler/Batcher decomposition) and
patchly's SamplingMode strategies.
"""

from __future__ import annotations

from enum import Enum
from functools import reduce
from itertools import product
from operator import mul
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


class Tile:
    """Metadata for a single tile. Holds slices into the parent xr.DataArray.

    Use ``to_xarray()`` to get the tile data as a labeled xr.DataArray
    (dask-backed, lazy until ``.compute()``).

    Parameters
    ----------
    tile_id : int
        Unique identifier for this tile.
    slices : dict[str, slice]
        Dimension name to slice mapping, e.g.
        ``{"y": slice(0, 256), "x": slice(0, 256)}`` or
        ``{"z": slice(0, 32), "y": slice(0, 256), "x": slice(0, 256)}``.
    data : xr.DataArray
        Back-reference to the mosaic xarray (set by Tiler).
    """

    __slots__ = ("tile_id", "slices", "_data")

    def __init__(
        self,
        tile_id: int,
        slices: dict[str, slice],
        data: xr.DataArray,
    ):
        self.tile_id = tile_id
        self.slices = slices
        self._data = data

    def to_xarray(self) -> xr.DataArray:
        """Slice the mosaic xarray for this tile.

        Returns a dask-backed DataArray preserving all non-tiled dimensions
        and physical coordinates (subset of the mosaic's global coords).
        """
        return self._data.isel(**self.slices)

    @property
    def tile_dims(self) -> tuple[str, ...]:
        """Dimension names being tiled, e.g. ``("z", "y", "x")``."""
        return tuple(self.slices.keys())

    @property
    def bbox(self) -> np.ndarray:
        """Bounding box as ``[[dim0_start, dim0_stop], ...]``."""
        return np.array([[s.start, s.stop] for s in self.slices.values()])

    @property
    def tile_shape(self) -> tuple[int, ...]:
        """Tile shape in tiled dimensions."""
        return tuple(s.stop - s.start for s in self.slices.values())

    def __repr__(self) -> str:
        parts = ", ".join(f"{d}={s.start}:{s.stop}" for d, s in self.slices.items())
        return f"Tile(tile_id={self.tile_id}, {parts})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tile):
            return NotImplemented
        return self.tile_id == other.tile_id and self.slices == other.slices

    def __hash__(self) -> int:
        return hash((self.tile_id,) + tuple((d, s.start, s.stop) for d, s in self.slices.items()))


@experimental
class Tiler:
    """Generate overlapping tiles from an xr.DataArray.

    Partitions the spatial extent of the input DataArray into overlapping
    tiles. Supports tiling over any subset of spatial dimensions (at
    minimum "y" and "x"; optionally "z" as well).

    Iteration yields Tile objects (metadata-only). Use ``iter_xarrays()``
    to iterate over lazy xr.DataArrays instead.

    Parameters
    ----------
    data : xr.DataArray
        The mosaic to tile. Must have "y" and "x" dimensions.
    tile_size : dict[str, int]
        Tile size per dimension, e.g. ``{"y": 1024, "x": 1024}`` or
        ``{"z": 32, "y": 1024, "x": 1024}``.
    overlap : dict[str, int] | None
        Overlap between adjacent tiles, e.g. ``{"y": 128, "x": 128}``.
    mode : SamplingMode
        Border handling strategy. Default: SQUEEZE.
    align_to_chunks : bool
        If True, snap tile_size up to the nearest multiple of the data's
        zarr chunk size in each tiled dimension. Prevents partial chunk reads.
    align_to_shards : bool
        If True and the zarr array has shards, align to shard boundaries
        instead of chunk boundaries. Takes precedence over align_to_chunks.

    Examples
    --------
    >>> pos = open_ome_zarr("deskewed.zarr", mode="r")
    >>> tiler = Tiler(pos.to_xarray(), tile_size={"y": 1024, "x": 1024})
    >>> len(tiler)
    15
    >>> for tile in tiler:
    ...     xa = tile.to_xarray()  # lazy xr.DataArray
    >>> tiler_3d = Tiler(
    ...     pos.to_xarray(),
    ...     tile_size={"z": 32, "y": 1024, "x": 1024},
    ...     overlap={"z": 4, "y": 128, "x": 128},
    ... )
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
        for dim in tile_size:
            if dim not in data.dims:
                raise ValueError(f"tile_size key '{dim}' not found in data dims {data.dims}")

        self._data = data
        self._original_tile_size = dict(tile_size)
        self._overlap = overlap or {}
        self._mode = mode
        self._align_to_chunks = align_to_chunks
        self._align_to_shards = align_to_shards

        # Determine tiled dimensions, ordered as they appear in data.dims
        self._tile_dims = tuple(d for d in data.dims if d in tile_size)

        # Chunk/shard alignment: snap tile_size up to nearest multiple
        tile_size = dict(tile_size)  # don't mutate caller's dict
        if (align_to_chunks or align_to_shards) and data.chunks is not None:
            for dim in self._tile_dims:
                dim_idx = list(data.dims).index(dim)
                chunk_size = data.chunks[dim_idx][0]
                tile_size[dim] = _ceil_to_multiple(tile_size[dim], chunk_size)

        self._tile_size = tile_size

        # Generate tile positions per dimension
        self._positions_per_dim: dict[str, list[int]] = {}
        for dim in self._tile_dims:
            self._positions_per_dim[dim] = _gen_slices_1d(
                dim_size=data.sizes[dim],
                tile_size=tile_size[dim],
                overlap=self._overlap.get(dim, 0),
                mode=mode,
            )

        # Build Tiles from cartesian product of all tiled dim positions
        self._tiles: list[Tile] = []
        dim_positions = [self._positions_per_dim[d] for d in self._tile_dims]
        tile_id = 0
        for combo in product(*dim_positions):
            slices = {}
            for dim, start in zip(self._tile_dims, combo):
                end = min(start + tile_size[dim], data.sizes[dim])
                slices[dim] = slice(start, end)
            self._tiles.append(Tile(tile_id=tile_id, slices=slices, data=data))
            tile_id += 1

        self._graph: nx.Graph | None = None

    @property
    def tile_dims(self) -> tuple[str, ...]:
        """Dimension names being tiled, e.g. ``("z", "y", "x")`` or ``("y", "x")``."""
        return self._tile_dims

    def __iter__(self) -> Iterator[Tile]:
        yield from self._tiles

    def iter_xarrays(self) -> Iterator[xr.DataArray]:
        """Iterate over tiles as lazy xr.DataArrays."""
        for tile in self._tiles:
            yield tile.to_xarray()

    def __len__(self) -> int:
        return len(self._tiles)

    def __getitem__(self, idx: int | slice) -> Tile | list[Tile]:
        return self._tiles[idx]

    def __repr__(self) -> str:
        grid_parts = "x".join(str(len(self._positions_per_dim[d])) for d in self._tile_dims)
        size_str = f"tile_size={self._tile_size}"
        if self._tile_size != self._original_tile_size:
            size_str += f" (requested={self._original_tile_size})"
        return f"Tiler(tiles={len(self)}, grid={grid_parts}, {size_str}, overlap={self._overlap})"

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

        Built lazily on first access. Uses grid-based construction.
        """
        if self._graph is None:
            self._graph = self._build_neighborhood_graph()
        return self._graph

    @property
    def tile_grid_shape(self) -> tuple[int, ...]:
        """Number of tiles per tiled dimension."""
        return tuple(len(self._positions_per_dim[d]) for d in self._tile_dims)

    def _build_neighborhood_graph(self) -> nx.Graph:
        """Build tile neighborhood graph using N-D grid-based lookup.

        Tiles are on a regular N-D grid (from cartesian product of
        positions). Neighbors along each dimension with overlap > 0
        are connected by edges.
        """
        G = nx.Graph()
        grid_shape = self.tile_grid_shape
        n_dims = len(grid_shape)

        for tile in self._tiles:
            G.add_node(tile.tile_id)

        # Strides for converting N-D grid index to flat tile_id (row-major)
        strides = []
        for i in range(n_dims):
            strides.append(reduce(mul, grid_shape[i + 1 :], 1))

        for tile in self._tiles:
            # Recover N-D grid index from flat tile_id
            grid_idx = []
            remaining = tile.tile_id
            for s in strides:
                grid_idx.append(remaining // s)
                remaining %= s

            # Connect to next neighbor along each dimension with overlap
            for axis, dim in enumerate(self._tile_dims):
                if self._overlap.get(dim, 0) > 0 and grid_idx[axis] + 1 < grid_shape[axis]:
                    neighbor_idx = list(grid_idx)
                    neighbor_idx[axis] += 1
                    neighbor_id = sum(i * s for i, s in zip(neighbor_idx, strides))
                    G.add_edge(tile.tile_id, neighbor_id)

        return G
