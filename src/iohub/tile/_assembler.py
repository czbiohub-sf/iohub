"""Assembler — write processed tiles back to zarr with overlap blending.

Implements the weighted accumulation pattern (from patchly's Aggregator):

    output[bbox] += tile_data * weight
    weight_map[bbox] += weight
    result = output / weight_map

Both accumulator and weight map are zarr arrays on disk, enabling
out-of-core reassembly and concurrent writes from multiple workers.
"""

from __future__ import annotations

import logging
import threading
from itertools import product
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

from iohub._experimental import experimental
from iohub.ngff import open_ome_zarr
from iohub.tile._blenders import Blender, get_blender
from iohub.tile._tiler import Tile, Tiler

logger = logging.getLogger(__name__)


def _resolve_chunks(
    dims: tuple[str, ...],
    shape: tuple[int, ...],
    data: xr.DataArray,
    chunks: dict[str, int] | None,
) -> tuple[int, ...]:
    """Resolve zarr chunk sizes from explicit dict, source data, or array shape."""
    if chunks is not None:
        return tuple(chunks.get(d, s) for d, s in zip(dims, shape))
    if data.chunks is not None:
        return tuple(data.chunks[data.dims.index(d)][0] if d in data.dims else s for d, s in zip(dims, shape))
    return shape


def _create_scratch_zarr(
    store: str | zarr.Group,
    name: str | None,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: np.dtype,
    fill_value: float,
    shards: tuple[int, ...] | None = None,
) -> zarr.Array:
    """Create a temporary zarr array for accumulation scratch space.

    TODO: Consider using iohub's zarr utilities instead of raw zarr APIs.
    """
    if isinstance(store, zarr.Group):
        return store.create_array(
            name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            fill_value=fill_value,
            shards=shards,
        )
    if shards is not None:
        from zarr.codecs import ShardingCodec

        return zarr.open_array(
            store,
            mode="w",
            shape=shape,
            chunks=shards,
            dtype=dtype,
            fill_value=fill_value,
            codecs=[ShardingCodec(chunk_shape=chunks)],
        )
    return zarr.open_array(
        store,
        mode="w",
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        fill_value=fill_value,
    )


@experimental
class Assembler:
    """Reassemble processed tiles into an OME-Zarr with overlap blending.

    Parameters
    ----------
    tiler : Tiler
        The tiler used to generate the tiles. Provides mosaic shape
        and overlap metadata.
    output : str | Path
        Path for the output OME-Zarr store.
    source_position : Position
        Source Position node — metadata (channel names, scale transforms)
        is copied to the output OME-Zarr.
    weights : str | Blender
        Blending strategy. Built-in: ``"gaussian"`` (default), ``"uniform"``.
    dtype : np.dtype | None
        Output dtype. Defaults to the input data's dtype.
    chunks : dict[str, int] | None
        Chunk sizes for the output zarr, keyed by dim name.
        Defaults to the tiler's data chunk sizes.
    shards : dict[str, int] | None
        Shard sizes for the output zarr, keyed by dim name.
        When provided, multiple chunks are grouped into larger
        shard files — useful for HPC parallel filesystems.
    """

    def __init__(
        self,
        tiler: Tiler,
        output: str | Path,
        source_position,
        weights: str | Blender = "gaussian",
        dtype: np.dtype | None = None,
        chunks: dict[str, int] | None = None,
        shards: dict[str, int] | None = None,
    ):
        self._tiler = tiler
        self._blender = get_blender(weights)
        data = tiler.data
        self._dtype = np.dtype(dtype) if dtype is not None else data.dtype
        # Accumulator needs at least float32 precision for weighted sums
        self._accum_dtype = np.float32 if self._dtype.itemsize < 4 else self._dtype
        self._finalized = False
        self._result: xr.DataArray | None = None
        self._source_position = source_position

        # Full output dims from the data (e.g. ("t", "c", "z", "y", "x"))
        self._dims = tuple(str(d) for d in data.dims)
        self._shape = tuple(data.sizes[d] for d in self._dims)
        self._tile_dims = tiler.tile_dims

        self._chunks = _resolve_chunks(self._dims, self._shape, data, chunks)

        self._shards = None
        if shards is not None:
            self._shards = tuple(shards.get(d, c) for d, c in zip(self._dims, self._chunks))

        self._output_path = Path(output)

        accum_store = str(self._output_path.parent / (self._output_path.name + ".accum"))
        weight_store = str(self._output_path.parent / (self._output_path.name + ".weights"))

        self._accum = _create_scratch_zarr(
            accum_store,
            None,
            shape=self._shape,
            chunks=self._chunks,
            dtype=self._accum_dtype,
            fill_value=0.0,
            shards=self._shards,
        )
        self._weight_map = _create_scratch_zarr(
            weight_store,
            None,
            shape=self._shape,
            chunks=self._chunks,
            dtype=self._accum_dtype,
            fill_value=0.0,
            shards=self._shards,
        )

        # Cache weight kernels by tile shape
        self._weight_cache: dict[tuple[int, ...], np.ndarray] = {}

        self._overlap = tiler.overlap

        # Lock for thread-safe append (protects read-modify-write on zarr chunks)
        self._lock = threading.Lock()

    def validate_parallel_safety(self) -> bool:
        """Check if tiles can be safely processed in parallel.

        Returns True if no two tiles write to overlapping zarr chunks.
        When False, tiles must be processed in waves (see Tiler.graph
        for coloring-based wave scheduling).
        """
        # Map dim name -> chunk size for tiled dimensions
        dim_chunk_sizes = {d: self._chunks[self._dims.index(d)] for d in self._tile_dims}

        seen_chunks: set[tuple[int, ...]] = set()

        for tile in self._tiler:
            # Compute chunk indices touched by this tile in each tiled dim
            chunk_ranges = []
            for dim in self._tile_dims:
                s = tile.slices[dim]
                cs = dim_chunk_sizes[dim]
                chunk_ranges.append(range(s.start // cs, (s.stop - 1) // cs + 1))

            tile_chunks = set(product(*chunk_ranges))

            if tile_chunks & seen_chunks:
                return False
            seen_chunks |= tile_chunks

        return True

    def _get_weight_kernel(self, tile_shape: tuple[int, ...]) -> np.ndarray:
        """Get or compute the weight kernel for a tile shape."""
        if tile_shape not in self._weight_cache:
            w = self._blender.weights(tile_shape, self._overlap)
            self._weight_cache[tile_shape] = w.astype(self._accum_dtype)
        return self._weight_cache[tile_shape]

    def append(self, tile: Tile, result: xr.DataArray | np.ndarray):
        """Write a processed tile into the accumulator.

        Parameters
        ----------
        tile : Tile
            The tile spec (provides spatial slices into the mosaic).
        result : xr.DataArray | np.ndarray
            Processed tile data. Shape must match tile extent
            (broadcast dimensions are supported).
        """
        if self._finalized:
            raise RuntimeError("Assembler already finalized. Create a new Assembler to write more tiles.")

        data = result.values if isinstance(result, xr.DataArray) else result
        data = data.astype(self._accum_dtype)

        # Weight kernel covers tiled dims, broadcast over non-tiled leading dims
        weight = self._get_weight_kernel(tile.tile_shape)

        # Build index: slice for tiled dims, slice(None) for others
        idx = tuple(tile.slices[d] if d in tile.slices else slice(None) for d in self._dims)

        # Compute weighted data before acquiring lock
        weighted_data = data * weight

        # Thread-safe read-modify-write on zarr chunks
        with self._lock:
            existing_accum = self._accum[idx]
            self._accum[idx] = existing_accum + weighted_data

            existing_weight = self._weight_map[idx]
            self._weight_map[idx] = existing_weight + weight

    def get_output(self) -> xr.DataArray:
        """Finalize: normalize and write to an OME-Zarr output store.

        Creates a proper OME-Zarr with metadata copied from the source
        Position. Processes chunk-by-chunk to avoid OOM.
        Idempotent — safe to call multiple times.

        Returns
        -------
        xr.DataArray
            Result backed by the output OME-Zarr.
        """
        if self._finalized and self._result is not None:
            return self._result

        from copy import deepcopy

        src = self._source_position

        # Create output OME-Zarr with metadata from source
        dst = open_ome_zarr(
            str(self._output_path),
            layout="fov",
            mode="w-",
            channel_names=list(src.channel_names),
        )
        src_transforms = deepcopy(src.metadata.multiscales[0].datasets[0].coordinate_transformations)
        dst.create_zeros(
            "0",
            shape=self._shape,
            dtype=self._dtype,
            chunks=self._chunks,
        )
        dst.metadata.multiscales[0].datasets[0].coordinate_transformations = src_transforms
        dst.dump_meta()

        output = dst["0"]

        # Normalize chunk-by-chunk to avoid OOM
        all_chunk_slices = list(self._iter_chunk_slices())
        n_chunks = len(all_chunk_slices)
        logger.info("Normalizing %d chunks...", n_chunks)
        for i, slices in enumerate(all_chunk_slices):
            accum_chunk = self._accum[slices]
            weight_chunk = self._weight_map[slices]

            with np.errstate(divide="ignore", invalid="ignore"):
                normalized = np.where(
                    weight_chunk > 0,
                    accum_chunk / weight_chunk,
                    0.0,
                )
            output[slices] = normalized.astype(self._dtype)

            if (i + 1) % 100 == 0 or i + 1 == n_chunks:
                logger.info("  Normalized %d/%d chunks", i + 1, n_chunks)

        self._result = dst.to_xarray()
        self._finalized = True
        return self._result

    def _iter_chunk_slices(self):
        """Yield tuple-of-slices for each chunk in the zarr array."""
        ranges = []
        for dim_size, chunk_size in zip(self._shape, self._chunks):
            starts = list(range(0, dim_size, chunk_size))
            ranges.append([slice(s, min(s + chunk_size, dim_size)) for s in starts])

        for combo in product(*ranges):
            yield tuple(combo)
