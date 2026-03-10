"""Assembler — write processed tiles back to zarr with overlap blending.

Implements the weighted accumulation pattern (from patchly's Aggregator):

    output[bbox] += tile_data * weight
    weight_map[bbox] += weight
    result = output / weight_map

Both accumulator and weight map are zarr arrays on disk, enabling
out-of-core reassembly and concurrent writes from multiple workers.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
import zarr

from iohub._experimental import experimental
from iohub.tile._blenders import Blender, get_blender
from iohub.tile._slicer import Slicer, TileSpec


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
    """Reassemble processed tiles into a zarr array with overlap blending.

    Parameters
    ----------
    slicer : Slicer
        The slicer used to generate the tiles. Provides mosaic shape
        and overlap metadata.
    output : str | Path | zarr.Group
        Path for the output zarr array, or an open ``zarr.Group``
        in which ``"data"``, ``"accum"``, and ``"weights"`` arrays
        will be created.
    weights : str | Blender
        Blending strategy. Built-in: ``"gaussian"`` (default), ``"uniform"``.
    dtype : np.dtype | None
        Output dtype. Defaults to float32.
    chunks : dict[str, int] | None
        Chunk sizes for the output zarr, keyed by dim name.
        Defaults to the slicer's data chunk sizes.
    shards : dict[str, int] | None
        Shard sizes for the output zarr, keyed by dim name.
        When provided, multiple chunks are grouped into larger
        shard files — useful for HPC parallel filesystems.
    """

    def __init__(
        self,
        slicer: Slicer,
        output: str | Path | zarr.Group,
        weights: str | Blender = "gaussian",
        dtype: np.dtype | None = None,
        chunks: dict[str, int] | None = None,
        shards: dict[str, int] | None = None,
    ):
        self._slicer = slicer
        self._blender = get_blender(weights)
        self._dtype = np.dtype(dtype) if dtype is not None else np.dtype("float32")
        self._finalized = False
        self._result: xr.DataArray | None = None

        # Full output dims from the data (e.g. ("t", "c", "z", "y", "x"))
        data = slicer.data
        self._dims = tuple(str(d) for d in data.dims)
        self._shape = tuple(data.sizes[d] for d in self._dims)
        self._tile_dims = slicer.tile_dims

        self._chunks = _resolve_chunks(self._dims, self._shape, data, chunks)

        self._shards = None
        if shards is not None:
            self._shards = tuple(shards.get(d, c) for d, c in zip(self._dims, self._chunks))

        # Store output reference: either a zarr.Group or a Path
        if isinstance(output, zarr.Group):
            self._output_group: zarr.Group | None = output
            self._output_path: Path | None = None
        else:
            self._output_group = None
            self._output_path = Path(output)

        # Pre-allocate accumulator and weight map zarr arrays (float64 for precision)
        if self._output_group is not None:
            accum_store: str | zarr.Group = self._output_group
            weight_store: str | zarr.Group = self._output_group
            accum_name = "_accum"
            weight_name = "_weights"
        else:
            accum_store = str(self._output_path.parent / (self._output_path.name + ".accum"))
            weight_store = str(self._output_path.parent / (self._output_path.name + ".weights"))
            accum_name = None
            weight_name = None

        self._accum = _create_scratch_zarr(
            accum_store,
            accum_name,
            shape=self._shape,
            chunks=self._chunks,
            dtype=np.float64,
            fill_value=0.0,
            shards=self._shards,
        )
        self._weight_map = _create_scratch_zarr(
            weight_store,
            weight_name,
            shape=self._shape,
            chunks=self._chunks,
            dtype=np.float64,
            fill_value=0.0,
            shards=self._shards,
        )

        # Cache weight kernels by tile shape
        self._weight_cache: dict[tuple[int, ...], np.ndarray] = {}

        self._overlap = slicer.overlap

    def validate_parallel_safety(self) -> bool:
        """Check if tiles can be safely processed in parallel.

        Returns True if no two tiles write to overlapping zarr chunks.
        When False, tiles must be processed in waves (see Slicer.graph
        for coloring-based wave scheduling).
        """
        # Map dim name -> chunk size for tiled dimensions
        dim_chunk_sizes = {d: self._chunks[self._dims.index(d)] for d in self._tile_dims}

        seen_chunks: set[tuple[int, ...]] = set()

        for tile in self._slicer:
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
            self._weight_cache[tile_shape] = self._blender.weights(tile_shape, self._overlap)
        return self._weight_cache[tile_shape]

    def append(self, tile: TileSpec, result: xr.DataArray | np.ndarray):
        """Write a processed tile into the accumulator.

        Parameters
        ----------
        tile : TileSpec
            The tile spec (provides spatial slices into the mosaic).
        result : xr.DataArray | np.ndarray
            Processed tile data. Shape must match tile extent
            (broadcast dimensions are supported).
        """
        if self._finalized:
            raise RuntimeError("Assembler already finalized. Create a new Assembler to write more tiles.")

        data = result.values if isinstance(result, xr.DataArray) else result
        data = data.astype(np.float64)

        # Weight kernel covers tiled dims, broadcast over non-tiled leading dims
        weight = self._get_weight_kernel(tile.tile_shape)

        # Build index: slice for tiled dims, slice(None) for others
        idx = tuple(tile.slices[d] if d in tile.slices else slice(None) for d in self._dims)

        existing_accum = self._accum[idx]
        self._accum[idx] = existing_accum + data * weight

        existing_weight = self._weight_map[idx]
        self._weight_map[idx] = existing_weight + weight

    def get_output(self) -> xr.DataArray:
        """Finalize: normalize accumulator by weight map and return result.

        Processes chunk-by-chunk to avoid loading the full array into memory.
        Idempotent — safe to call multiple times.

        Returns
        -------
        xr.DataArray
            Result array backed by the output zarr, with physical coordinates
            from the source mosaic.
        """
        if self._finalized and self._result is not None:
            return self._result

        # Create output zarr
        if self._output_group is not None:
            output = _create_scratch_zarr(
                self._output_group,
                "data",
                shape=self._shape,
                chunks=self._chunks,
                dtype=self._dtype,
                fill_value=0,
                shards=self._shards,
            )
        else:
            output = _create_scratch_zarr(
                str(self._output_path),
                None,
                shape=self._shape,
                chunks=self._chunks,
                dtype=self._dtype,
                fill_value=0,
                shards=self._shards,
            )

        # Normalize chunk-by-chunk to avoid OOM
        chunk_slices = self._iter_chunk_slices()
        for slices in chunk_slices:
            accum_chunk = self._accum[slices]
            weight_chunk = self._weight_map[slices]

            # Avoid division by zero: where weight is 0, output is 0
            with np.errstate(divide="ignore", invalid="ignore"):
                normalized = np.where(
                    weight_chunk > 0,
                    accum_chunk / weight_chunk,
                    0.0,
                )
            output[slices] = normalized.astype(self._dtype)

        # Build xr.DataArray with coordinates from source mosaic
        source = self._slicer.data
        coords = {d: source.coords[d].values for d in self._dims if d in source.coords}

        dask_data = da.from_zarr(output)
        self._result = xr.DataArray(
            data=dask_data,
            dims=self._dims,
            coords=coords,
        )
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
