"""ZarrPython implementation -- consolidates all zarr internal imports."""

from __future__ import annotations

from typing import Any

import numpy as np
import zarr

from iohub.core.config import ZarrConfig
from iohub.core.downsample import downsample_block, iter_work_regions, target_region_to_source
from iohub.core.protocol import ZarrImplementation
from iohub.core.specs import ArraySpec
from iohub.core.types import StorePath


class ZarrPythonImplementation(ZarrImplementation[zarr.Group, zarr.Array]):
    """Zarr-python backed I/O implementation."""

    def __init__(self, config: ZarrConfig | None = None):
        self.config = config or ZarrConfig()
        # Apply codec pipeline config globally — zarr-python reads this at I/O time,
        # not at group-open time, so it must persist beyond the open_group call.
        zarr.config.set(
            {
                "codec_pipeline.path": self.config.codec_pipeline,
                "codec_pipeline.validate_checksums": self.config.validate_checksums,
            }
        )

    # -- Group operations --------------------------------------------------

    def open_group(self, path: StorePath, mode: str, zarr_format: int | None = None) -> zarr.Group:
        return zarr.open_group(path, mode=mode, zarr_format=zarr_format)

    def group_keys(self, group: zarr.Group) -> list[str]:
        return sorted(group.group_keys())

    def array_keys(self, group: zarr.Group) -> list[str]:
        return sorted(group.array_keys())

    def close(self, group: zarr.Group) -> None:
        group.store.close()

    def get_zarr_format(self, group: zarr.Group) -> int:
        return group.metadata.zarr_format

    # -- Array lifecycle ---------------------------------------------------

    def create_array(self, group: zarr.Group, name: str, spec: ArraySpec, *, overwrite: bool = False) -> zarr.Array:
        path = f"{group.path}/{name}" if group.path != "/" else name
        outer_chunks = tuple(spec.chunk_grid["configuration"]["chunk_shape"])
        return zarr.open_array(
            group.store,
            path=path,
            mode="w" if overwrite else "w-",
            shape=spec.shape,
            dtype=spec.data_type,
            chunks=outer_chunks,
            chunk_key_encoding=spec.chunk_key_encoding,
            codecs=spec.codecs,
            fill_value=spec.fill_value,
            dimension_names=spec.dimension_names,
            attributes=spec.attributes,
            zarr_format=3,
        )

    def create_array_v2(
        self,
        group: zarr.Group,
        name: str,
        *,
        shape: tuple[int, ...],
        dtype: np.dtype,
        chunks: tuple[int, ...],
        fill_value: int = 0,
        overwrite: bool = False,
    ) -> zarr.Array:
        from numcodecs import Blosc

        cname = self.config.compressor.cname
        clevel = self.config.compressor.clevel
        shuffle_map = {
            "noshuffle": Blosc.NOSHUFFLE,
            "shuffle": Blosc.SHUFFLE,
            "bitshuffle": Blosc.BITSHUFFLE,
        }
        shuffle = shuffle_map.get(self.config.compressor.shuffle, Blosc.BITSHUFFLE)
        return group.create_array(
            name=name,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            overwrite=overwrite,
            fill_value=fill_value,
            chunk_key_encoding={"name": "v2", "separator": "/"},
            compressor=Blosc(cname=cname, clevel=clevel, shuffle=shuffle),
        )

    def open_array(self, group: zarr.Group, name: str) -> zarr.Array:
        return group[name]

    # -- Array I/O ---------------------------------------------------------

    def read(self, handle: zarr.Array, selection: Any) -> np.ndarray:
        return np.asarray(handle[selection])

    def write(self, handle: zarr.Array, selection: Any, data: np.ndarray) -> None:
        handle[selection] = data

    def read_oindex(self, handle: zarr.Array, selection: Any) -> np.ndarray:
        return np.asarray(handle.oindex[selection])

    def write_oindex(self, handle: zarr.Array, selection: Any, data: np.ndarray) -> None:
        handle.oindex[selection] = data

    def resize(self, handle: zarr.Array, new_shape: tuple[int, ...]) -> None:
        handle.resize(new_shape)

    # -- Array metadata ----------------------------------------------------

    def get_shape(self, handle: zarr.Array) -> tuple[int, ...]:
        return handle.shape

    def get_dtype(self, handle: zarr.Array) -> np.dtype:
        return handle.dtype

    def get_chunks(self, handle: zarr.Array) -> tuple[int, ...]:
        return handle.chunks

    def get_shards(self, handle: zarr.Array) -> tuple[int, ...] | None:
        return getattr(handle, "shards", None)

    # -- Conversions -------------------------------------------------------

    def to_dask(self, handle: zarr.Array) -> Any:
        import dask.array as da

        return da.from_zarr(handle)

    def write_from_dask(self, handle: zarr.Array, dask_array: Any) -> None:
        from dask.array import to_zarr

        to_zarr(dask_array, handle)

    # -- High-performance operations ---------------------------------------

    def downsample(
        self,
        source: zarr.Array,
        target: zarr.Array,
        factors: list[int],
        method: str = "mean",
    ) -> None:
        """Downsample source into target sequentially (shard-by-shard)."""
        for region in self.iter_work_regions(target):
            self.downsample_region(source, target, region, factors, method)

    def downsample_region(
        self,
        source: zarr.Array,
        target: zarr.Array,
        target_region: tuple[slice, ...],
        factors: list[int],
        method: str = "mean",
    ) -> None:
        """Read, downsample, and write a single shard-aligned region."""
        source_region = target_region_to_source(target_region, factors, source.shape)
        source_data = np.asarray(source[source_region])
        local_factors = []
        for sl_src, sl_tgt, f in zip(source_region, target_region, factors, strict=False):
            src_size = sl_src.stop - sl_src.start
            tgt_size = sl_tgt.stop - sl_tgt.start
            local_factors.append(max(1, src_size // tgt_size) if tgt_size > 0 else f)
        downsampled = downsample_block(source_data, local_factors, method)
        target[target_region] = downsampled

    def iter_work_regions(self, target: zarr.Array) -> list[tuple[slice, ...]]:
        """Return shard/chunk-aligned regions for parallel iteration."""
        step_shape = self.get_shards(target) or target.chunks
        return iter_work_regions(target.shape, step_shape)
