"""TensorStore implementation -- zarr-python groups + TensorStore array I/O."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import zarr

from iohub.core.config import TensorStoreConfig
from iohub.core.protocol import ZarrImplementation
from iohub.core.specs import ArraySpec
from iohub.core.types import StorePath

try:
    import tensorstore as ts

    _TS_AVAILABLE = True
except ImportError:
    ts = None  # type: ignore[assignment]
    _TS_AVAILABLE = False

if TYPE_CHECKING:
    import tensorstore as ts


def _fill_value_for_spec(data_type: str, fill_value: int | float) -> object:
    """Return a TensorStore-compatible fill value for the given dtype string."""
    if data_type == "bool":
        return bool(fill_value)
    if data_type in ("complex64", "complex128"):
        # TensorStore requires [real, imag] for complex fill values
        return [float(fill_value), 0.0]
    return fill_value


def _ts_open(spec: dict, **kwargs) -> ts.TensorStore:
    """Thin wrapper that converts TensorStore errors to standard Python exceptions."""
    import tensorstore as ts

    try:
        return ts.open(spec, **kwargs).result()
    except Exception as exc:
        msg = str(exc).lower()
        if "not found" in msg or "no such file" in msg:
            raise FileNotFoundError(str(exc)) from exc
        if "already exists" in msg:
            raise FileExistsError(str(exc)) from exc
        raise RuntimeError(f"TensorStore error: {exc}") from exc


def _spec_to_ts(spec: ArraySpec, path: str) -> dict:
    fill_value = _fill_value_for_spec(spec.data_type, spec.fill_value)
    metadata: dict = {
        "shape": list(spec.shape),
        "data_type": spec.data_type,
        "chunk_grid": spec.chunk_grid,
        "chunk_key_encoding": spec.chunk_key_encoding,
        "codecs": spec.codecs,
        "fill_value": fill_value,
    }
    if spec.dimension_names:
        metadata["dimension_names"] = spec.dimension_names
    return {"driver": "zarr3", "kvstore": {"driver": "file", "path": path}, "metadata": metadata}


def _resolve_array_path(group: zarr.Group, name: str) -> str:
    """Get filesystem path for an array within a zarr.Group."""
    store = group.store
    if not hasattr(store, "root"):
        raise TypeError(f"TensorStore requires a LocalStore (filesystem) backend, got {type(store).__name__!r}.")
    root = store.root
    gpath = group.path
    if gpath:
        return str(root / gpath / name)
    return str(root / name)


_TS_IMPL_BASE = ZarrImplementation[zarr.Group, ts.TensorStore] if _TS_AVAILABLE else object  # type: ignore[assignment]


class TensorStoreImplementation(_TS_IMPL_BASE):
    """Hybrid implementation: zarr-python groups + TensorStore array I/O.

    Group operations (metadata, hierarchy) are delegated to zarr-python.
    Array operations (create, read, write, downsample) use TensorStore
    for high-performance I/O with configurable concurrency and caching.
    """

    def __init__(self, config: TensorStoreConfig | None = None):
        self.config = config or TensorStoreConfig()
        self._array_cache: dict[str, Any] = {}

    def _context(self) -> ts.Context:
        import tensorstore as ts

        # If the caller provided a shared Context on the config, always use it.
        # Sharing one Context across many TensorStoreImplementation instances
        # lets a single cache pool + thread pool serve every open_ome_zarr
        # call — important for workloads that open dozens of plates.
        if self.config.shared_context is not None:
            return self.config.shared_context

        if not hasattr(self, "_ctx") or self._ctx is None:
            ctx_opts = dict(self.config.context or {})
            if self.config.data_copy_concurrency:
                ctx_opts.setdefault(
                    "data_copy_concurrency",
                    {"limit": self.config.data_copy_concurrency},
                )
            if self.config.cache_pool_bytes is not None:
                ctx_opts.setdefault(
                    "cache_pool",
                    {"total_bytes_limit": self.config.cache_pool_bytes},
                )
            if self.config.file_io_concurrency is not None:
                ctx_opts.setdefault(
                    "file_io_concurrency",
                    {"limit": self.config.file_io_concurrency},
                )
            if not self.config.file_io_sync:
                ctx_opts.setdefault("file_io_sync", False)
            if self.config.file_io_locking != "auto":
                ctx_opts.setdefault("file_io_locking", self.config.file_io_locking)
            if self.config.extra_context:
                ctx_opts.update(self.config.extra_context)
            self._ctx = ts.Context(ctx_opts)
        return self._ctx

    # -- Group operations (delegated to zarr-python) -----------------------

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

    def create_array(self, group: zarr.Group, name: str, spec: ArraySpec, *, overwrite: bool = False) -> ts.TensorStore:
        path = _resolve_array_path(group, name)
        self._array_cache.pop(path, None)
        ts_spec = _spec_to_ts(spec, path)
        return _ts_open(ts_spec, create=True, delete_existing=overwrite, context=self._context())

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
    ) -> ts.TensorStore:
        shuffle_map = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}
        comp = self.config.compressor
        path = _resolve_array_path(group, name)
        self._array_cache.pop(path, None)
        # TensorStore zarr2 driver requires bool fill_value for bool dtype
        resolved_dtype = np.dtype(dtype)
        if resolved_dtype.kind == "b":
            fill_value = bool(fill_value)
        spec = {
            "driver": "zarr2",
            "kvstore": {"driver": "file", "path": path},
            "metadata": {
                "shape": list(shape),
                "chunks": list(chunks),
                "dtype": resolved_dtype.str,
                "compressor": {
                    "id": "blosc",
                    "cname": comp.cname,
                    "clevel": comp.clevel,
                    "shuffle": shuffle_map.get(comp.shuffle, 2),
                },
                "fill_value": fill_value,
                "order": "C",
                "filters": None,
            },
        }
        return _ts_open(spec, create=True, delete_existing=overwrite, context=self._context())

    def open_array(self, group: zarr.Group, name: str) -> ts.TensorStore:
        key = _resolve_array_path(group, name)
        if key not in self._array_cache:
            driver = "zarr3" if group.metadata.zarr_format == 3 else "zarr2"
            writable = not getattr(group.store, "read_only", False)
            spec = {
                "driver": driver,
                "kvstore": {"driver": "file", "path": key},
            }
            open_kwargs: dict[str, Any] = {
                "open": True,
                "read": True,
                "write": writable,
                "context": self._context(),
            }
            if self.config.recheck_cached_data is not None:
                open_kwargs["recheck_cached_data"] = self.config.recheck_cached_data
            self._array_cache[key] = _ts_open(spec, **open_kwargs)
        return self._array_cache[key]

    # -- Array I/O ---------------------------------------------------------

    def read(self, handle: ts.TensorStore, selection: Any) -> np.ndarray:
        return np.asarray(handle[selection].read().result())

    def write(self, handle: ts.TensorStore, selection: Any, data: np.ndarray) -> None:
        handle[selection].write(data).result()

    def read_oindex(self, handle: ts.TensorStore, selection: Any) -> np.ndarray:
        return np.asarray(handle.oindex[tuple(selection)].read().result())

    def write_oindex(self, handle: ts.TensorStore, selection: Any, data: np.ndarray) -> None:
        handle.oindex[tuple(selection)].write(data).result()

    def resize(self, handle: ts.TensorStore, new_shape: tuple[int, ...]) -> None:
        handle.resize(exclusive_max=new_shape).result()

    # -- Array metadata ----------------------------------------------------

    def get_shape(self, handle: ts.TensorStore) -> tuple[int, ...]:
        return tuple(handle.shape)

    def get_dtype(self, handle: ts.TensorStore) -> np.dtype:
        return handle.dtype.numpy_dtype

    def get_chunks(self, handle: ts.TensorStore) -> tuple[int, ...]:
        cl = handle.chunk_layout
        return tuple(cl.read_chunk.shape) if cl.read_chunk.shape else self.get_shape(handle)

    def get_shards(self, handle: ts.TensorStore) -> tuple[int, ...] | None:
        cl = handle.chunk_layout
        read_shape = cl.read_chunk.shape
        write_shape = cl.write_chunk.shape
        if read_shape and write_shape and tuple(read_shape) != tuple(write_shape):
            return tuple(write_shape)
        return None

    # -- Conversions -------------------------------------------------------

    def to_dask(self, handle: ts.TensorStore) -> Any:
        import dask.array as da

        return da.from_array(handle, chunks=self.get_chunks(handle))

    def write_from_dask(self, handle: ts.TensorStore, dask_array: Any) -> None:
        import dask

        result = dask_array.store(handle, lock=False, compute=False)
        dask.compute(result)

    # -- High-performance operations ---------------------------------------

    def downsample(
        self,
        source: ts.TensorStore,
        target: ts.TensorStore,
        factors: list[int],
        method: str = "mean",
    ) -> None:
        """Downsample source into target using native ts.downsample."""
        import tensorstore as ts

        downsampled = ts.downsample(source, downsample_factors=factors, method=method)
        sharded = self.get_shards(target) is not None
        if sharded:
            with ts.Transaction() as txn:
                for region in self.iter_work_regions(target):
                    target.with_transaction(txn)[region].write(downsampled.with_transaction(txn)[region]).result()
        else:
            for region in self.iter_work_regions(target):
                with ts.Transaction() as txn:
                    target.with_transaction(txn)[region].write(downsampled.with_transaction(txn)[region]).result()

    def downsample_region(
        self,
        source: ts.TensorStore,
        target: ts.TensorStore,
        target_region: tuple[slice, ...],
        factors: list[int],
        method: str = "mean",
    ) -> None:
        """Downsample a single region using TensorStore."""
        import tensorstore as ts

        downsampled = ts.downsample(source, downsample_factors=factors, method=method)
        target[target_region].write(downsampled[target_region]).result()

    def iter_work_regions(self, target: ts.TensorStore) -> list[tuple[slice, ...]]:
        """Return chunk/shard-aligned regions covering all dimensions."""
        import itertools

        cl = target.chunk_layout
        write_shape = cl.write_chunk.shape or target.shape
        dim_ranges = []
        for _dim, (total, step) in enumerate(zip(target.shape, write_shape, strict=False)):
            starts = range(0, total, step)
            dim_ranges.append([slice(s, min(s + step, total)) for s in starts])
        return [tuple(region) for region in itertools.product(*dim_ranges)]
