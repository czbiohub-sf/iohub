"""TensorStore implementation (optional dependency)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

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


class _TsAttrs(dict):
    """Persistent attrs for a _TsGroup.

    Reads from ``zarr.json`` (v3) or ``.zattrs`` (v2).
    Writes always go to ``zarr.json``.
    """

    def __init__(self, group: _TsGroup):
        self._group = group
        super().__init__(self._load())

    def _load(self) -> dict:
        if self._group.zarr_driver == "zarr2":
            zattrs = self._group.path / ".zattrs"
            if zattrs.exists():
                return json.loads(zattrs.read_text())
            return {}
        zarr_json = self._group.path / "zarr.json"
        if zarr_json.exists():
            return json.loads(zarr_json.read_text()).get("attributes", {})
        return {}

    def _save(self) -> None:
        if self._group.zarr_driver == "zarr2":
            (self._group.path / ".zattrs").write_text(json.dumps(dict(self)))
            return
        zarr_json = self._group.path / "zarr.json"
        if zarr_json.exists():
            meta = json.loads(zarr_json.read_text())
        else:
            meta = {"zarr_format": 3, "node_type": "group", "attributes": {}}
        meta["attributes"] = dict(self)
        zarr_json.write_text(json.dumps(meta))

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._save()

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._save()


def _detect_zarr_driver(path: Path, zarr_format: int | None = None) -> str:
    """Detect zarr format for a store root. Called once at open_group time."""
    if zarr_format == 2:
        return "zarr2"
    if zarr_format == 3:
        return "zarr3"
    if (path / "zarr.json").exists():
        return "zarr3"
    if (path / ".zattrs").exists() or (path / ".zgroup").exists():
        return "zarr2"
    return "zarr3"


class _TsGroup:
    """Lightweight group handle (tensorstore has no native group concept)."""

    def __init__(
        self,
        path: Path,
        mode: str,
        impl: TensorStoreImplementation,
        zarr_driver: str = "zarr3",
        root: Path | None = None,
    ):
        if mode == "w-" and path.exists():
            raise FileExistsError(f"Store already exists: {path}")
        if mode in ("w", "w-", "a") and not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            if zarr_driver == "zarr2":
                (path / ".zgroup").write_text('{"zarr_format": 2}')
            else:
                (path / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group", "attributes": {}}')
        self.path = path
        self.mode = mode
        self._impl = impl
        self.zarr_driver = zarr_driver
        self._root = root if root is not None else path

    def create_group(self, name: str, overwrite: bool = False) -> _TsGroup:
        sub = self.path / name
        if sub.exists() and not overwrite:
            return _TsGroup(path=sub, mode="a", impl=self._impl, zarr_driver=self.zarr_driver)
        sub.mkdir(parents=True, exist_ok=True)
        if self.zarr_driver == "zarr2":
            zgroup = sub / ".zgroup"
            if not zgroup.exists() or overwrite:
                zgroup.write_text('{"zarr_format": 2}')
        else:
            zarr_json = sub / "zarr.json"
            if not zarr_json.exists() or overwrite:
                zarr_json.write_text(json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}))
        return _TsGroup(path=sub, mode="a", impl=self._impl, zarr_driver=self.zarr_driver)

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None

    def __delitem__(self, name: str) -> None:
        import shutil

        sub = self.path / name
        if not sub.exists():
            raise KeyError(name)
        shutil.rmtree(sub)

    def __getitem__(self, name: str):
        result = self.get(name)
        if result is None:
            raise KeyError(name)
        return result

    def get(self, name: str, default=None):
        sub = self.path / name
        if not sub.is_dir():
            return default
        if self.zarr_driver == "zarr3":
            try:
                meta = json.loads((sub / "zarr.json").read_text())
                if meta.get("node_type") == "array":
                    return self._impl.open_array(self, name)
                if meta.get("node_type") == "group":
                    return _TsGroup(path=sub, mode="a", impl=self._impl, zarr_driver=self.zarr_driver, root=self._root)
            except (OSError, ValueError):
                pass
        else:
            if (sub / ".zarray").exists():
                return self._impl.open_array(self, name)
            if (sub / ".zgroup").exists():
                return _TsGroup(path=sub, mode="a", impl=self._impl, zarr_driver=self.zarr_driver, root=self._root)
        return default

    @property
    def store(self) -> _TsGroup:
        return self

    @property
    def root(self) -> Path:
        return self.path

    @property
    def name(self) -> str:
        try:
            rel = self.path.relative_to(self._root)
            return "/" + str(rel) if str(rel) != "." else "/"
        except ValueError:
            return str(self.path)

    @property
    def basename(self) -> str:
        return self.path.name

    @property
    def attrs(self) -> _TsAttrs:
        if not hasattr(self, "_attrs_cache") or self._attrs_cache is None:
            self._attrs_cache = _TsAttrs(self)
        return self._attrs_cache

    def tree(self, level: int | None = None) -> str:
        lines = [self.basename]
        self._tree_lines(self.path, "", level, 0, lines)
        return "\n".join(lines)

    def _tree_lines(self, p: Path, prefix: str, max_level: int | None, depth: int, lines: list) -> None:
        if max_level is not None and depth >= max_level:
            return
        try:
            children = sorted(
                d for d in (entry.name for entry in p.iterdir()) if (p / d).is_dir() and not d.startswith(".")
            )
        except OSError:
            return
        for i, child in enumerate(children):
            connector = "└── " if i == len(children) - 1 else "├── "
            lines.append(f"{prefix}{connector}{child}")
            extension = "    " if i == len(children) - 1 else "│   "
            self._tree_lines(p / child, prefix + extension, max_level, depth + 1, lines)


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


_TS_IMPL_BASE = ZarrImplementation[_TsGroup, ts.TensorStore] if _TS_AVAILABLE else object  # type: ignore[assignment]


class TensorStoreImplementation(_TS_IMPL_BASE):
    """TensorStore-backed I/O implementation."""

    def __init__(self, config: TensorStoreConfig | None = None):
        self.config = config or TensorStoreConfig()
        self._array_cache: dict[str, Any] = {}

    def _context(self) -> ts.Context:
        import tensorstore as ts

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

    # -- Group operations --------------------------------------------------

    def open_group(self, path: StorePath, mode: str, zarr_format: int | None = None) -> _TsGroup:
        p = Path(path)
        driver = _detect_zarr_driver(p, zarr_format)
        return _TsGroup(path=p, mode=mode, impl=self, zarr_driver=driver, root=p)

    def _iter_children(self, group: _TsGroup, node_type: str) -> list[str]:
        """Return sorted child names matching node_type ('group' or 'array')."""
        p = Path(group.path)
        if not p.is_dir():
            return []
        keys: list[str] = []
        match group.zarr_driver:
            case "zarr3":
                for entry in p.iterdir():
                    d = entry.name
                    if not entry.is_dir() or d.startswith("."):
                        continue
                    try:
                        meta = json.loads((entry / "zarr.json").read_text())
                        if meta.get("node_type") == node_type:
                            keys.append(d)
                    except (OSError, ValueError):
                        pass
            case "zarr2":
                sentinel = {"group": ".zgroup", "array": ".zarray"}[node_type]
                keys = [e.name for e in p.iterdir() if (p / e.name / sentinel).exists()]
        return sorted(keys)

    def group_keys(self, group: _TsGroup) -> list[str]:
        return self._iter_children(group, "group")

    def array_keys(self, group: _TsGroup) -> list[str]:
        return self._iter_children(group, "array")

    def close(self, group: _TsGroup) -> None:
        pass  # TensorStore handles are not persistent connections

    def get_zarr_format(self, group: _TsGroup) -> int:
        return 2 if group.zarr_driver == "zarr2" else 3

    # -- Array lifecycle ---------------------------------------------------

    def create_array(self, group: _TsGroup, name: str, spec: ArraySpec, *, overwrite: bool = False) -> ts.TensorStore:
        ts_spec = _spec_to_ts(spec, str(Path(group.path) / name))
        return _ts_open(ts_spec, create=True, delete_existing=overwrite, context=self._context())

    def create_array_v2(
        self,
        group: _TsGroup,
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
        spec = {
            "driver": "zarr2",
            "kvstore": {"driver": "file", "path": str(Path(group.path) / name)},
            "metadata": {
                "shape": list(shape),
                "chunks": list(chunks),
                "dtype": np.dtype(dtype).str,  # zarr2 uses NumPy dtype strings e.g. "<u2"
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

    def open_array(self, group: _TsGroup, name: str) -> ts.TensorStore:
        key = str(Path(group.path) / name)
        if key not in self._array_cache:
            spec = {
                "driver": group.zarr_driver,
                "kvstore": {"driver": "file", "path": key},
            }
            self._array_cache[key] = _ts_open(
                spec,
                open=True,
                read=True,
                write=(group.mode != "r"),
                context=self._context(),
            )
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
