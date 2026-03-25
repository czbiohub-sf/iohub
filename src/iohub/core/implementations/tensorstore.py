"""TensorStore implementation (optional dependency)."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
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
        zarr_json = self._group.path / "zarr.json"
        if zarr_json.exists():
            return json.loads(zarr_json.read_text()).get("attributes", {})
        zattrs = self._group.path / ".zattrs"
        if zattrs.exists():
            return json.loads(zattrs.read_text())
        return {}

    def _save(self) -> None:
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


def _detect_zarr_driver(path: Path) -> str:
    """Detect zarr format for a store root. Called once at open_group time."""
    if (path / "zarr.json").exists():
        return "zarr3"
    if (path / ".zattrs").exists() or (path / ".zgroup").exists():
        return "zarr2"
    return "zarr3"  # new stores  will be created as v3 always


class _TsGroup:
    """Lightweight group handle (tensorstore has no native group concept)."""

    def __init__(
        self,
        path: Path,
        mode: str,
        impl: TensorStoreImplementation,
        zarr_driver: str = "zarr3",
    ):
        self.path = path
        self.mode = mode
        self._impl = impl
        self.zarr_driver = zarr_driver

    def create_group(self, name: str, overwrite: bool = False) -> "_TsGroup":
        sub = self.path / name
        if sub.exists() and not overwrite:
            return _TsGroup(path=sub, mode=self.mode, impl=self._impl, zarr_driver=self.zarr_driver)
        sub.mkdir(parents=True, exist_ok=True)
        zarr_json = sub / "zarr.json"
        if not zarr_json.exists() or overwrite:
            zarr_json.write_text(json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}))
        return _TsGroup(path=sub, mode=self.mode, impl=self._impl, zarr_driver=self.zarr_driver)

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None

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
                    return _TsGroup(path=sub, mode=self.mode, impl=self._impl, zarr_driver=self.zarr_driver)
            except (OSError, ValueError):
                pass
        else:
            if (sub / ".zarray").exists():
                return self._impl.open_array(self, name)
            if (sub / ".zgroup").exists():
                return _TsGroup(path=sub, mode=self.mode, impl=self._impl, zarr_driver=self.zarr_driver)
        return default

    @property
    def store(self) -> _TsGroup:
        return self

    @property
    def root(self) -> Path:
        return self.path

    @property
    def name(self) -> str:
        return str(self.path)

    @property
    def basename(self) -> str:
        return self.path.name

    @property
    def attrs(self) -> _TsAttrs:
        return _TsAttrs(self)

    def tree(self, level: int | None = None) -> str:
        lines = [self.basename]
        self._tree_lines(self.path, "", level, 0, lines)
        return "\n".join(lines)

    def _tree_lines(self, p: Path, prefix: str, max_level: int | None, depth: int, lines: list) -> None:
        if max_level is not None and depth >= max_level:
            return
        try:
            children = sorted(d for d in os.listdir(p) if (p / d).is_dir() and not d.startswith("."))
        except OSError:
            return
        for i, child in enumerate(children):
            connector = "└── " if i == len(children) - 1 else "├── "
            lines.append(f"{prefix}{connector}{child}")
            extension = "    " if i == len(children) - 1 else "│   "
            self._tree_lines(p / child, prefix + extension, max_level, depth + 1, lines)


def _spec_to_ts(spec: ArraySpec, path: str) -> dict:
    metadata: dict = {
        "shape": list(spec.shape),
        "data_type": spec.data_type,
        "chunk_grid": spec.chunk_grid,
        "chunk_key_encoding": spec.chunk_key_encoding,
        "codecs": spec.codecs,
        "fill_value": spec.fill_value,
    }
    if spec.dimension_names:
        metadata["dimension_names"] = spec.dimension_names
    return {"driver": "zarr3", "kvstore": {"driver": "file", "path": path}, "metadata": metadata}


_TS_IMPL_BASE = ZarrImplementation[_TsGroup, ts.TensorStore] if _TS_AVAILABLE else object  # type: ignore[assignment]


class TensorStoreImplementation(_TS_IMPL_BASE):
    """TensorStore-backed I/O implementation."""

    def __init__(self, config: TensorStoreConfig | None = None):
        self.config = config or TensorStoreConfig()

    def _context(self) -> ts.Context:
        import tensorstore as ts

        ctx_opts = self.config.context or {}
        if self.config.data_copy_concurrency:
            ctx_opts.setdefault(
                "data_copy_concurrency",
                {"limit": self.config.data_copy_concurrency},
            )
        return ts.Context(ctx_opts)

    # -- Group operations --------------------------------------------------

    def open_group(self, path: StorePath, mode: str, zarr_format: int | None = None) -> _TsGroup:
        p = Path(path)
        zarr_driver = _detect_zarr_driver(p)
        return _TsGroup(path=p, mode=mode, impl=self, zarr_driver=zarr_driver)

    def group_keys(self, group: _TsGroup) -> list[str]:
        p = Path(group.path)
        if not p.is_dir():
            return []
        keys = []
        if group.zarr_driver == "zarr3":
            for d in os.listdir(p):
                sub = p / d
                if not sub.is_dir() or d.startswith("."):
                    continue
                try:
                    meta = json.loads((sub / "zarr.json").read_text())
                    if meta.get("node_type") == "group":
                        keys.append(d)
                except (OSError, ValueError):
                    pass
        else:
            keys = [d for d in os.listdir(p) if (p / d / ".zgroup").exists()]
        return sorted(keys)

    def array_keys(self, group: _TsGroup) -> list[str]:
        p = Path(group.path)
        if not p.is_dir():
            return []
        if group.zarr_driver == "zarr3":
            keys = []
            for d in os.listdir(p):
                sub = p / d
                if not sub.is_dir():
                    continue
                try:
                    meta = json.loads((sub / "zarr.json").read_text())
                    if meta.get("node_type") == "array":
                        keys.append(d)
                except (OSError, ValueError):
                    pass
        else:
            keys = [d for d in os.listdir(p) if (p / d / ".zarray").exists()]
        return sorted(keys)

    def close(self, group: _TsGroup) -> None:
        pass  # TensorStore handles are not persistent connections

    # -- Array lifecycle ---------------------------------------------------

    def create_array(self, group: _TsGroup, name: str, spec: ArraySpec, *, overwrite: bool = False) -> ts.TensorStore:
        import tensorstore as ts

        ts_spec = _spec_to_ts(spec, str(Path(group.path) / name))
        return ts.open(ts_spec, create=True, context=self._context()).result()

    def open_array(self, group: _TsGroup, name: str) -> ts.TensorStore:
        import tensorstore as ts

        spec = {
            "driver": group.zarr_driver,
            "kvstore": {"driver": "file", "path": str(Path(group.path) / name)},
        }
        return ts.open(spec, context=self._context()).result()

    # -- Array I/O ---------------------------------------------------------

    def read(self, handle: ts.TensorStore, selection: Any) -> np.ndarray:
        return np.asarray(handle[selection].read().result())

    def write(self, handle: ts.TensorStore, selection: Any, data: np.ndarray) -> None:
        handle[selection].write(data).result()

    def read_oindex(self, handle: ts.TensorStore, selection: Any) -> np.ndarray:
        import tensorstore as ts

        indexed = handle
        for dim, sel in enumerate(selection):
            indexed = indexed[ts.d[dim][sel]]
        return np.asarray(indexed.read().result())

    def write_oindex(self, handle: ts.TensorStore, selection: Any, data: np.ndarray) -> None:
        import tensorstore as ts

        indexed = handle
        for dim, sel in enumerate(selection):
            indexed = indexed[ts.d[dim][sel]]
        indexed.write(data).result()

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

    def open_for_write(self, handle: ts.TensorStore, zarr_format: int = 3) -> ts.TensorStore:
        """TensorStore handles are already high-performance, return as-is."""
        return handle

    def downsample(
        self,
        source: ts.TensorStore,
        target: ts.TensorStore,
        factors: Iterable[int],
        method: str = "mean",
    ) -> None:
        """Sequential downsample using native ts.downsample."""
        import tensorstore as ts

        downsampled = ts.downsample(source, downsample_factors=factors, method=method)
        step = target.chunk_layout.write_chunk.shape[0]
        for start in range(0, downsampled.shape[0], step):
            with ts.Transaction() as txn:
                target_with_txn = target.with_transaction(txn)
                downsampled_with_txn = downsampled.with_transaction(txn)
                stop = min(start + step, downsampled.shape[0])
                target_with_txn[start:stop].write(downsampled_with_txn[start:stop]).result()

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
        """Return chunk-aligned regions for parallel iteration."""
        step = target.chunk_layout.write_chunk.shape[0]
        regions = []
        for start in range(0, target.shape[0], step):
            stop = min(start + step, target.shape[0])
            full_slices = (slice(start, stop),) + tuple(slice(0, s) for s in target.shape[1:])
            regions.append(full_slices)
        return regions
