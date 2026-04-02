"""ZarrImplementation Protocol -- the contract for zarr I/O backends.

Type parameter conventions:
  G -- the native group handle type (e.g. ``zarr.Group``, ``_TsGroup``)
  A -- the native array handle type (e.g. ``zarr.Array``, ``ts.TensorStore``)

Concrete bindings per implementation:
  ZarrPythonImplementation  ->  G=zarr.Group,  A=zarr.Array
  TensorStoreImplementation ->  G=_TsGroup,    A=ts.TensorStore
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np

from iohub.core.specs import ArraySpec
from iohub.core.types import StorePath


@runtime_checkable
class GroupBackend[G](Protocol):
    """Protocol for opening and navigating zarr groups."""

    def open_group(self, path: StorePath, mode: str, zarr_format: int | None = None) -> G: ...

    def group_keys(self, group: G) -> list[str]: ...

    def array_keys(self, group: G) -> list[str]: ...

    def close(self, group: G) -> None: ...

    def get_zarr_format(self, group: G) -> int: ...


@runtime_checkable
class ArrayBackend[G, A](Protocol):
    """Protocol for creating and opening arrays within a group."""

    def create_array(self, group: G, name: str, spec: ArraySpec, *, overwrite: bool = False) -> A:
        """Create an array from an ``ArraySpec``. Returns the native handle."""
        ...

    def open_array(self, group: G, name: str) -> A: ...

    def create_array_v2(
        self,
        group: G,
        name: str,
        *,
        shape: tuple[int, ...],
        dtype: Any,
        chunks: tuple[int, ...],
        fill_value: int = 0,
        overwrite: bool = False,
    ) -> A:
        """Create a zarr v2 array. Only required for v0.4 store support."""
        raise NotImplementedError(f"{type(self).__name__!r} does not support zarr v2 array creation.")


@runtime_checkable
class ArrayIO[A](Protocol):
    """Protocol for reading, writing, and querying array data."""

    def read(self, handle: A, selection: Any) -> np.ndarray: ...

    def write(self, handle: A, selection: Any, data: np.ndarray) -> None: ...

    def read_oindex(self, handle: A, selection: Any) -> np.ndarray: ...

    def write_oindex(self, handle: A, selection: Any, data: np.ndarray) -> None: ...

    def resize(self, handle: A, new_shape: tuple[int, ...]) -> None: ...

    def get_shape(self, handle: A) -> tuple[int, ...]: ...

    def get_dtype(self, handle: A) -> np.dtype: ...

    def get_chunks(self, handle: A) -> tuple[int, ...]: ...

    def get_shards(self, handle: A) -> tuple[int, ...] | None: ...

    def to_dask(self, handle: A) -> Any:
        """Return the handle wrapped as a dask array."""
        ...

    def write_from_dask(self, handle: A, dask_array: Any) -> None:
        """Write a dask array into the store."""
        ...

    def downsample(
        self,
        source: A,
        target: A,
        factors: list[int],
        method: str = "mean",
    ) -> None:
        """Downsample source into target sequentially."""
        ...

    def downsample_region(
        self,
        source: A,
        target: A,
        target_region: tuple[slice, ...],
        factors: list[int],
        method: str = "mean",
    ) -> None:
        """Downsample source into target for a single region.

        ``target_region`` is expressed in **target coordinates**.
        Implementations are responsible for mapping back to source
        coordinates (e.g. multiplying by ``factors``) as needed.
        """
        ...

    def iter_work_regions(self, target: A) -> list[tuple[slice, ...]]:
        """Return shard/chunk-aligned regions suitable for parallel iteration."""
        ...


@runtime_checkable
class ZarrImplementation[G, A](GroupBackend[G], ArrayBackend[G, A], ArrayIO[A], Protocol):
    """Combined Protocol satisfied by any full zarr I/O backend.

    A conforming class must implement all methods from
    :class:`GroupBackend`, :class:`ArrayBackend`, and :class:`ArrayIO`.
    Specialise as ``ZarrImplementation[zarr.Group, zarr.Array]`` for
    static type-checker use.
    """

    ...
