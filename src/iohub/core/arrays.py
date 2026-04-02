"""NGFFArray -- implementation-agnostic N-dimensional array base class."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from iohub.core.protocol import ZarrImplementation
from iohub.core.utils import pad_shape


class _OIndexProxy:
    """Proxy for orthogonal indexing on an NGFFArray."""

    def __init__(self, handle: Any, impl: ZarrImplementation):
        self._handle = handle
        self._impl = impl

    def __getitem__(self, key):
        return self._impl.read_oindex(self._handle, key)

    def __setitem__(self, key, value):
        self._impl.write_oindex(self._handle, key, value)


class NGFFArray:
    """Base class for NGFF N-dimensional arrays.

    Delegates all I/O to the configured :class:`ZarrImplementation`.
    Subclassed by ``ImageArray`` (5D) and ``TiledImageArray`` in ``nodes.py``.
    """

    _SUPPORTED_DIMS: str
    _N_DIMS: int

    def __init__(
        self,
        handle: Any,
        impl: ZarrImplementation,
        dim_names: tuple[str, ...] | None = None,
    ):
        self._handle = handle
        self._impl = impl
        self._dim_names = dim_names

    @classmethod
    def from_handle(
        cls,
        handle: Any,
        impl: ZarrImplementation,
        dim_names: tuple[str, ...] | None = None,
    ) -> Self:
        """Construct from an implementation-native handle."""
        return cls(handle, impl, dim_names=dim_names)

    def _get_dim(self, idx: int) -> int:
        return pad_shape(self.shape, target=self._N_DIMS)[idx]

    # -- Metadata ----------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self._impl.get_shape(self._handle)

    @property
    def dtype(self) -> np.dtype:
        return self._impl.get_dtype(self._handle)

    @property
    def chunks(self) -> tuple[int, ...]:
        return self._impl.get_chunks(self._handle)

    @property
    def shards(self) -> tuple[int, ...] | None:
        return self._impl.get_shards(self._handle)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def nbytes(self) -> int:
        """Total number of bytes for the uncompressed array."""
        return int(np.prod(self.shape)) * self.dtype.itemsize

    @property
    def metadata(self):
        """Delegate to native handle's metadata (if available)."""
        return getattr(self._handle, "metadata", None)

    @property
    def path(self) -> str:
        """Path of the underlying array (if available)."""
        handle = self._handle
        if hasattr(handle, "path"):
            return str(handle.path)
        return ""

    @property
    def basename(self) -> str:
        handle = self._handle
        if hasattr(handle, "basename"):
            return handle.basename
        if hasattr(handle, "path"):
            return str(handle.path).rsplit("/", 1)[-1]
        return ""

    # -- I/O ---------------------------------------------------------------

    def __getitem__(self, key):
        return self._impl.read(self._handle, key)

    def __setitem__(self, key, value):
        self._impl.write(self._handle, key, value)

    @property
    def oindex(self):
        return _OIndexProxy(self._handle, self._impl)

    def resize(self, new_shape: tuple[int, ...]) -> None:
        """Resize the array in-place."""
        self._impl.resize(self._handle, new_shape)

    def append(self, data: np.ndarray, axis: int = 0) -> tuple[int, ...]:
        """Append data along axis. Returns the new shape."""
        data = np.asarray(data)
        old_shape = self.shape
        new_shape = tuple(s + (data.shape[i] if i == axis else 0) for i, s in enumerate(old_shape))
        self.resize(new_shape)
        region = tuple(slice(old_shape[i], new_shape[i]) if i == axis else slice(None) for i in range(len(old_shape)))
        self._impl.write(self._handle, region, data)
        return new_shape

    # -- Conversions -------------------------------------------------------

    def numpy(self) -> NDArray:
        """Return the whole array as an in-RAM NumPy array."""
        return self[:]

    def __array__(self, dtype=None, copy=None):
        arr = self.numpy()
        return arr if dtype is None else arr.astype(dtype)

    def dask_array(self):
        """Return as a dask array (delegates to implementation)."""
        return self._impl.to_dask(self._handle)

    def write_from_dask(self, dask_array) -> None:
        """Write a dask array into this array's store."""
        self._impl.write_from_dask(self._handle, dask_array)

    def downsample_into(
        self,
        target: NGFFArray,
        factors: list[int],
        method: str = "mean",
    ) -> None:
        """Downsample this array into a target array."""
        self._impl.downsample(
            self._handle,
            target._handle,
            factors,
            method,
        )

    # -- Escape hatch ------------------------------------------------------

    @property
    def native(self):
        """Raw implementation handle (``zarr.Array``, TensorStore, etc.)."""
        return self._handle
