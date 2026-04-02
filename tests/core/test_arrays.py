"""Unit tests for iohub.core.arrays (NGFFArray)."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from iohub.core.arrays import NGFFArray
from iohub.core.implementations.zarr_python import ZarrPythonImplementation


@pytest.fixture
def impl():
    return ZarrPythonImplementation()


@pytest.fixture
def zarr_handle(tmp_path):
    store = zarr.open_group(str(tmp_path / "test.zarr"), mode="w", zarr_format=3)
    arr = store.create_array("data", shape=(3, 4), dtype=np.float32, chunks=(3, 4))
    arr[:] = np.arange(12, dtype=np.float32).reshape(3, 4)
    return arr


# ---------------------------------------------------------------------------
# from_handle returns the calling subclass (Self)
# ---------------------------------------------------------------------------


class _SubArray(NGFFArray):
    _N_DIMS = 2
    _SUPPORTED_DIMS = "YX"


def test_from_handle_returns_base_class(zarr_handle, impl):
    result = NGFFArray.from_handle(zarr_handle, impl)
    assert type(result) is NGFFArray


def test_from_handle_returns_subclass(zarr_handle, impl):
    """from_handle must return an instance of the calling subclass, not NGFFArray."""
    result = _SubArray.from_handle(zarr_handle, impl)
    assert isinstance(result, _SubArray)
    assert type(result) is _SubArray


def test_from_handle_preserves_dim_names(zarr_handle, impl):
    dim_names = ("y", "x")
    result = _SubArray.from_handle(zarr_handle, impl, dim_names=dim_names)
    assert result._dim_names == dim_names


# ---------------------------------------------------------------------------
# __array__ protocol
# ---------------------------------------------------------------------------


def test_array_protocol_basic(zarr_handle, impl):
    arr = NGFFArray(zarr_handle, impl)
    result = np.array(arr)
    assert result.shape == (3, 4)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.arange(12, dtype=np.float32).reshape(3, 4))


def test_array_protocol_dtype_cast(zarr_handle, impl):
    arr = NGFFArray(zarr_handle, impl)
    result = np.array(arr, dtype=np.float64)
    assert result.dtype == np.float64


def test_array_protocol_copy_none(zarr_handle, impl):
    """copy=None must not raise (NumPy 2.0 compat)."""
    arr = NGFFArray(zarr_handle, impl)
    result = np.array(arr, copy=None)
    assert result.shape == (3, 4)


def test_array_protocol_copy_true(zarr_handle, impl):
    """copy=True must return a valid array (always a copy from storage)."""
    arr = NGFFArray(zarr_handle, impl)
    result = np.array(arr, copy=True)
    assert result.shape == (3, 4)


# ---------------------------------------------------------------------------
# metadata properties
# ---------------------------------------------------------------------------


def test_shape(zarr_handle, impl):
    arr = NGFFArray(zarr_handle, impl)
    assert arr.shape == (3, 4)


def test_dtype(zarr_handle, impl):
    arr = NGFFArray(zarr_handle, impl)
    assert arr.dtype == np.float32


def test_ndim(zarr_handle, impl):
    arr = NGFFArray(zarr_handle, impl)
    assert arr.ndim == 2


def test_nbytes(zarr_handle, impl):
    arr = NGFFArray(zarr_handle, impl)
    assert arr.nbytes == 3 * 4 * np.dtype(np.float32).itemsize
