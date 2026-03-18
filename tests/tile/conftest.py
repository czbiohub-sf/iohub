"""Fixtures and hypothesis strategies for iohub.tile tests."""

import warnings

import hypothesis.strategies as st
import numpy as np
import pytest
import xarray as xr
from hypothesis import HealthCheck, settings

from iohub._experimental import ExperimentalWarning
from iohub.ngff import open_ome_zarr

# Default hypothesis settings for tile tests
settings.register_profile(
    "tile",
    max_examples=15,
    deadline=5000,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
settings.load_profile("tile")


@pytest.fixture(autouse=True)
def suppress_experimental_warnings():
    """Suppress ExperimentalWarning in all tile tests."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        yield


@pytest.fixture
def synthetic_5d():
    """Small 5D xr.DataArray (1,1,4,64,128) with random float32 data and physical coords."""
    rng = np.random.default_rng(42)
    data = rng.random((1, 1, 4, 64, 128), dtype=np.float32)
    return xr.DataArray(
        data,
        dims=("t", "c", "z", "y", "x"),
        coords={
            "y": np.arange(64, dtype=np.float64) * 0.325,
            "x": np.arange(128, dtype=np.float64) * 0.325,
        },
    )


@pytest.fixture
def synthetic_5d_large_z():
    """5D xr.DataArray (1,1,16,64,128) with enough Z for meaningful Z-tiling."""
    rng = np.random.default_rng(99)
    data = rng.random((1, 1, 16, 64, 128), dtype=np.float32)
    return xr.DataArray(
        data,
        dims=("t", "c", "z", "y", "x"),
        coords={
            "z": np.arange(16, dtype=np.float64) * 0.5,
            "y": np.arange(64, dtype=np.float64) * 0.325,
            "x": np.arange(128, dtype=np.float64) * 0.325,
        },
    )


@pytest.fixture
def synthetic_position(tmp_path, synthetic_5d):
    """Position node backed by a real OME-Zarr store (1,1,4,64,128) float32."""
    path = tmp_path / "synthetic.zarr"
    pos = open_ome_zarr(str(path), layout="fov", mode="w-", channel_names=["ch0"])
    pos.create_image("0", synthetic_5d.values, chunks=(1, 1, 4, 64, 128))
    return pos


@pytest.fixture
def synthetic_position_large_z(tmp_path, synthetic_5d_large_z):
    """Position node (1,1,16,64,128) float32 with Z coords for ZYX tiling."""
    path = tmp_path / "synthetic_z.zarr"
    pos = open_ome_zarr(str(path), layout="fov", mode="w-", channel_names=["ch0"])
    pos.create_image("0", synthetic_5d_large_z.values, chunks=(1, 1, 16, 64, 128))
    return pos


@st.composite
def tile_params(draw, y_size=64, x_size=128):
    """Draw a valid (tile_size, overlap) pair for a given YX shape."""
    tile_y = draw(st.integers(8, y_size))
    tile_x = draw(st.integers(8, x_size))
    overlap_y = draw(st.integers(0, tile_y - 1))
    overlap_x = draw(st.integers(0, tile_x - 1))
    return (
        {"y": tile_y, "x": tile_x},
        {"y": overlap_y, "x": overlap_x},
    )


@st.composite
def tile_params_zyx(draw, z_size=16, y_size=64, x_size=128):
    """Draw a valid (tile_size, overlap) pair for ZYX tiling."""
    tile_z = draw(st.integers(4, z_size))
    tile_y = draw(st.integers(8, y_size))
    tile_x = draw(st.integers(8, x_size))
    overlap_z = draw(st.integers(0, tile_z - 1))
    overlap_y = draw(st.integers(0, tile_y - 1))
    overlap_x = draw(st.integers(0, tile_x - 1))
    return (
        {"z": tile_z, "y": tile_y, "x": tile_x},
        {"z": overlap_z, "y": overlap_y, "x": overlap_x},
    )
