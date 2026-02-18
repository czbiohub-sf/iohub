"""Tests for Slicer tile generation."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from hypothesis import given
from hypothesis import strategies as st

from iohub.tile import SamplingMode, Slicer, TileSpec
from tests.tile.conftest import tile_params


@given(
    params=tile_params(),
    mode=st.sampled_from([SamplingMode.SQUEEZE, SamplingMode.EDGE]),
)
def test_tiles_cover_full_extent(synthetic_5d, params, mode):
    """For SQUEEZE/EDGE modes, tiles must cover the full YX extent."""
    tile_size, overlap = params
    slicer = Slicer(synthetic_5d, tile_size=tile_size, overlap=overlap, mode=mode)

    covered_y = np.zeros(synthetic_5d.sizes["y"], dtype=bool)
    covered_x = np.zeros(synthetic_5d.sizes["x"], dtype=bool)
    for tile in slicer:
        covered_y[tile.y_slice] = True
        covered_x[tile.x_slice] = True
    assert covered_y.all(), f"Y not fully covered with {tile_size}, {overlap}"
    assert covered_x.all(), f"X not fully covered with {tile_size}, {overlap}"

    assert len(slicer) == slicer.tile_grid_shape[0] * slicer.tile_grid_shape[1]


def test_single_tile_when_oversized(synthetic_5d):
    """When tile_size > data, a single tile covering the full extent is returned."""
    slicer = Slicer(synthetic_5d, tile_size={"y": 999, "x": 999})
    assert len(slicer) == 1
    assert slicer[0].shape_yx == (64, 128)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"data_dims": ("a", "b"), "tile_size": {"y": 5, "x": 5}}, "'y' and 'x' dims"),
        ({"tile_size": {"y": 32}}, "tile_size must specify"),
        ({"tile_size": {"y": 32, "x": 64}, "overlap": {"y": 32, "x": 0}}, "overlap.*must be less"),
    ],
)
def test_invalid_inputs(synthetic_5d, kwargs, match):
    """Invalid dimensions, missing tile_size keys, or excessive overlap are rejected."""
    if "data_dims" in kwargs:
        data = xr.DataArray(np.zeros((10, 10)), dims=kwargs.pop("data_dims"))
    else:
        data = synthetic_5d
    with pytest.raises(ValueError, match=match):
        Slicer(data, **kwargs)


def test_chunk_alignment_snaps_up():
    """align_to_chunks rounds tile_size up to chunk multiples."""
    dask_data = da.from_array(np.ones((1, 1, 2, 256, 512), dtype=np.float32), chunks=(1, 1, 2, 64, 128))
    data = xr.DataArray(dask_data, dims=("t", "c", "z", "y", "x"))
    slicer = Slicer(data, tile_size={"y": 20, "x": 50}, align_to_chunks=True)
    # 20 → 64, 50 → 128
    assert slicer._tile_size["y"] >= 64
    assert slicer._tile_size["x"] >= 128


def test_iter_yields_correct_types(synthetic_5d):
    """__iter__ yields TileSpecs, iter_xarrays yields DataArrays."""
    slicer = Slicer(synthetic_5d, tile_size={"y": 32, "x": 64})
    tiles = list(slicer)
    xas = list(slicer.iter_xarrays())
    assert all(isinstance(t, TileSpec) for t in tiles)
    assert all(isinstance(xa, xr.DataArray) for xa in xas)
    assert len(tiles) == len(xas)
