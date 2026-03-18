"""Tests for Tiler tile generation."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from hypothesis import given
from hypothesis import strategies as st

from iohub.tile import SamplingMode, Tile, Tiler
from tests.tile.conftest import tile_params, tile_params_zyx


@given(
    params=tile_params(),
    mode=st.sampled_from([SamplingMode.SQUEEZE, SamplingMode.EDGE]),
)
def test_tiles_cover_full_extent(synthetic_5d, params, mode):
    """For SQUEEZE/EDGE modes, tiles must cover the full YX extent."""
    tile_size, overlap = params
    tiler = Tiler(synthetic_5d, tile_size=tile_size, overlap=overlap, mode=mode)

    covered_y = np.zeros(synthetic_5d.sizes["y"], dtype=bool)
    covered_x = np.zeros(synthetic_5d.sizes["x"], dtype=bool)
    for tile in tiler:
        covered_y[tile.slices["y"]] = True
        covered_x[tile.slices["x"]] = True
    assert covered_y.all(), f"Y not fully covered with {tile_size}, {overlap}"
    assert covered_x.all(), f"X not fully covered with {tile_size}, {overlap}"

    from functools import reduce
    from operator import mul

    assert len(tiler) == reduce(mul, tiler.tile_grid_shape, 1)


def test_single_tile_when_oversized(synthetic_5d):
    """When tile_size > data, a single tile covering the full extent is returned."""
    tiler = Tiler(synthetic_5d, tile_size={"y": 999, "x": 999})
    assert len(tiler) == 1
    assert tiler[0].tile_shape == (64, 128)


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
        Tiler(data, **kwargs)


def test_chunk_alignment_snaps_up():
    """align_to_chunks rounds tile_size up to chunk multiples."""
    dask_data = da.from_array(np.ones((1, 1, 2, 256, 512), dtype=np.float32), chunks=(1, 1, 2, 64, 128))
    data = xr.DataArray(dask_data, dims=("t", "c", "z", "y", "x"))
    tiler = Tiler(data, tile_size={"y": 20, "x": 50}, align_to_chunks=True)
    # 20 → 64, 50 → 128
    assert tiler._tile_size["y"] >= 64
    assert tiler._tile_size["x"] >= 128


def test_iter_yields_correct_types(synthetic_5d):
    """__iter__ yields Tiles, iter_xarrays yields DataArrays."""
    tiler = Tiler(synthetic_5d, tile_size={"y": 32, "x": 64})
    tiles = list(tiler)
    xas = list(tiler.iter_xarrays())
    assert all(isinstance(t, Tile) for t in tiles)
    assert all(isinstance(xa, xr.DataArray) for xa in xas)
    assert len(tiles) == len(xas)


# ---------------------------------------------------------------------------
# ZYX tiling tests
# ---------------------------------------------------------------------------


def test_zyx_tiler_grid_shape(synthetic_5d_large_z):
    """ZYX tiler produces a 3-tuple grid shape."""
    tiler = Tiler(
        synthetic_5d_large_z,
        tile_size={"z": 8, "y": 32, "x": 64},
        overlap={"z": 2, "y": 8, "x": 16},
    )
    assert len(tiler.tile_grid_shape) == 3
    assert tiler.tile_dims == ("z", "y", "x")

    from functools import reduce
    from operator import mul

    assert len(tiler) == reduce(mul, tiler.tile_grid_shape, 1)


def test_zyx_tile_spec_properties(synthetic_5d_large_z):
    """ZYX Tile has correct dims, shape, and bbox."""
    tiler = Tiler(
        synthetic_5d_large_z,
        tile_size={"z": 8, "y": 32, "x": 64},
    )
    tile = tiler[0]
    assert tile.tile_dims == ("z", "y", "x")
    assert len(tile.tile_shape) == 3
    assert tile.bbox.shape == (3, 2)
    assert "z" in tile.slices


@given(
    params=tile_params_zyx(),
    mode=st.sampled_from([SamplingMode.SQUEEZE, SamplingMode.EDGE]),
)
def test_zyx_tiles_cover_full_extent(synthetic_5d_large_z, params, mode):
    """For SQUEEZE/EDGE modes, ZYX tiles must cover all tiled dimensions."""
    tile_size, overlap = params
    tiler = Tiler(synthetic_5d_large_z, tile_size=tile_size, overlap=overlap, mode=mode)

    for dim in ("z", "y", "x"):
        covered = np.zeros(synthetic_5d_large_z.sizes[dim], dtype=bool)
        for tile in tiler:
            covered[tile.slices[dim]] = True
        assert covered.all(), f"{dim} not fully covered with {tile_size}, {overlap}"


def test_zyx_neighborhood_graph(synthetic_5d_large_z):
    """ZYX neighborhood graph has Z-direction edges."""
    tiler = Tiler(
        synthetic_5d_large_z,
        tile_size={"z": 8, "y": 64, "x": 128},
        overlap={"z": 2, "y": 0, "x": 0},
    )
    # With only Z overlap and single tile in YX, graph should have Z-edges
    assert tiler.graph.number_of_edges() > 0
    assert len(tiler) > 1


def test_zyx_to_xarray_slices_correctly(synthetic_5d_large_z):
    """to_xarray() on ZYX tile slices Z, Y, and X dims correctly."""
    tiler = Tiler(
        synthetic_5d_large_z,
        tile_size={"z": 4, "y": 32, "x": 64},
    )
    tile = tiler[0]
    xa = tile.to_xarray()
    assert xa.sizes["z"] == 4
    assert xa.sizes["y"] == 32
    assert xa.sizes["x"] == 64
    # T and C should be unsliced
    assert xa.sizes["t"] == synthetic_5d_large_z.sizes["t"]
    assert xa.sizes["c"] == synthetic_5d_large_z.sizes["c"]


def test_zyx_invalid_dim_rejected(synthetic_5d):
    """tile_size with a dim not in data raises ValueError."""
    with pytest.raises(ValueError, match="not found in data dims"):
        Tiler(synthetic_5d, tile_size={"z": 2, "y": 32, "x": 64, "w": 10})
