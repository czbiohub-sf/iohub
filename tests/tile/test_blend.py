"""Tests for _blend_tiles and map_tiles."""

import dask.array
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from iohub.tile import Slicer, get_blender, map_tiles
from iohub.tile._blend import _blend_tiles
from tests.tile.conftest import tile_params

# ---- _blend_tiles unit tests ----


def test_blend_tiles_single_tile(synthetic_5d):
    """Single tile returns the tile itself."""
    slicer = Slicer(synthetic_5d, tile_size={"y": 64, "x": 128})
    specs = list(slicer)
    tiles = [s.to_xarray() for s in specs]
    blender = get_blender("uniform")
    result = _blend_tiles(tiles, specs, blender, slicer)
    # Single tile — should be the same object
    assert result is tiles[0]


def test_blend_tiles_no_overlap(synthetic_5d):
    """Non-overlapping tiles: each cell has exactly 1 contributor."""
    slicer = Slicer(synthetic_5d, tile_size={"y": 32, "x": 64}, overlap={"y": 0, "x": 0})
    specs = list(slicer)
    tiles = [s.to_xarray() for s in specs]
    blender = get_blender("uniform")
    result = _blend_tiles(tiles, specs, blender, slicer)
    assert result.shape == synthetic_5d.shape
    np.testing.assert_allclose(result.values, synthetic_5d.values, atol=1e-6)


def test_blend_tiles_uniform_identity(synthetic_5d):
    """Uniform weights + identity fn = original data."""
    slicer = Slicer(
        synthetic_5d,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
    )
    specs = list(slicer)
    tiles = [s.to_xarray() for s in specs]
    blender = get_blender("uniform")
    result = _blend_tiles(tiles, specs, blender, slicer)
    assert result.shape == synthetic_5d.shape
    np.testing.assert_allclose(result.values, synthetic_5d.values, atol=1e-5)


def test_blend_tiles_gaussian_identity(synthetic_5d):
    """Gaussian weights + identity fn = original data."""
    slicer = Slicer(
        synthetic_5d,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
    )
    specs = list(slicer)
    tiles = [s.to_xarray() for s in specs]
    blender = get_blender("gaussian")
    result = _blend_tiles(tiles, specs, blender, slicer)
    assert result.shape == synthetic_5d.shape
    np.testing.assert_allclose(result.values, synthetic_5d.values, atol=1e-5)


def test_blend_tiles_is_lazy(synthetic_5d):
    """Result is dask-backed and overlap callback hasn't run yet."""
    call_count = 0

    class CountingBlender:
        """Blender that counts weight-computation calls."""

        def weights(self, tile_shape, overlap, metadata=None):
            nonlocal call_count
            call_count += 1
            return np.ones(tile_shape, dtype=np.float64)

    slicer = Slicer(
        synthetic_5d,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
    )
    specs = list(slicer)
    tiles = [s.to_xarray() for s in specs]
    result = _blend_tiles(tiles, specs, CountingBlender(), slicer)

    # Graph built but no overlap callbacks have run yet
    assert isinstance(result.data, dask.array.Array)
    assert call_count == 0, (
        f"Blender.weights() called {call_count} times during graph construction — overlap regions should stay lazy"
    )

    # Now trigger computation
    result.compute()
    assert call_count > 0, "Blender.weights() should run during compute()"


def test_blend_tiles_length_mismatch(synthetic_5d):
    """Mismatched tiles/specs raises ValueError."""
    slicer = Slicer(synthetic_5d, tile_size={"y": 32, "x": 64})
    specs = list(slicer)
    tiles = [specs[0].to_xarray()]  # wrong length
    blender = get_blender("uniform")
    try:
        _blend_tiles(tiles, specs, blender, slicer)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---- map_tiles tests ----


def test_map_tiles_identity_roundtrip(synthetic_5d):
    """map_tiles with identity fn preserves data."""
    result = map_tiles(
        synthetic_5d,
        fn=lambda t: t,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
    )
    assert isinstance(result.data, dask.array.Array)
    assert result.shape == synthetic_5d.shape
    np.testing.assert_allclose(result.values, synthetic_5d.values, atol=1e-5)


def test_map_tiles_scaling(synthetic_5d):
    """map_tiles correctly applies a scaling function."""
    result = map_tiles(
        synthetic_5d,
        fn=lambda t: t * 2,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
    )
    np.testing.assert_allclose(result.values, synthetic_5d.values * 2, atol=1e-5)


def test_map_tiles_numpy_return(synthetic_5d):
    """map_tiles handles fn returning np.ndarray."""
    result = map_tiles(
        synthetic_5d,
        fn=lambda t: t.values * 3,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
    )
    np.testing.assert_allclose(result.values, synthetic_5d.values * 3, atol=1e-5)


def test_map_tiles_no_overlap(synthetic_5d):
    """map_tiles works without overlap."""
    result = map_tiles(
        synthetic_5d,
        fn=lambda t: t,
        tile_size={"y": 32, "x": 64},
    )
    assert result.shape == synthetic_5d.shape
    np.testing.assert_allclose(result.values, synthetic_5d.values, atol=1e-6)


def test_map_tiles_distance_blender(synthetic_5d):
    """map_tiles with distance blender + identity = original."""
    result = map_tiles(
        synthetic_5d,
        fn=lambda t: t,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
        weights="distance",
    )
    np.testing.assert_allclose(result.values, synthetic_5d.values, atol=1e-5)


@given(params=tile_params(), blender=st.sampled_from(["uniform", "gaussian"]))
def test_map_tiles_roundtrip_hypothesis(synthetic_5d, params, blender):
    """Property: map_tiles(identity) == original for any valid tiling."""
    tile_size, overlap = params
    result = map_tiles(
        synthetic_5d,
        fn=lambda t: t,
        tile_size=tile_size,
        overlap=overlap,
        weights=blender,
    )
    assert result.shape == synthetic_5d.shape
    np.testing.assert_allclose(result.values, synthetic_5d.values, atol=1e-5)
