"""Tests for _blend_tiles and apply_func_tiled."""

import dask.array
import numpy as np

from iohub.tile import Tiler, apply_func_tiled, get_blender
from iohub.tile._blend import _blend_tiles

# ---- _blend_tiles unit tests ----


def test_blend_tiles_single_tile(synthetic_5d):
    """Single tile returns the tile itself."""
    tiler = Tiler(synthetic_5d, tile_size={"y": 64, "x": 128})
    specs = list(tiler)
    tiles = [s.to_xarray() for s in specs]
    blender = get_blender("uniform")
    result = _blend_tiles(tiles, specs, blender, tiler)
    # Single tile — should be the same object
    assert result is tiles[0]


def test_blend_tiles_no_overlap(synthetic_5d):
    """Non-overlapping tiles: each cell has exactly 1 contributor."""
    tiler = Tiler(synthetic_5d, tile_size={"y": 32, "x": 64}, overlap={"y": 0, "x": 0})
    specs = list(tiler)
    tiles = [s.to_xarray() for s in specs]
    blender = get_blender("uniform")
    result = _blend_tiles(tiles, specs, blender, tiler)
    assert result.shape == synthetic_5d.shape
    np.testing.assert_allclose(result.values, synthetic_5d.values, atol=1e-6)


def test_blend_tiles_uniform_identity(synthetic_5d):
    """Uniform weights + identity fn = original data."""
    tiler = Tiler(
        synthetic_5d,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
    )
    specs = list(tiler)
    tiles = [s.to_xarray() for s in specs]
    blender = get_blender("uniform")
    result = _blend_tiles(tiles, specs, blender, tiler)
    assert result.shape == synthetic_5d.shape
    np.testing.assert_allclose(result.values, synthetic_5d.values, atol=1e-5)


def test_blend_tiles_gaussian_identity(synthetic_5d):
    """Gaussian weights + identity fn = original data."""
    tiler = Tiler(
        synthetic_5d,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
    )
    specs = list(tiler)
    tiles = [s.to_xarray() for s in specs]
    blender = get_blender("gaussian")
    result = _blend_tiles(tiles, specs, blender, tiler)
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

    tiler = Tiler(
        synthetic_5d,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
    )
    specs = list(tiler)
    tiles = [s.to_xarray() for s in specs]
    result = _blend_tiles(tiles, specs, CountingBlender(), tiler)

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
    tiler = Tiler(synthetic_5d, tile_size={"y": 32, "x": 64})
    specs = list(tiler)
    tiles = [specs[0].to_xarray()]  # wrong length
    blender = get_blender("uniform")
    try:
        _blend_tiles(tiles, specs, blender, tiler)
        assert False, "Expected ValueError"
    except ValueError:
        pass


# ---- apply_func_tiled tests ----


def test_apply_func_tiled_identity_roundtrip(synthetic_position):
    """apply_func_tiled with identity fn preserves data."""
    original = synthetic_position.data[:]
    result = apply_func_tiled(
        synthetic_position,
        fn=lambda t: t,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
    )
    assert isinstance(result.data, dask.array.Array)
    assert result.shape == original.shape
    np.testing.assert_allclose(result.values, original, atol=1e-5)


def test_apply_func_tiled_scaling(synthetic_position):
    """apply_func_tiled correctly applies a scaling function."""
    original = synthetic_position.data[:]
    result = apply_func_tiled(
        synthetic_position,
        fn=lambda t: t * 2,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
    )
    np.testing.assert_allclose(result.values, original * 2, atol=1e-5)


def test_apply_func_tiled_numpy_return(synthetic_position):
    """apply_func_tiled handles fn returning np.ndarray."""
    original = synthetic_position.data[:]
    result = apply_func_tiled(
        synthetic_position,
        fn=lambda t: t.values * 3,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
    )
    np.testing.assert_allclose(result.values, original * 3, atol=1e-5)


def test_apply_func_tiled_no_overlap(synthetic_position):
    """apply_func_tiled works without overlap."""
    original = synthetic_position.data[:]
    result = apply_func_tiled(
        synthetic_position,
        fn=lambda t: t,
        tile_size={"y": 32, "x": 64},
    )
    assert result.shape == original.shape
    np.testing.assert_allclose(result.values, original, atol=1e-6)


def test_apply_func_tiled_distance_blender(synthetic_position):
    """apply_func_tiled with distance blender + identity = original."""
    original = synthetic_position.data[:]
    result = apply_func_tiled(
        synthetic_position,
        fn=lambda t: t,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
        weights="distance",
    )
    np.testing.assert_allclose(result.values, original, atol=1e-5)


# ---------------------------------------------------------------------------
# ZYX tiling tests
# ---------------------------------------------------------------------------


def test_zyx_blend_tiles_uniform_identity(synthetic_5d_large_z):
    """ZYX uniform blending with identity preserves data."""
    tiler = Tiler(
        synthetic_5d_large_z,
        tile_size={"z": 8, "y": 32, "x": 64},
        overlap={"z": 2, "y": 8, "x": 16},
    )
    specs = list(tiler)
    tiles = [s.to_xarray() for s in specs]
    blender = get_blender("uniform")
    result = _blend_tiles(tiles, specs, blender, tiler)
    assert result.shape == synthetic_5d_large_z.shape
    np.testing.assert_allclose(result.values, synthetic_5d_large_z.values, atol=1e-5)


def test_zyx_apply_func_tiled_identity(synthetic_position_large_z):
    """ZYX apply_func_tiled with identity fn preserves data."""
    original = synthetic_position_large_z.data[:]
    result = apply_func_tiled(
        synthetic_position_large_z,
        fn=lambda t: t,
        tile_size={"z": 8, "y": 32, "x": 64},
        overlap={"z": 2, "y": 8, "x": 16},
    )
    assert isinstance(result.data, dask.array.Array)
    assert result.shape == original.shape
    np.testing.assert_allclose(result.values, original, atol=1e-5)


def test_zyx_apply_func_tiled_scaling(synthetic_position_large_z):
    """ZYX apply_func_tiled correctly applies a scaling function."""
    original = synthetic_position_large_z.data[:]
    result = apply_func_tiled(
        synthetic_position_large_z,
        fn=lambda t: t * 3,
        tile_size={"z": 4, "y": 32, "x": 64},
        overlap={"z": 1, "y": 8, "x": 16},
    )
    np.testing.assert_allclose(result.values, original * 3, atol=1e-5)
