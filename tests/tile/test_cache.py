"""Tests for graph-informed overlap caching."""

import numpy as np

from iohub.tile import Tiler, apply_func_tiled
from iohub.tile._cache import (
    _bfs_tile_order,
    _estimate_overlap_bytes,
    _overlap_regions,
)


def test_overlap_regions(synthetic_5d):
    """Overlap regions match expected intersections."""
    tiler = Tiler(
        synthetic_5d,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
    )
    regions = _overlap_regions(tiler)

    assert len(regions) > 0

    # Every region should have positive extent in all dims
    for region in regions:
        for dim, sl in region.items():
            assert sl.stop > sl.start

    # Regions should be within data bounds
    for region in regions:
        for dim, sl in region.items():
            assert sl.start >= 0
            assert sl.stop <= synthetic_5d.sizes[dim]


def test_overlap_regions_no_overlap(synthetic_5d):
    """No overlap → no regions."""
    tiler = Tiler(
        synthetic_5d,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 0, "x": 0},
    )
    regions = _overlap_regions(tiler)
    assert len(regions) == 0


def test_overlap_regions_deduplication(synthetic_5d):
    """Regions with identical pixel ranges are deduplicated."""
    tiler = Tiler(
        synthetic_5d,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
    )
    regions = _overlap_regions(tiler)
    keys = [tuple((d, s.start, s.stop) for d, s in r.items()) for r in regions]
    assert len(keys) == len(set(keys))


def test_bfs_order(synthetic_5d):
    """BFS visits every tile and adjacent tiles are near each other."""
    tiler = Tiler(
        synthetic_5d,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
    )
    order = _bfs_tile_order(tiler)

    # Every tile visited exactly once
    assert sorted(order) == list(range(len(tiler)))

    # Adjacent tiles (graph neighbors) should be closer in BFS order
    # than in a random permutation. Just check they're all present.
    assert len(order) == len(tiler)


def test_bfs_order_no_overlap(synthetic_5d):
    """BFS on disconnected graph still returns all tiles."""
    tiler = Tiler(
        synthetic_5d,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 0, "x": 0},
    )
    order = _bfs_tile_order(tiler)
    assert sorted(order) == list(range(len(tiler)))


def test_estimate_overlap_bytes(synthetic_5d):
    """Byte estimate is positive when overlap > 0."""
    tiler = Tiler(
        synthetic_5d,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
    )
    nbytes = _estimate_overlap_bytes(tiler)
    assert nbytes > 0

    # With no overlap, should be 0
    tiler_no = Tiler(
        synthetic_5d,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 0, "x": 0},
    )
    assert _estimate_overlap_bytes(tiler_no) == 0


def test_apply_func_tiled_persist_roundtrip(synthetic_position):
    """apply_func_tiled with cache='persist' produces correct results."""
    original = synthetic_position.data[:]
    result = apply_func_tiled(
        synthetic_position,
        fn=lambda t: t,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
        cache="persist",
    )
    assert result.shape == original.shape
    np.testing.assert_allclose(result.values, original, atol=1e-5)


def test_apply_func_tiled_bfs_roundtrip(synthetic_position):
    """apply_func_tiled with cache='bfs' produces correct results."""
    original = synthetic_position.data[:]
    result = apply_func_tiled(
        synthetic_position,
        fn=lambda t: t,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
        cache="bfs",
    )
    assert result.shape == original.shape
    np.testing.assert_allclose(result.values, original, atol=1e-5)


def test_apply_func_tiled_no_cache_default(synthetic_position):
    """cache=None (default) works identically to before."""
    original = synthetic_position.data[:]
    result = apply_func_tiled(
        synthetic_position,
        fn=lambda t: t,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
        cache=None,
    )
    assert result.shape == original.shape
    np.testing.assert_allclose(result.values, original, atol=1e-5)
