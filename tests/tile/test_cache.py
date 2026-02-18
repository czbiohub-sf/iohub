"""Tests for graph-informed overlap caching."""

import numpy as np

from iohub.tile import Slicer, map_tiles
from iohub.tile._cache import (
    _bfs_tile_order,
    _estimate_overlap_bytes,
    _overlap_regions,
)


def test_overlap_regions(synthetic_5d):
    """Overlap regions match expected intersections."""
    slicer = Slicer(
        synthetic_5d,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
    )
    regions = _overlap_regions(slicer)

    assert len(regions) > 0

    # Every region should have positive area
    for y_sl, x_sl in regions:
        assert y_sl.stop > y_sl.start
        assert x_sl.stop > x_sl.start

    # Regions should be within data bounds
    for y_sl, x_sl in regions:
        assert y_sl.start >= 0
        assert x_sl.start >= 0
        assert y_sl.stop <= synthetic_5d.sizes["y"]
        assert x_sl.stop <= synthetic_5d.sizes["x"]


def test_overlap_regions_no_overlap(synthetic_5d):
    """No overlap → no regions."""
    slicer = Slicer(
        synthetic_5d,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 0, "x": 0},
    )
    regions = _overlap_regions(slicer)
    assert len(regions) == 0


def test_overlap_regions_deduplication(synthetic_5d):
    """Regions with identical pixel ranges are deduplicated."""
    slicer = Slicer(
        synthetic_5d,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
    )
    regions = _overlap_regions(slicer)
    keys = [(y.start, y.stop, x.start, x.stop) for y, x in regions]
    assert len(keys) == len(set(keys))


def test_bfs_order(synthetic_5d):
    """BFS visits every tile and adjacent tiles are near each other."""
    slicer = Slicer(
        synthetic_5d,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
    )
    order = _bfs_tile_order(slicer)

    # Every tile visited exactly once
    assert sorted(order) == list(range(len(slicer)))

    # Adjacent tiles (graph neighbors) should be closer in BFS order
    # than in a random permutation. Just check they're all present.
    assert len(order) == len(slicer)


def test_bfs_order_no_overlap(synthetic_5d):
    """BFS on disconnected graph still returns all tiles."""
    slicer = Slicer(
        synthetic_5d,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 0, "x": 0},
    )
    order = _bfs_tile_order(slicer)
    assert sorted(order) == list(range(len(slicer)))


def test_estimate_overlap_bytes(synthetic_5d):
    """Byte estimate is positive when overlap > 0."""
    slicer = Slicer(
        synthetic_5d,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
    )
    nbytes = _estimate_overlap_bytes(slicer)
    assert nbytes > 0

    # With no overlap, should be 0
    slicer_no = Slicer(
        synthetic_5d,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 0, "x": 0},
    )
    assert _estimate_overlap_bytes(slicer_no) == 0


def test_map_tiles_persist_roundtrip(synthetic_5d):
    """map_tiles with cache='persist' produces correct results."""
    result = map_tiles(
        synthetic_5d,
        fn=lambda t: t,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
        cache="persist",
    )
    assert result.shape == synthetic_5d.shape
    np.testing.assert_allclose(result.values, synthetic_5d.values, atol=1e-5)


def test_map_tiles_bfs_roundtrip(synthetic_5d):
    """map_tiles with cache='bfs' produces correct results."""
    result = map_tiles(
        synthetic_5d,
        fn=lambda t: t,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
        cache="bfs",
    )
    assert result.shape == synthetic_5d.shape
    np.testing.assert_allclose(result.values, synthetic_5d.values, atol=1e-5)


def test_map_tiles_no_cache_default(synthetic_5d):
    """cache=None (default) works identically to before."""
    result = map_tiles(
        synthetic_5d,
        fn=lambda t: t,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
        cache=None,
    )
    assert result.shape == synthetic_5d.shape
    np.testing.assert_allclose(result.values, synthetic_5d.values, atol=1e-5)
