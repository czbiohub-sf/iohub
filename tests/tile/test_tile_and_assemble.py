"""Tests for the three-phase tile store API: create_tile_store, process_tiles, stitch_from_store."""

import numpy as np
import pytest

from iohub.tile import create_tile_store, process_tiles, stitch_from_store


def _identity(t):
    return t


def _scale2(t):
    return t * 2


def test_roundtrip_yx(synthetic_position, tmp_path):
    """Three-phase identity round-trip preserves data (YX tiling)."""
    original = synthetic_position.data[:]
    store = str(tmp_path / "tiles.zarr")
    output = str(tmp_path / "out.zarr")

    batches = create_tile_store(
        synthetic_position,
        tile_size={"y": 32, "x": 64},
        store=store,
        overlap={"y": 8, "x": 16},
    )
    for batch in batches:
        process_tiles(synthetic_position, _identity, store, batch)

    stitch_from_store(store, output, synthetic_position, weights="uniform")

    from iohub.ngff import open_ome_zarr

    result = open_ome_zarr(output, layout="fov").data[:]
    assert result.shape == original.shape
    np.testing.assert_allclose(result.astype(np.float32), original.astype(np.float32), atol=1e-4)


def test_roundtrip_gaussian(synthetic_position, tmp_path):
    """Gaussian blending round-trip preserves identity."""
    original = synthetic_position.data[:]
    store = str(tmp_path / "tiles.zarr")
    output = str(tmp_path / "out.zarr")

    batches = create_tile_store(
        synthetic_position,
        tile_size={"y": 32, "x": 64},
        store=store,
        overlap={"y": 8, "x": 16},
    )
    for batch in batches:
        process_tiles(synthetic_position, _identity, store, batch)

    stitch_from_store(store, output, synthetic_position, weights="gaussian")

    from iohub.ngff import open_ome_zarr

    result = open_ome_zarr(output, layout="fov").data[:]
    np.testing.assert_allclose(result.astype(np.float32), original.astype(np.float32), atol=1e-4)


def test_scaling(synthetic_position, tmp_path):
    """process_tiles correctly applies a scaling function."""
    original = synthetic_position.data[:]
    store = str(tmp_path / "tiles.zarr")
    output = str(tmp_path / "out.zarr")

    batches = create_tile_store(
        synthetic_position,
        tile_size={"y": 32, "x": 64},
        store=store,
        overlap={"y": 8, "x": 16},
    )
    for batch in batches:
        process_tiles(synthetic_position, _scale2, store, batch)

    stitch_from_store(store, output, synthetic_position, weights="uniform")

    from iohub.ngff import open_ome_zarr

    result = open_ome_zarr(output, layout="fov").data[:]
    np.testing.assert_allclose(result.astype(np.float32), (original * 2).astype(np.float32), atol=1e-4)


def test_create_tile_store_returns_batches(synthetic_position, tmp_path):
    """create_tile_store returns correct batches."""
    store = str(tmp_path / "tiles.zarr")
    batches = create_tile_store(
        synthetic_position,
        tile_size={"y": 32, "x": 64},
        store=store,
        overlap={"y": 8, "x": 16},
        tile_batch_size=3,
    )
    all_ids = [tid for batch in batches for tid in batch]
    # All IDs present, no duplicates
    assert sorted(all_ids) == list(range(len(all_ids)))
    # Each batch ≤ batch_size
    assert all(len(b) <= 3 for b in batches)


def test_store_already_exists_raises(synthetic_position, tmp_path):
    """create_tile_store raises if store already exists."""
    store = str(tmp_path / "tiles.zarr")
    create_tile_store(synthetic_position, tile_size={"y": 32, "x": 64}, store=store)
    with pytest.raises(FileExistsError):
        create_tile_store(synthetic_position, tile_size={"y": 32, "x": 64}, store=store)


def test_tile_ids_subset_parallel_pattern(synthetic_position, tmp_path):
    """process_tiles with disjoint tile_id subsets (SLURM pattern) stitches correctly."""
    original = synthetic_position.data[:]
    store = str(tmp_path / "tiles.zarr")
    output = str(tmp_path / "out.zarr")

    batches = create_tile_store(
        synthetic_position,
        tile_size={"y": 32, "x": 64},
        store=store,
        overlap={"y": 8, "x": 16},
        tile_batch_size=3,  # small batches to simulate multiple jobs
    )
    # Each batch processed independently (simulates separate SLURM jobs)
    for batch in batches:
        process_tiles(synthetic_position, _identity, store, batch)

    stitch_from_store(store, output, synthetic_position, weights="uniform")

    from iohub.ngff import open_ome_zarr

    result = open_ome_zarr(output, layout="fov").data[:]
    np.testing.assert_allclose(result.astype(np.float32), original.astype(np.float32), atol=1e-4)


def test_stitch_raises_on_missing_tile(synthetic_position, tmp_path):
    """stitch_from_store raises FileNotFoundError when a tile FOV is missing."""
    store = str(tmp_path / "tiles.zarr")
    output = str(tmp_path / "out.zarr")

    batches = create_tile_store(
        synthetic_position,
        tile_size={"y": 32, "x": 64},
        store=store,
        overlap={"y": 8, "x": 16},
        tile_batch_size=2,  # ensure multiple batches
    )
    assert len(batches) > 1, "Need multiple batches for this test"
    # Only process the first batch — leave the rest missing
    process_tiles(synthetic_position, _identity, store, batches[0])

    with pytest.raises(FileNotFoundError, match="Tile"):
        stitch_from_store(store, output, synthetic_position)


def test_process_tiles_wrong_shape_raises(synthetic_position, tmp_path):
    """process_tiles raises if fn returns wrong spatial shape."""
    store = str(tmp_path / "tiles.zarr")
    create_tile_store(
        synthetic_position,
        tile_size={"y": 32, "x": 64},
        store=store,
    )

    # fn that changes spatial shape
    def bad_fn(t):
        return t.values[:, :, :, :16, :16]  # crop to wrong size

    with pytest.raises(ValueError, match="preserve spatial dimensions"):
        process_tiles(synthetic_position, bad_fn, store, [0])


# ---------------------------------------------------------------------------
# ZYX tiling tests
# ---------------------------------------------------------------------------


def test_zyx_roundtrip(synthetic_position_large_z, tmp_path):
    """Three-phase ZYX identity round-trip preserves data."""
    original = synthetic_position_large_z.data[:]
    store = str(tmp_path / "tiles_zyx.zarr")
    output = str(tmp_path / "out_zyx.zarr")

    batches = create_tile_store(
        synthetic_position_large_z,
        tile_size={"z": 8, "y": 32, "x": 64},
        store=store,
        overlap={"z": 2, "y": 8, "x": 16},
    )
    for batch in batches:
        process_tiles(synthetic_position_large_z, _identity, store, batch)

    stitch_from_store(store, output, synthetic_position_large_z, weights="uniform")

    from iohub.ngff import open_ome_zarr

    result = open_ome_zarr(output, layout="fov").data[:]
    assert result.shape == original.shape
    np.testing.assert_allclose(result.astype(np.float32), original.astype(np.float32), atol=1e-4)
