"""Tests for the tile_and_assemble convenience function."""

import numpy as np
import zarr
from hypothesis import given
from hypothesis import strategies as st

from iohub.tile import tile_and_assemble
from tests.tile.conftest import tile_params


@given(params=tile_params(), blender=st.sampled_from(["uniform", "gaussian"]))
def test_roundtrip(synthetic_5d, params, blender, tmp_path):
    """tile_and_assemble with identity fn preserves data."""
    import uuid

    tile_size, overlap = params
    out_dir = tmp_path / uuid.uuid4().hex
    out_dir.mkdir()
    result = tile_and_assemble(
        synthetic_5d,
        fn=lambda t: t,
        tile_size=tile_size,
        output=str(out_dir / "out.zarr"),
        overlap=overlap,
        weights=blender,
    )
    assert result.shape == synthetic_5d.shape
    np.testing.assert_allclose(result.values, synthetic_5d.values, atol=1e-5)


def test_with_memory_store(synthetic_5d):
    """tile_and_assemble works with zarr.Group on MemoryStore."""
    group = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    result = tile_and_assemble(
        synthetic_5d,
        fn=lambda t: t * 2,
        tile_size={"y": 32, "x": 64},
        output=group,
        overlap={"y": 8, "x": 16},
    )
    np.testing.assert_allclose(result.values, synthetic_5d.values * 2, atol=1e-5)
