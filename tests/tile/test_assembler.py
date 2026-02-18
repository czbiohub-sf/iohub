"""Tests for Assembler tile reassembly."""

import numpy as np
import pytest
import zarr
from hypothesis import given
from hypothesis import strategies as st

from iohub.tile import Assembler, Slicer
from tests.tile.conftest import tile_params


@given(params=tile_params(), blender=st.sampled_from(["uniform", "gaussian"]))
def test_roundtrip_preserves_data(synthetic_5d, params, blender, tmp_path):
    """Identity round-trip (slicer → pass-through → assembler) preserves data and coords."""
    tile_size, overlap = params
    slicer = Slicer(synthetic_5d, tile_size=tile_size, overlap=overlap)
    # Unique output dir per hypothesis example (tmp_path is shared across examples)
    import uuid

    out_dir = tmp_path / uuid.uuid4().hex
    out_dir.mkdir()
    asm = Assembler(slicer, output=str(out_dir / "out.zarr"), weights=blender, dtype=synthetic_5d.dtype)
    for tile in slicer:
        asm.append(tile, tile.to_xarray())
    result = asm.get_output()

    assert result.shape == synthetic_5d.shape
    np.testing.assert_allclose(result.values, synthetic_5d.values, atol=1e-5)
    for dim in ("y", "x"):
        if dim in synthetic_5d.coords:
            np.testing.assert_array_equal(result.coords[dim].values, synthetic_5d.coords[dim].values)


def test_roundtrip_with_zarr_group(synthetic_5d):
    """Round-trip works with zarr.Group (MemoryStore) as output."""
    group = zarr.open_group(zarr.storage.MemoryStore(), mode="w")
    slicer = Slicer(synthetic_5d, tile_size={"y": 32, "x": 64}, overlap={"y": 8, "x": 8})
    asm = Assembler(slicer, output=group, weights="uniform", dtype=np.float32)
    for tile in slicer:
        asm.append(tile, tile.to_xarray())
    result = asm.get_output()
    np.testing.assert_allclose(result.values, synthetic_5d.values, atol=1e-6)


def test_append_after_finalize_raises(synthetic_5d, tmp_path):
    """Appending after get_output() raises RuntimeError."""
    slicer = Slicer(synthetic_5d, tile_size={"y": 64, "x": 128})
    asm = Assembler(slicer, output=str(tmp_path / "out.zarr"), weights="uniform")
    for tile in slicer:
        asm.append(tile, tile.to_xarray())
    asm.get_output()
    with pytest.raises(RuntimeError, match="already finalized"):
        asm.append(slicer[0], slicer[0].to_xarray())


def test_parallel_safety(synthetic_5d, tmp_path):
    """validate_parallel_safety detects chunk conflicts from overlapping tiles."""
    slicer_safe = Slicer(synthetic_5d, tile_size={"y": 32, "x": 64})
    asm_safe = Assembler(slicer_safe, output=str(tmp_path / "a.zarr"), chunks={"y": 32, "x": 64})
    assert asm_safe.validate_parallel_safety() is True

    slicer_unsafe = Slicer(synthetic_5d, tile_size={"y": 32, "x": 64}, overlap={"y": 8, "x": 16})
    asm_unsafe = Assembler(slicer_unsafe, output=str(tmp_path / "b.zarr"), chunks={"y": 32, "x": 64})
    assert asm_unsafe.validate_parallel_safety() is False


def test_numpy_input_accepted(synthetic_5d, tmp_path):
    """append() accepts raw np.ndarray in addition to xr.DataArray."""
    slicer = Slicer(synthetic_5d, tile_size={"y": 64, "x": 128})
    asm = Assembler(slicer, output=str(tmp_path / "out.zarr"), weights="uniform", dtype=np.float32)
    for tile in slicer:
        asm.append(tile, tile.to_xarray().values)
    result = asm.get_output()
    np.testing.assert_allclose(result.values, synthetic_5d.values, atol=1e-6)
