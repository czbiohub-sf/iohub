"""Tests for Assembler tile reassembly."""

import numpy as np
import pytest

from iohub.tile import Tiler
from iohub.tile._assembler import Assembler


def test_roundtrip_preserves_data(synthetic_position, tmp_path):
    """Identity round-trip (tiler -> assembler) preserves data and coords."""
    original = synthetic_position.data[:]
    xa = synthetic_position.to_xarray()
    tiler = Tiler(xa, tile_size={"y": 32, "x": 64}, overlap={"y": 8, "x": 16})
    asm = Assembler(
        tiler,
        output=str(tmp_path / "out.zarr"),
        source_position=synthetic_position,
        weights="uniform",
    )
    for tile in tiler:
        asm.append(tile, tile.to_xarray())
    result = asm.get_output()

    assert result.shape == original.shape
    np.testing.assert_allclose(result.values, original, atol=1e-5)


def test_roundtrip_gaussian(synthetic_position, tmp_path):
    """Gaussian blending round-trip preserves identity."""
    original = synthetic_position.data[:]
    xa = synthetic_position.to_xarray()
    tiler = Tiler(xa, tile_size={"y": 32, "x": 64}, overlap={"y": 8, "x": 16})
    asm = Assembler(
        tiler,
        output=str(tmp_path / "gauss.zarr"),
        source_position=synthetic_position,
        weights="gaussian",
    )
    for tile in tiler:
        asm.append(tile, tile.to_xarray())
    result = asm.get_output()

    assert result.shape == original.shape
    np.testing.assert_allclose(result.values, original, atol=1e-5)


def test_append_after_finalize_raises(synthetic_position, tmp_path):
    """Appending after get_output() raises RuntimeError."""
    xa = synthetic_position.to_xarray()
    tiler = Tiler(xa, tile_size={"y": 64, "x": 128})
    asm = Assembler(
        tiler,
        output=str(tmp_path / "out.zarr"),
        source_position=synthetic_position,
        weights="uniform",
    )
    for tile in tiler:
        asm.append(tile, tile.to_xarray())
    asm.get_output()
    with pytest.raises(RuntimeError, match="already finalized"):
        asm.append(tiler[0], tiler[0].to_xarray())


def test_parallel_safety(synthetic_position, tmp_path):
    """validate_parallel_safety detects chunk conflicts from overlapping tiles."""
    xa = synthetic_position.to_xarray()

    tiler_safe = Tiler(xa, tile_size={"y": 32, "x": 64})
    asm_safe = Assembler(
        tiler_safe,
        output=str(tmp_path / "a.zarr"),
        source_position=synthetic_position,
        chunks={"y": 32, "x": 64},
    )
    assert asm_safe.validate_parallel_safety() is True

    tiler_unsafe = Tiler(xa, tile_size={"y": 32, "x": 64}, overlap={"y": 8, "x": 16})
    asm_unsafe = Assembler(
        tiler_unsafe,
        output=str(tmp_path / "b.zarr"),
        source_position=synthetic_position,
        chunks={"y": 32, "x": 64},
    )
    assert asm_unsafe.validate_parallel_safety() is False


def test_numpy_input_accepted(synthetic_position, tmp_path):
    """append() accepts raw np.ndarray in addition to xr.DataArray."""
    original = synthetic_position.data[:]
    xa = synthetic_position.to_xarray()
    tiler = Tiler(xa, tile_size={"y": 64, "x": 128})
    asm = Assembler(
        tiler,
        output=str(tmp_path / "out.zarr"),
        source_position=synthetic_position,
        weights="uniform",
    )
    for tile in tiler:
        asm.append(tile, tile.to_xarray().values)
    result = asm.get_output()
    np.testing.assert_allclose(result.values, original, atol=1e-6)
