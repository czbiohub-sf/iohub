"""Tests for the ND2 reader and converter.

Requires a real ND2 file; set ``IOHUB_TEST_ND2`` to its path to run.
Osprey ND2s live under a group-restricted path, so these are skipped in CI.
"""

from __future__ import annotations

import os
from pathlib import Path

import nd2
import numpy as np
import pytest

from iohub import open_ome_zarr, read_images
from iohub.convert import TIFFConverter
from iohub.nd2 import ND2Dataset

ND2_PATH = os.environ.get("IOHUB_TEST_ND2")
pytestmark = pytest.mark.skipif(
    not (ND2_PATH and Path(ND2_PATH).is_file()),
    reason="Set IOHUB_TEST_ND2 to a readable .nd2 file to run",
)


def test_read_images_dispatch():
    reader = read_images(ND2_PATH)
    assert isinstance(reader, ND2Dataset)
    reader.close()


def test_reader_shape_and_dtype():
    with nd2.ND2File(ND2_PATH) as expected, ND2Dataset(ND2_PATH) as reader:
        _, fov = next(iter(reader))
        # Always canonical 5D (T, C, Z, Y, X).
        assert fov.axes_names == ["T", "C", "Z", "Y", "X"]
        assert fov.dtype == expected.dtype
        assert len(reader.channel_names) == reader.channels


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_convert_round_trip(tmp_path, version):
    out = tmp_path / "out.zarr"
    TIFFConverter(ND2_PATH, out, version=version)()
    with nd2.ND2File(ND2_PATH) as expected, open_ome_zarr(out, mode="r", version=version) as plate:
        # squeeze=False keeps all present axes; padded to (T, C, Z, Y, X),
        # plus a leading P axis only when the file has stage positions.
        exp = np.asarray(expected.to_xarray(squeeze=False))
        if "P" in expected.sizes:
            exp = exp[0]  # first FOV
        _, pos = next(plate.positions())
        got = pos["0"][:]  # (T, C, Z, Y, X)
        # native dtype preserved (no float16 downcast)
        assert got.dtype == expected.dtype
        # byte-for-byte match, no axis swaps
        np.testing.assert_array_equal(got, exp)
