"""Tests for the ND2 reader and converter."""

import nd2
import numpy as np
import pytest

from iohub import open_ome_zarr, read_images
from iohub.convert import TIFFConverter
from iohub.nd2 import ND2Dataset
from tests.conftest import nd2_tcz


def test_read_images_dispatch():
    reader = read_images(nd2_tcz)
    assert isinstance(reader, ND2Dataset)
    reader.close()


def test_reader_shape_and_dtype():
    with nd2.ND2File(nd2_tcz) as expected, ND2Dataset(nd2_tcz) as reader:
        _, fov = next(iter(reader))
        # Always canonical 5D (T, C, Z, Y, X).
        assert fov.axes_names == ["T", "C", "Z", "Y", "X"]
        assert fov.dtype == expected.dtype
        assert len(reader.channel_names) == reader.channels


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_convert_round_trip(tmp_path, version):
    out = tmp_path / "out.zarr"
    TIFFConverter(nd2_tcz, out, version=version)()
    with nd2.ND2File(nd2_tcz) as expected, open_ome_zarr(out, mode="r", version=version) as plate:
        # nd2 returns axes in native (non-canonical) order, e.g. (Z, C) for
        # some files; align by label to canonical (T, C, Z, Y, X) before
        # comparing so the check holds for any axis combination.
        xda = expected.to_xarray(squeeze=False)
        if "P" in xda.dims:
            xda = xda.isel(P=0)  # first FOV
        for dim in ("T", "C", "Z"):
            if dim not in xda.dims:
                xda = xda.expand_dims(dim)
        exp = xda.transpose("T", "C", "Z", "Y", "X").to_numpy()
        _, pos = next(plate.positions())
        got = pos["0"][:]  # (T, C, Z, Y, X)
        # native dtype preserved (no float16 downcast)
        assert got.dtype == expected.dtype
        # byte-for-byte match, canonical order, no axis swaps
        np.testing.assert_array_equal(got, exp)
