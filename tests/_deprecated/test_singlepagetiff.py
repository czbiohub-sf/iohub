import numpy as np
import zarr

from iohub._deprecated.singlepagetiff import MicromanagerSequenceReader
from tests.conftest import (
    mm2gamma_singlepage_tiffs,
    mm2gamma_singlepage_tiffs_incomplete,
    mm1422_singlepage_tiffs,
)


def pytest_generate_tests(metafunc):
    if "single_page_tiff" in metafunc.fixturenames:
        metafunc.parametrize(
            "single_page_tiff",
            mm2gamma_singlepage_tiffs + mm1422_singlepage_tiffs,
        )


def test_constructor(single_page_tiff):
    """
    test that constructor parses metadata properly
        no data extraction in this test
    """
    mmr = MicromanagerSequenceReader(single_page_tiff, extract_data=False)
    assert mmr.micromanager_metadata is not None
    assert mmr.width > 0
    assert mmr.height > 0
    assert mmr.frames > 0
    assert mmr.slices > 0
    assert mmr.channels > 0


def test_output_dims(single_page_tiff):
    """
    test that output dimensions are always (t, c, z, y, x)
    """
    mmr = MicromanagerSequenceReader(single_page_tiff, extract_data=False)
    assert mmr.get_zarr(0).shape[0] == mmr.frames
    assert mmr.get_zarr(0).shape[1] == mmr.channels
    assert mmr.get_zarr(0).shape[2] == mmr.slices
    assert mmr.get_zarr(0).shape[3] == mmr.height
    assert mmr.get_zarr(0).shape[4] == mmr.width


def test_output_dims_incomplete():
    """
    test that output dimensions are correct for interrupted data
    """
    mmr = MicromanagerSequenceReader(
        mm2gamma_singlepage_tiffs_incomplete, extract_data=True
    )
    assert mmr.get_zarr(0).shape[0] == mmr.frames
    assert mmr.get_zarr(0).shape[1] == mmr.channels
    assert mmr.get_zarr(0).shape[2] == mmr.slices
    assert mmr.get_zarr(0).shape[3] == mmr.height
    assert mmr.get_zarr(0).shape[4] == mmr.width
    assert mmr.get_zarr(0).shape[0] == 11


def test_get_zarr(single_page_tiff):
    mmr = MicromanagerSequenceReader(single_page_tiff, extract_data=True)
    for i in range(mmr.get_num_positions()):
        z = mmr.get_zarr(i)
        assert z.shape == mmr.shape
        assert isinstance(z, zarr.core.Array)


def test_get_array(single_page_tiff):
    mmr = MicromanagerSequenceReader(single_page_tiff, extract_data=True)
    for i in range(mmr.get_num_positions()):
        z = mmr.get_array(i)
        assert z.shape == mmr.shape
        assert isinstance(z, np.ndarray)


def test_get_num_positions(single_page_tiff):
    mmr = MicromanagerSequenceReader(single_page_tiff, extract_data=True)
    assert mmr.get_num_positions() >= 1
