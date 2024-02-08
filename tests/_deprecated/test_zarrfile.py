import numpy as np
import zarr

from iohub.reader import read_micromanager
from iohub.zarrfile import ZarrReader


def test_constructor_mm2gamma(setup_test_data, setup_mm2gamma_zarr):
    """
    test that constructor parses metadata properly
        no data extraction in this test
    """

    _ = setup_test_data
    src = setup_mm2gamma_zarr

    reader = read_micromanager(src)
    assert isinstance(reader, ZarrReader)

    mmr = ZarrReader(src)

    assert mmr.mm_meta is not None
    assert mmr.z_step_size is not None
    assert mmr.width > 0
    assert mmr.height > 0
    assert mmr.frames > 0
    assert mmr.slices > 0
    assert mmr.channels > 0
    assert mmr.rows is not None
    assert mmr.columns is not None
    assert mmr.wells is not None
    assert mmr.hcs_meta is not None

    # Check HCS metadata copy
    meta = mmr.hcs_meta
    assert "plate" in meta.keys()
    assert "well" in meta.keys()
    assert len(meta["well"]) == mmr.get_num_positions()
    assert "images" in meta["well"][0]
    assert len(meta["well"][0]["images"]) != 0
    assert "path" in meta["well"][0]["images"][0]
    assert meta["well"][0]["images"][0]["path"] == "Pos_000"


def test_output_dims_mm2gamma(setup_test_data, setup_mm2gamma_zarr):
    """
    test that output dimensions are always (t, c, z, y, x)
    """

    _ = setup_test_data
    src = setup_mm2gamma_zarr
    mmr = ZarrReader(src)

    assert mmr.get_array(0).shape[0] == mmr.frames
    assert mmr.get_array(0).shape[1] == mmr.channels
    assert mmr.get_array(0).shape[2] == mmr.slices
    assert mmr.get_array(0).shape[3] == mmr.height
    assert mmr.get_array(0).shape[4] == mmr.width


def test_get_zarr_mm2gamma(setup_test_data, setup_mm2gamma_zarr):

    _ = setup_test_data
    src = setup_mm2gamma_zarr
    mmr = ZarrReader(src)

    for i in range(mmr.get_num_positions()):
        z = mmr.get_zarr(i)
        assert isinstance(z, zarr.core.Array)


def test_get_array_mm2gamma(setup_test_data, setup_mm2gamma_zarr):

    _ = setup_test_data
    src = setup_mm2gamma_zarr
    mmr = ZarrReader(src)

    for i in range(mmr.get_num_positions()):
        z = mmr.get_array(i)
        assert z.shape == mmr.shape
        assert isinstance(z, np.ndarray)


def test_get_image_mm2gamma(setup_test_data, setup_mm2gamma_zarr):

    _ = setup_test_data
    src = setup_mm2gamma_zarr
    mmr = ZarrReader(src)

    for i in range(mmr.get_num_positions()):
        z = mmr.get_image(i, t=0, c=0, z=0)
        assert z.shape == (mmr.shape[-2], mmr.shape[-1])
        assert isinstance(z, np.ndarray)


def test_get_num_positions_mm2gamma(setup_test_data, setup_mm2gamma_zarr):

    _ = setup_test_data
    src = setup_mm2gamma_zarr
    mmr = ZarrReader(src)

    assert mmr.get_num_positions() == 4
