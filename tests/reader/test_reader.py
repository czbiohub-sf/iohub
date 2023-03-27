import dask.array
import numpy as np
import pytest
import zarr

from iohub.multipagetiff import MicromanagerOmeTiffReader
from iohub.ndtiff import NDTiffReader
from iohub.reader import read_micromanager
from iohub.singlepagetiff import MicromanagerSequenceReader

# todo: consider tests for handling ometiff
# when singlepagetifff is specified (or vice versa)
# todo: consider tests for handling of positions
# when extract_data is True and False.


# test exceptions
def test_datatype(setup_test_data, setup_mm2gamma_ome_tiffs):
    _ = setup_test_data
    # choose a specific folder
    _, one_folder, _ = setup_mm2gamma_ome_tiffs
    with pytest.raises(ValueError):
        _ = read_micromanager(
            one_folder, data_type="unsupportedformat", extract_data=False
        )


# ===== test mm2gamma =========== #


# test ometiff reader
def test_ometiff_constructor_mm2gamma(
    setup_test_data, setup_mm2gamma_ome_tiffs
):
    _ = setup_test_data
    # choose a specific folder
    _, one_folder, _ = setup_mm2gamma_ome_tiffs
    mmr = read_micromanager(one_folder, extract_data=False)

    assert isinstance(mmr, MicromanagerOmeTiffReader)

    assert mmr.mm_meta is not None
    assert mmr.stage_positions is not None
    assert mmr.width > 0
    assert mmr.height > 0
    assert mmr.frames > 0
    assert mmr.slices > 0
    assert mmr.channels > 0
    if mmr.channels > 1:
        assert mmr.channel_names is not None
    if mmr.slices > 1:
        assert mmr.z_step_size is not None


def test_ometiff_zarr_mm2gamma(setup_test_data, setup_mm2gamma_ome_tiffs):
    _ = setup_test_data
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = read_micromanager(rand_folder, extract_data=True)
    assert isinstance(mmr, MicromanagerOmeTiffReader)

    for i in range(mmr.get_num_positions()):
        z = mmr.get_zarr(i)
        assert z.shape == mmr.shape
        assert isinstance(z, zarr.core.Array)


def test_ometiff_array_mm2gamma(setup_test_data, setup_mm2gamma_ome_tiffs):
    _ = setup_test_data
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = read_micromanager(rand_folder, extract_data=True)

    assert isinstance(mmr, MicromanagerOmeTiffReader)

    for i in range(mmr.get_num_positions()):
        z = mmr.get_array(i)
        assert z.shape == mmr.shape
        assert isinstance(z, np.ndarray)


# test sequence reader
def test_sequence_constructor_mm2gamma(
    setup_test_data, setup_mm2gamma_singlepage_tiffs
):
    _ = setup_test_data
    _, one_folder, _ = setup_mm2gamma_singlepage_tiffs
    mmr = read_micromanager(one_folder, extract_data=False)

    assert isinstance(mmr, MicromanagerSequenceReader)

    assert mmr.mm_meta is not None
    assert mmr.width > 0
    assert mmr.height > 0
    assert mmr.frames > 0
    assert mmr.slices > 0
    assert mmr.channels > 0
    if mmr.channels > 1:
        assert mmr.channel_names is not None
    if mmr.slices > 1:
        assert mmr.z_step_size is not None


def test_sequence_zarr_mm2gamma(
    setup_test_data, setup_mm2gamma_singlepage_tiffs
):
    _ = setup_test_data
    _, _, rand_folder = setup_mm2gamma_singlepage_tiffs
    mmr = read_micromanager(rand_folder, extract_data=True)

    assert isinstance(mmr, MicromanagerSequenceReader)
    for i in range(mmr.get_num_positions()):
        z = mmr.get_zarr(i)
        assert z.shape == mmr.shape
        assert isinstance(z, zarr.core.Array)


def test_sequence_array_zarr_mm2gamma(
    setup_test_data, setup_mm2gamma_singlepage_tiffs
):
    _ = setup_test_data
    _, _, rand_folder = setup_mm2gamma_singlepage_tiffs
    mmr = read_micromanager(rand_folder, extract_data=True)

    assert isinstance(mmr, MicromanagerSequenceReader)
    for i in range(mmr.get_num_positions()):
        z = mmr.get_array(i)
        assert z.shape == mmr.shape
        assert isinstance(z, np.ndarray)


# test pycromanager reader
def test_pycromanager_constructor(
    setup_test_data, setup_pycromanager_test_data
):
    _ = setup_test_data
    # choose a specific folder
    first_dir, rand_dir, ptcz_dir = setup_pycromanager_test_data
    mmr = read_micromanager(rand_dir, extract_data=False)

    assert isinstance(mmr, NDTiffReader)

    assert mmr.mm_meta is not None
    assert mmr.width > 0
    assert mmr.height > 0
    assert mmr.frames > 0
    assert mmr.slices > 0
    assert mmr.channels > 0
    assert mmr.stage_positions is not None
    if mmr.channels > 1:
        assert mmr.channel_names is not None
    if mmr.slices > 1:
        assert mmr.z_step_size is not None


def test_pycromanager_get_zarr(setup_test_data, setup_pycromanager_test_data):
    _ = setup_test_data
    first_dir, rand_dir, ptcz_dir = setup_pycromanager_test_data
    mmr = read_micromanager(rand_dir)
    assert isinstance(mmr, NDTiffReader)

    for i in range(mmr.get_num_positions()):
        z = mmr.get_zarr(i)
        assert z.shape == mmr.shape
        assert isinstance(z, dask.array.Array)


def test_pycromanager_get_array(setup_test_data, setup_pycromanager_test_data):
    _ = setup_test_data
    first_dir, rand_dir, ptcz_dir = setup_pycromanager_test_data
    mmr = read_micromanager(rand_dir)
    assert isinstance(mmr, NDTiffReader)

    for i in range(mmr.get_num_positions()):
        z = mmr.get_array(i)
        assert z.shape == mmr.shape
        assert isinstance(z, np.ndarray)


# ===== test mm1.4.22  =========== #
def test_ometiff_constructor_mm1422(setup_test_data, setup_mm1422_ome_tiffs):
    _ = setup_test_data
    # choose a specific folder
    _, one_folder, _ = setup_mm1422_ome_tiffs
    mmr = read_micromanager(one_folder, extract_data=False)

    assert isinstance(mmr, MicromanagerOmeTiffReader)

    assert mmr.mm_meta is not None
    assert mmr.stage_positions is not None
    assert mmr.width > 0
    assert mmr.height > 0
    assert mmr.frames > 0
    assert mmr.slices > 0
    assert mmr.channels > 0
    if mmr.channels > 1:
        assert mmr.channel_names is not None
    if mmr.slices > 1:
        assert mmr.z_step_size is not None


def test_ometiff_zarr_mm1422(setup_test_data, setup_mm1422_ome_tiffs):
    _ = setup_test_data
    _, _, rand_folder = setup_mm1422_ome_tiffs
    mmr = read_micromanager(rand_folder, extract_data=True)

    assert isinstance(mmr, MicromanagerOmeTiffReader)

    for i in range(mmr.get_num_positions()):
        z = mmr.get_zarr(i)
        assert z.shape == mmr.shape
        assert isinstance(z, zarr.core.Array)


def test_ometiff_array_zarr_mm1422(setup_test_data, setup_mm1422_ome_tiffs):
    _ = setup_test_data
    _, _, rand_folder = setup_mm1422_ome_tiffs
    mmr = read_micromanager(rand_folder, extract_data=True)

    assert isinstance(mmr, MicromanagerOmeTiffReader)

    for i in range(mmr.get_num_positions()):
        z = mmr.get_array(i)
        assert z.shape == mmr.shape
        assert isinstance(z, np.ndarray)


# test sequence constructor
def test_sequence_constructor_mm1422(
    setup_test_data, setup_mm1422_singlepage_tiffs
):
    _ = setup_test_data
    _, one_folder, _ = setup_mm1422_singlepage_tiffs
    mmr = read_micromanager(one_folder, extract_data=False)

    assert isinstance(mmr, MicromanagerSequenceReader)

    assert mmr.mm_meta is not None
    assert mmr.width > 0
    assert mmr.height > 0
    assert mmr.frames > 0
    assert mmr.slices > 0
    assert mmr.channels > 0
    if mmr.channels > 1:
        assert mmr.channel_names is not None
    if mmr.slices > 1:
        assert mmr.z_step_size is not None


def test_sequence_zarr_mm1422(setup_test_data, setup_mm1422_singlepage_tiffs):
    _ = setup_test_data
    _, _, rand_folder = setup_mm1422_singlepage_tiffs
    mmr = read_micromanager(rand_folder, extract_data=True)

    assert isinstance(mmr, MicromanagerSequenceReader)

    for i in range(mmr.get_num_positions()):
        z = mmr.get_zarr(i)
        assert z.shape == mmr.shape
        assert isinstance(z, zarr.core.Array)


def test_sequence_array_zarr_mm1422(
    setup_test_data, setup_mm1422_singlepage_tiffs
):
    _ = setup_test_data
    _, _, rand_folder = setup_mm1422_singlepage_tiffs
    mmr = read_micromanager(rand_folder, extract_data=True)

    assert isinstance(mmr, MicromanagerSequenceReader)

    for i in range(mmr.get_num_positions()):
        z = mmr.get_array(i)
        assert z.shape == mmr.shape
        assert isinstance(z, np.ndarray)


# ===== test property setters =========== #
def test_height(setup_test_data, setup_mm2gamma_ome_tiffs):
    _ = setup_test_data
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = read_micromanager(rand_folder, extract_data=False)

    assert isinstance(mmr, MicromanagerOmeTiffReader)

    mmr.height = 100
    assert mmr.height == mmr.height


def test_width(setup_test_data, setup_mm2gamma_ome_tiffs):
    _ = setup_test_data
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = read_micromanager(rand_folder, extract_data=False)

    assert isinstance(mmr, MicromanagerOmeTiffReader)

    mmr.width = 100
    assert mmr.width == mmr.width


def test_frames(setup_test_data, setup_mm2gamma_ome_tiffs):
    _ = setup_test_data
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = read_micromanager(rand_folder, extract_data=False)

    assert isinstance(mmr, MicromanagerOmeTiffReader)

    mmr.frames = 100
    assert mmr.frames == mmr.frames


def test_slices(setup_test_data, setup_mm2gamma_ome_tiffs):
    _ = setup_test_data
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = read_micromanager(rand_folder, extract_data=False)

    assert isinstance(mmr, MicromanagerOmeTiffReader)

    mmr.slices = 100
    assert mmr.slices == mmr.slices


def test_channels(setup_test_data, setup_mm2gamma_ome_tiffs):
    _ = setup_test_data
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = read_micromanager(rand_folder, extract_data=False)

    assert isinstance(mmr, MicromanagerOmeTiffReader)

    mmr.channels = 100
    assert mmr.channels == mmr.channels


def test_channel_names(setup_test_data, setup_mm2gamma_ome_tiffs):
    _ = setup_test_data
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = read_micromanager(rand_folder, extract_data=False)

    assert isinstance(mmr, MicromanagerOmeTiffReader)

    mmr.channel_names = 100
    assert mmr.channel_names == mmr.channel_names


def test_mm_meta(setup_test_data, setup_mm2gamma_ome_tiffs):
    _ = setup_test_data
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = read_micromanager(rand_folder, extract_data=False)

    assert isinstance(mmr, MicromanagerOmeTiffReader)

    mmr.mm_meta = {"newkey": "newval"}
    assert mmr.mm_meta == mmr.mm_meta

    with pytest.raises(TypeError):
        mmr.mm_meta = 1


def test_stage_positions(setup_test_data, setup_mm2gamma_ome_tiffs):
    _ = setup_test_data
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = read_micromanager(rand_folder, extract_data=False)

    assert isinstance(mmr, MicromanagerOmeTiffReader)

    mmr.stage_positions = ["pos one"]
    assert mmr.stage_positions == mmr.stage_positions

    with pytest.raises(TypeError):
        mmr.stage_positions = 1


def test_z_step_size(setup_test_data, setup_mm2gamma_ome_tiffs):
    _ = setup_test_data
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = read_micromanager(rand_folder, extract_data=False)

    assert isinstance(mmr, MicromanagerOmeTiffReader)

    mmr.z_step_size = 1.75
    assert mmr.z_step_size == mmr.z_step_size
