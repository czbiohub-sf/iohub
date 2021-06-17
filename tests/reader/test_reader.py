import pytest
import zarr
import numpy as np
from waveorder.io.reader import MicromanagerReader

# todo: consider tests for handling ometiff when singlepagetifff is specified (or vice versa)
# todo: consider tests for handling of positions when extract_data is True and False.


# test exceptions
def test_datatype(setup_mm2gamma_ome_tiffs):
    # choose a specific folder
    _, one_folder, _ = setup_mm2gamma_ome_tiffs
    with pytest.raises(NotImplementedError):
        mmr = MicromanagerReader(one_folder,
                                 'unsupportedformat',
                                 extract_data=False)


# ===== test mm2gamma =========== #
def test_ometiff_constructor_mm2gamma(setup_mm2gamma_ome_tiffs):
    # choose a specific folder
    _, one_folder, _ = setup_mm2gamma_ome_tiffs
    mmr = MicromanagerReader(one_folder,
                             'ometiff',
                             extract_data=False)

    assert(mmr.mm_meta is not None)
    assert(mmr.stage_positions is not None)
    assert(mmr.z_step_size is not None)
    assert(mmr.width is not 0)
    assert(mmr.height is not 0)
    assert(mmr.frames is not 0)
    assert(mmr.slices is not 0)
    assert(mmr.channels is not 0)
    assert(mmr.channel_names is not None)


def test_ometiff_zarr_mm2gamma(setup_mm2gamma_ome_tiffs):
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = MicromanagerReader(rand_folder,
                             'ometiff',
                             extract_data=True)
    for i in range(mmr.get_num_positions()):
        z = mmr.get_zarr(i)
        assert (z.shape == mmr.shape)
        assert(isinstance(z, zarr.core.Array))


def test_ometiff_array_mm2gamma(setup_mm2gamma_ome_tiffs):
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = MicromanagerReader(rand_folder,
                             'ometiff',
                             extract_data=True)
    for i in range(mmr.get_num_positions()):
        z = mmr.get_array(i)
        assert(z.shape == mmr.shape)
        assert(isinstance(z, np.ndarray))


# test sequence constructor
def test_sequence_constructor_mm2gamma(setup_mm2gamma_singlepage_tiffs):
    _, one_folder, _ = setup_mm2gamma_singlepage_tiffs
    mmr = MicromanagerReader(one_folder,
                             'singlepagetiff',
                             extract_data=False)
    assert(mmr.mm_meta is not None)
    assert(mmr.z_step_size is not None)
    assert(mmr.width is not 0)
    assert(mmr.height is not 0)
    assert(mmr.frames is not 0)
    assert(mmr.slices is not 0)
    assert(mmr.channels is not 0)


def test_sequence_zarr_mm2gamma(setup_mm2gamma_singlepage_tiffs):
    _, _, rand_folder = setup_mm2gamma_singlepage_tiffs
    mmr = MicromanagerReader(rand_folder,
                             'singlepagetiff',
                             extract_data=True)
    for i in range(mmr.get_num_positions()):
        z = mmr.get_zarr(i)
        assert(z.shape == mmr.shape)
        assert(isinstance(z, zarr.core.Array))


def test_sequence_array_zarr_mm2gamma(setup_mm2gamma_singlepage_tiffs):
    _, _, rand_folder = setup_mm2gamma_singlepage_tiffs
    mmr = MicromanagerReader(rand_folder,
                             'singlepagetiff',
                             extract_data=True)
    for i in range(mmr.get_num_positions()):
        z = mmr.get_array(i)
        assert(z.shape == mmr.shape)
        assert(isinstance(z, np.ndarray))


# ===== test mm1.4.22  =========== #
def test_ometiff_constructor_mm1422(setup_mm1422_ome_tiffs):
    # choose a specific folder
    _, one_folder, _ = setup_mm1422_ome_tiffs
    mmr = MicromanagerReader(one_folder,
                             'ometiff',
                             extract_data=False)

    assert(mmr.mm_meta is not None)
    assert(mmr.stage_positions is not None)
    assert(mmr.z_step_size is not None)
    assert(mmr.width is not 0)
    assert(mmr.height is not 0)
    assert(mmr.frames is not 0)
    assert(mmr.slices is not 0)
    assert(mmr.channels is not 0)
    assert(mmr.channel_names is not None)


def test_ometiff_zarr_mm1422(setup_mm1422_ome_tiffs):
    _, _, rand_folder = setup_mm1422_ome_tiffs
    mmr = MicromanagerReader(rand_folder,
                             'ometiff',
                             extract_data=True)
    for i in range(mmr.get_num_positions()):
        z = mmr.get_zarr(i)
        assert (z.shape == mmr.shape)
        assert(isinstance(z, zarr.core.Array))


def test_ometiff_array_zarr_mm1422(setup_mm1422_ome_tiffs):
    _, _, rand_folder = setup_mm1422_ome_tiffs
    mmr = MicromanagerReader(rand_folder,
                             'ometiff',
                             extract_data=True)
    for i in range(mmr.get_num_positions()):
        z = mmr.get_array(i)
        assert(z.shape == mmr.shape)
        assert(isinstance(z, np.ndarray))


# test sequence constructor
def test_sequence_constructor_mm1422(setup_mm1422_singlepage_tiffs):
    _, one_folder, _ = setup_mm1422_singlepage_tiffs
    mmr = MicromanagerReader(one_folder,
                             'singlepagetiff',
                             extract_data=False)
    assert(mmr.mm_meta is not None)
    assert(mmr.z_step_size is not None)
    assert(mmr.width is not 0)
    assert(mmr.height is not 0)
    assert(mmr.frames is not 0)
    assert(mmr.slices is not 0)
    assert(mmr.channels is not 0)


def test_sequence_zarr_mm1422(setup_mm1422_singlepage_tiffs):
    _, _, rand_folder = setup_mm1422_singlepage_tiffs
    mmr = MicromanagerReader(rand_folder,
                             'singlepagetiff',
                             extract_data=True)
    for i in range(mmr.get_num_positions()):
        z = mmr.get_zarr(i)
        assert(z.shape == mmr.shape)
        assert(isinstance(z, zarr.core.Array))


def test_sequence_array_zarr_mm1422(setup_mm1422_singlepage_tiffs):
    _, _, rand_folder = setup_mm1422_singlepage_tiffs
    mmr = MicromanagerReader(rand_folder,
                             'singlepagetiff',
                             extract_data=True)
    for i in range(mmr.get_num_positions()):
        z = mmr.get_array(i)
        assert(z.shape == mmr.shape)
        assert(isinstance(z, np.ndarray))


# ===== test property setters =========== #
def test_height(setup_mm2gamma_ome_tiffs):
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = MicromanagerReader(rand_folder,
                             'ometiff',
                             extract_data=False)
    mmr.height = 100
    assert(mmr.height == mmr.reader.height)


def test_width(setup_mm2gamma_ome_tiffs):
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = MicromanagerReader(rand_folder,
                             'ometiff',
                             extract_data=False)
    mmr.width = 100
    assert (mmr.width == mmr.reader.width)


def test_frames(setup_mm2gamma_ome_tiffs):
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = MicromanagerReader(rand_folder,
                             'ometiff',
                             extract_data=False)
    mmr.frames = 100
    assert (mmr.frames == mmr.reader.frames)


def test_slices(setup_mm2gamma_ome_tiffs):
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = MicromanagerReader(rand_folder,
                             'ometiff',
                             extract_data=False)
    mmr.slices = 100
    assert (mmr.slices == mmr.reader.slices)


def test_channels(setup_mm2gamma_ome_tiffs):
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = MicromanagerReader(rand_folder,
                             'ometiff',
                             extract_data=False)
    mmr.channels = 100
    assert (mmr.channels == mmr.reader.channels)


def test_channel_names(setup_mm2gamma_ome_tiffs):
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = MicromanagerReader(rand_folder,
                             'ometiff',
                             extract_data=False)
    mmr.channel_names = 100
    assert (mmr.channel_names == mmr.reader.channel_names)


def test_mm_meta(setup_mm2gamma_ome_tiffs):
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = MicromanagerReader(rand_folder,
                             'ometiff',
                             extract_data=False)
    mmr.mm_meta = {'newkey': 'newval'}
    assert (mmr.mm_meta == mmr.reader.mm_meta)

    with pytest.raises(AssertionError):
        mmr.mm_meta = 1


def test_stage_positions(setup_mm2gamma_ome_tiffs):
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = MicromanagerReader(rand_folder,
                             'ometiff',
                             extract_data=False)
    mmr.stage_positions = ['pos one']
    assert (mmr.stage_positions == mmr.reader.stage_positions)

    with pytest.raises(AssertionError):
        mmr.stage_positions = 1

def test_z_step_size(setup_mm2gamma_ome_tiffs):
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = MicromanagerReader(rand_folder,
                             'ometiff',
                             extract_data=False)
    mmr.z_step_size = 1.75
    assert (mmr.z_step_size == mmr.reader.z_step_size)