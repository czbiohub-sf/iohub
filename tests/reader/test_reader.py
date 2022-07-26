import pytest
import dask.array
import zarr
import numpy as np
from waveorder.io.reader import WaveorderReader
from waveorder.io.singlepagetiff import MicromanagerSequenceReader
from waveorder.io.multipagetiff import MicromanagerOmeTiffReader
from waveorder.io.pycromanager import PycromanagerReader

# todo: consider tests for handling ometiff when singlepagetifff is specified (or vice versa)
# todo: consider tests for handling of positions when extract_data is True and False.


# test exceptions
def test_datatype(setup_test_data, setup_mm2gamma_ome_tiffs):

    fold = setup_test_data
    # choose a specific folder
    _, one_folder, _ = setup_mm2gamma_ome_tiffs
    with pytest.raises(NotImplementedError):
        mmr = WaveorderReader(one_folder,
                                 data_type='unsupportedformat',
                              extract_data=False)


# ===== test mm2gamma =========== #

# test ometiff reader
def test_ometiff_constructor_mm2gamma(setup_test_data, setup_mm2gamma_ome_tiffs):

    fold = setup_test_data
    # choose a specific folder
    _, one_folder, _ = setup_mm2gamma_ome_tiffs
    mmr = WaveorderReader(one_folder,
                          extract_data=False)

    assert (isinstance(mmr.reader, MicromanagerOmeTiffReader))

    assert(mmr.mm_meta is not None)
    assert(mmr.stage_positions is not None)
    assert(mmr.width is not 0)
    assert(mmr.height is not 0)
    assert(mmr.frames is not 0)
    assert(mmr.slices is not 0)
    assert(mmr.channels is not 0)
    if mmr.channels > 1:
        assert(mmr.channel_names is not None)
    if mmr.slices > 1:
        assert(mmr.z_step_size is not None)


def test_ometiff_zarr_mm2gamma(setup_test_data, setup_mm2gamma_ome_tiffs):

    fold = setup_test_data
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = WaveorderReader(rand_folder,
                          extract_data=True)
    assert (isinstance(mmr.reader, MicromanagerOmeTiffReader))

    for i in range(mmr.get_num_positions()):
        z = mmr.get_zarr(i)
        assert (z.shape == mmr.shape)
        assert(isinstance(z, zarr.core.Array))


def test_ometiff_array_mm2gamma(setup_test_data, setup_mm2gamma_ome_tiffs):

    fold = setup_test_data
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = WaveorderReader(rand_folder,
                          extract_data=True)

    assert(isinstance(mmr.reader, MicromanagerOmeTiffReader))

    for i in range(mmr.get_num_positions()):
        z = mmr.get_array(i)
        assert(z.shape == mmr.shape)
        assert(isinstance(z, np.ndarray))


# test sequence reader
def test_sequence_constructor_mm2gamma(setup_test_data, setup_mm2gamma_singlepage_tiffs):

    fold = setup_test_data
    _, one_folder, _ = setup_mm2gamma_singlepage_tiffs
    mmr = WaveorderReader(one_folder,
                          extract_data=False)

    assert(isinstance(mmr.reader, MicromanagerSequenceReader))

    assert(mmr.mm_meta is not None)
    assert(mmr.width is not 0)
    assert(mmr.height is not 0)
    assert(mmr.frames is not 0)
    assert(mmr.slices is not 0)
    assert(mmr.channels is not 0)
    if mmr.channels > 1:
        assert(mmr.channel_names is not None)
    if mmr.slices > 1:
        assert(mmr.z_step_size is not None)


def test_sequence_zarr_mm2gamma(setup_test_data, setup_mm2gamma_singlepage_tiffs):

    fold = setup_test_data
    _, _, rand_folder = setup_mm2gamma_singlepage_tiffs
    mmr = WaveorderReader(rand_folder,
                          extract_data=True)

    assert(isinstance(mmr.reader, MicromanagerSequenceReader))
    for i in range(mmr.get_num_positions()):
        z = mmr.get_zarr(i)
        assert(z.shape == mmr.shape)
        assert(isinstance(z, zarr.core.Array))


def test_sequence_array_zarr_mm2gamma(setup_test_data, setup_mm2gamma_singlepage_tiffs):

    fold = setup_test_data
    _, _, rand_folder = setup_mm2gamma_singlepage_tiffs
    mmr = WaveorderReader(rand_folder,
                          extract_data=True)

    assert(isinstance(mmr.reader, MicromanagerSequenceReader))
    for i in range(mmr.get_num_positions()):
        z = mmr.get_array(i)
        assert(z.shape == mmr.shape)
        assert(isinstance(z, np.ndarray))

# test pycromanager reader
def test_pycromanager_constructor(setup_test_data, setup_pycromanager_test_data):

    fold = setup_test_data
    # choose a specific folder
    first_dir, rand_dir, ptcz_dir = setup_pycromanager_test_data
    mmr = WaveorderReader(rand_dir,
                          extract_data=False)

    assert (isinstance(mmr.reader, PycromanagerReader))

    assert(mmr.mm_meta is not None)
    assert(mmr.width is not 0)
    assert(mmr.height is not 0)
    assert(mmr.frames is not 0)
    assert(mmr.slices is not 0)
    assert(mmr.channels is not 0)
    assert(mmr.stage_positions is not None)
    if mmr.channels > 1:
        assert(mmr.channel_names is not None)
    if mmr.slices > 1:
        assert(mmr.z_step_size is not None)


def test_pycromanager_get_zarr(setup_test_data, setup_pycromanager_test_data):

    fold = setup_test_data
    first_dir, rand_dir, ptcz_dir = setup_pycromanager_test_data
    mmr = WaveorderReader(rand_dir)
    assert (isinstance(mmr.reader, PycromanagerReader))

    for i in range(mmr.get_num_positions()):
        z = mmr.get_zarr(i)
        assert (z.shape == mmr.shape)
        assert(isinstance(z, dask.array.Array))


def test_pycromanager_get_array(setup_test_data, setup_pycromanager_test_data):

    fold = setup_test_data
    first_dir, rand_dir, ptcz_dir = setup_pycromanager_test_data
    mmr = WaveorderReader(rand_dir)
    assert (isinstance(mmr.reader, PycromanagerReader))

    for i in range(mmr.get_num_positions()):
        z = mmr.get_array(i)
        assert (z.shape == mmr.shape)
        assert(isinstance(z, np.ndarray))


# ===== test mm1.4.22  =========== #
def test_ometiff_constructor_mm1422(setup_test_data, setup_mm1422_ome_tiffs):

    fold = setup_test_data
    # choose a specific folder
    _, one_folder, _ = setup_mm1422_ome_tiffs
    mmr = WaveorderReader(one_folder,
                          extract_data=False)

    assert (isinstance(mmr.reader, MicromanagerOmeTiffReader))

    assert(mmr.mm_meta is not None)
    assert(mmr.stage_positions is not None)
    assert(mmr.width is not 0)
    assert(mmr.height is not 0)
    assert(mmr.frames is not 0)
    assert(mmr.slices is not 0)
    assert(mmr.channels is not 0)
    if mmr.channels > 1:
        assert(mmr.channel_names is not None)
    if mmr.slices > 1:
        assert(mmr.z_step_size is not None)



def test_ometiff_zarr_mm1422(setup_test_data, setup_mm1422_ome_tiffs):

    fold = setup_test_data
    _, _, rand_folder = setup_mm1422_ome_tiffs
    mmr = WaveorderReader(rand_folder,
                          extract_data=True)

    assert (isinstance(mmr.reader, MicromanagerOmeTiffReader))

    for i in range(mmr.get_num_positions()):
        z = mmr.get_zarr(i)
        assert (z.shape == mmr.shape)
        assert(isinstance(z, zarr.core.Array))


def test_ometiff_array_zarr_mm1422(setup_test_data, setup_mm1422_ome_tiffs):

    fold = setup_test_data
    _, _, rand_folder = setup_mm1422_ome_tiffs
    mmr = WaveorderReader(rand_folder,
                          extract_data=True)

    assert (isinstance(mmr.reader, MicromanagerOmeTiffReader))

    for i in range(mmr.get_num_positions()):
        z = mmr.get_array(i)
        assert(z.shape == mmr.shape)
        assert(isinstance(z, np.ndarray))


# test sequence constructor
def test_sequence_constructor_mm1422(setup_test_data, setup_mm1422_singlepage_tiffs):

    fold = setup_test_data
    _, one_folder, _ = setup_mm1422_singlepage_tiffs
    mmr = WaveorderReader(one_folder,
                          extract_data=False)

    assert (isinstance(mmr.reader, MicromanagerSequenceReader))

    assert(mmr.mm_meta is not None)
    assert(mmr.width is not 0)
    assert(mmr.height is not 0)
    assert(mmr.frames is not 0)
    assert(mmr.slices is not 0)
    assert(mmr.channels is not 0)
    if mmr.channels > 1:
        assert(mmr.channel_names is not None)
    if mmr.slices > 1:
        assert(mmr.z_step_size is not None)



def test_sequence_zarr_mm1422(setup_test_data, setup_mm1422_singlepage_tiffs):

    fold = setup_test_data
    _, _, rand_folder = setup_mm1422_singlepage_tiffs
    mmr = WaveorderReader(rand_folder,
                          extract_data=True)

    assert (isinstance(mmr.reader, MicromanagerSequenceReader))

    for i in range(mmr.get_num_positions()):
        z = mmr.get_zarr(i)
        assert(z.shape == mmr.shape)
        assert(isinstance(z, zarr.core.Array))


def test_sequence_array_zarr_mm1422(setup_test_data, setup_mm1422_singlepage_tiffs):

    fold = setup_test_data
    _, _, rand_folder = setup_mm1422_singlepage_tiffs
    mmr = WaveorderReader(rand_folder,
                          extract_data=True)

    assert (isinstance(mmr.reader, MicromanagerSequenceReader))

    for i in range(mmr.get_num_positions()):
        z = mmr.get_array(i)
        assert(z.shape == mmr.shape)
        assert(isinstance(z, np.ndarray))


# ===== test property setters =========== #
def test_height(setup_test_data, setup_mm2gamma_ome_tiffs):

    fold = setup_test_data
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = WaveorderReader(rand_folder,
                          extract_data=False)

    assert (isinstance(mmr.reader, MicromanagerOmeTiffReader))

    mmr.height = 100
    assert(mmr.height == mmr.reader.height)


def test_width(setup_test_data, setup_mm2gamma_ome_tiffs):

    fold = setup_test_data
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = WaveorderReader(rand_folder,
                          extract_data=False)

    assert (isinstance(mmr.reader, MicromanagerOmeTiffReader))

    mmr.width = 100
    assert (mmr.width == mmr.reader.width)


def test_frames(setup_test_data, setup_mm2gamma_ome_tiffs):

    fold = setup_test_data
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = WaveorderReader(rand_folder,
                          extract_data=False)

    assert (isinstance(mmr.reader, MicromanagerOmeTiffReader))

    mmr.frames = 100
    assert (mmr.frames == mmr.reader.frames)


def test_slices(setup_test_data, setup_mm2gamma_ome_tiffs):

    fold = setup_test_data
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = WaveorderReader(rand_folder,
                          extract_data=False)

    assert (isinstance(mmr.reader, MicromanagerOmeTiffReader))

    mmr.slices = 100
    assert (mmr.slices == mmr.reader.slices)


def test_channels(setup_test_data, setup_mm2gamma_ome_tiffs):

    fold = setup_test_data
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = WaveorderReader(rand_folder,
                          extract_data=False)

    assert (isinstance(mmr.reader, MicromanagerOmeTiffReader))

    mmr.channels = 100
    assert (mmr.channels == mmr.reader.channels)


def test_channel_names(setup_test_data, setup_mm2gamma_ome_tiffs):

    fold = setup_test_data
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = WaveorderReader(rand_folder,
                          extract_data=False)

    assert (isinstance(mmr.reader, MicromanagerOmeTiffReader))

    mmr.channel_names = 100
    assert (mmr.channel_names == mmr.reader.channel_names)


def test_mm_meta(setup_test_data, setup_mm2gamma_ome_tiffs):

    fold = setup_test_data
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = WaveorderReader(rand_folder,
                          extract_data=False)

    assert (isinstance(mmr.reader, MicromanagerOmeTiffReader))

    mmr.mm_meta = {'newkey': 'newval'}
    assert (mmr.mm_meta == mmr.reader.mm_meta)

    with pytest.raises(AssertionError):
        mmr.mm_meta = 1


def test_stage_positions(setup_test_data, setup_mm2gamma_ome_tiffs):

    fold = setup_test_data
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = WaveorderReader(rand_folder,
                          extract_data=False)

    assert (isinstance(mmr.reader, MicromanagerOmeTiffReader))

    mmr.stage_positions = ['pos one']
    assert (mmr.stage_positions == mmr.reader.stage_positions)

    with pytest.raises(AssertionError):
        mmr.stage_positions = 1

def test_z_step_size(setup_test_data, setup_mm2gamma_ome_tiffs):

    fold = setup_test_data
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = WaveorderReader(rand_folder,
                          extract_data=False)

    assert (isinstance(mmr.reader, MicromanagerOmeTiffReader))

    mmr.z_step_size = 1.75
    assert (mmr.z_step_size == mmr.reader.z_step_size)