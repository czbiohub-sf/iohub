import numpy as np
import dask.array
from waveorder.io.pycromanager import PycromanagerReader


def test_constructor(setup_test_data, setup_pycromanager_test_data):
    """
    test that the constructor parses metadata properly
    """

    fold = setup_test_data
    # choose a random folder
    first_dir, rand_dir, ptcz_dir = setup_pycromanager_test_data
    mmr = PycromanagerReader(rand_dir)

    assert(mmr.mm_meta is not None)
    assert(mmr.width is not 0)
    assert(mmr.height is not 0)
    assert(mmr.frames is not 0)
    assert(mmr.slices is not 0)
    assert(mmr.channels is not 0)


def test_output_dims(setup_test_data, setup_pycromanager_test_data):
    """
    test that output dimensions are always (t, c, z, y, x)
    """
    fold = setup_test_data

    # run test on 3 random folders
    for i in range(3):
        _, rand_dir, _ = setup_pycromanager_test_data
        mmr = PycromanagerReader(rand_dir)
        za = mmr.get_zarr(0)

        assert(za.shape[0] == mmr.frames)
        assert(za.shape[1] == mmr.channels)
        assert(za.shape[2] == mmr.slices)
        assert(za.shape[3] == mmr.height)
        assert(za.shape[4] == mmr.width)


def test_output_dims_incomplete(setup_test_data, setup_pycromanager_test_data):
    # TODO
    pass


def test_get_zarr(setup_test_data, setup_pycromanager_test_data):
    fold = setup_test_data
    # choose a random folder
    first_dir, rand_dir, ptcz_dir = setup_pycromanager_test_data

    mmr = PycromanagerReader(rand_dir)
    arr = mmr.get_zarr(position=0)
    assert(arr.shape == mmr.shape)
    assert(isinstance(arr, dask.array.Array))


def test_get_array(setup_test_data, setup_pycromanager_test_data):
    fold = setup_test_data
    # choose a random folder
    first_dir, rand_dir, ptcz_dir = setup_pycromanager_test_data

    mmr = PycromanagerReader(rand_dir)
    arr = mmr.get_array(position=0)
    assert (arr.shape == mmr.shape)
    assert(isinstance(arr, np.ndarray))


def test_get_num_positions(setup_test_data, setup_pycromanager_test_data):
    fold = setup_test_data
    # choose a random folder
    first_dir, rand_dir, ptcz_dir = setup_pycromanager_test_data

    mmr = PycromanagerReader(rand_dir)
    assert(mmr.get_num_positions() >= 1)
