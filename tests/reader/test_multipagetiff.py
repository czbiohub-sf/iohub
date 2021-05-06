import numpy as np
import pytest
import os
import random

from waveorder.io.multipagetiff import MicromanagerOmeTiffReader


def test_constructor(setup_mm2gamma_ome_tiffs):
    """
    test that constructor parses metadata properly
        no data extraction in this test
    Parameters
    ----------
    setup_mm2gamma_ome_tiffs

    Returns
    -------

    """

    # choose a specific folder
    _, one_folder, _ = setup_mm2gamma_ome_tiffs
    mmr = MicromanagerOmeTiffReader(one_folder, extract_data=False)

    assert(mmr.mm_meta is not None)
    assert(mmr.width is not 0)
    assert(mmr.height is not 0)
    assert(mmr.frames is not 0)
    assert(mmr.slices is not 0)
    assert(mmr.channels is not 0)
    assert(mmr.master_ome_tiff is not None)


def test_output_dims(setup_mm2gamma_ome_tiffs):
    """
    test that output dimensions are always (t, c, z, y, x)
    Parameters
    ----------
    setup_mm2gamma_ome_tiffs

    Returns
    -------

    """

    # choose a random folder
    _, _, rand_folder = setup_mm2gamma_ome_tiffs
    mmr = MicromanagerOmeTiffReader(rand_folder, extract_data=True)

    assert(mmr.get_zarr(0).shape[0] == mmr.frames)
    assert(mmr.get_zarr(0).shape[1] == mmr.channels)
    assert(mmr.get_zarr(0).shape[2] == mmr.slices)
    assert(mmr.get_zarr(0).shape[3] == mmr.height)
    assert(mmr.get_zarr(0).shape[4] == mmr.width)

