import pytest
import numpy as np
import os

from waveorder.io.writer import WaveorderWriter

def test_constructor(setup_folder):
    """
    Test that constructor finds correct save directory

    Returns
    -------

    """
    folder = setup_folder
    writer = WaveorderWriter(folder+'/Test', 'physical')

    assert(writer._WaveorderWriter__save_dir == folder+'/Test')
    assert(writer.datatype == 'physical')

def test_constructor_existing(setup_folder):

    folder = setup_folder

    writer = WaveorderWriter(folder + '/Test', 'physical')
    writer.create_zarr_root('existing.zarr')

    writer_existing = WaveorderWriter(folder+'/Test/existing.zarr', 'physical', alt_name='test_alt')

    assert(writer_existing._WaveorderWriter__root_store_path == folder+'/Test/existing.zarr')
    assert(writer_existing._WaveorderWriter__builder_name == 'test_alt')
    assert(writer_existing.store is not None)
    assert(writer_existing._WaveorderWriter__builder is not None)
    assert(writer_existing._WaveorderWriter__builder.name == 'test_alt')

def test_create_functions(setup_folder):

    folder = setup_folder

    writer = WaveorderWriter(folder + '/Test', 'physical')

    writer.create_zarr_root('test_zarr_root')

    assert(writer._WaveorderWriter__root_store_path == folder+'/Test/test_zarr_root.zarr')
    assert (writer._WaveorderWriter__builder is not None)
    assert(writer._WaveorderWriter__builder.name == 'physical_data')
    assert(writer.store is not None)

    writer.create_position(0, prefix='prefix')

    assert(writer._WaveorderWriter__current_zarr_group is not None)
    assert(writer._WaveorderWriter__current_zarr_group.name == '/prefix_Pos_000.zarr')
    assert(writer.current_group_name == 'prefix_Pos_000.zarr')
    assert(writer.current_position == 0)

    writer.create_position(1, prefix='prefix')

    assert (writer._WaveorderWriter__current_zarr_group is not None)
    assert (writer._WaveorderWriter__current_zarr_group.name == '/prefix_Pos_001.zarr')
    assert (writer.current_group_name == 'prefix_Pos_001.zarr')
    assert (writer.current_position == 1)

    writer.open_position(0, prefix='prefix')

    assert (writer._WaveorderWriter__current_zarr_group is not None)
    assert (writer._WaveorderWriter__current_zarr_group.name == '/prefix_Pos_000.zarr')
    assert (writer.current_group_name == 'prefix_Pos_000.zarr')
    assert (writer.current_position == 0)

def test_init_array(setup_folder):

    folder = setup_folder
    writer = WaveorderWriter(folder + '/Test', 'physical')
    writer.create_zarr_root('test_zarr_root')
    writer.create_position(0)

    data = np.random.rand(3, 3, 65, 128, 128)

    data_shape = data.shape
    chunk_size = (1,1,1,128,128)
    chan_names = ['Phase3D', 'Retardance', 'Other']
    clims = [(-0.5, 0.5), (0,25), (0,10000)]

    writer.init_array(data_shape, chunk_size, chan_names, clims)

    meta_folder = writer.store['Pos_000.zarr']['physical_data']
    meta = meta_folder.attrs.asdict()
    array = meta_folder['array']

    assert(meta_folder is not None)
    assert(array is not None)
    assert(array.shape == data.shape)
    assert(array.chunks == chunk_size)
    assert(array.dtype == 'float32')

    assert(meta is not None)
    assert('multiscales' in meta)
    assert('omero' in meta)
    assert('rdefs' in meta['omero'])

    # Test Chan Names and clims
    for i in range(len(meta['omero']['channels'])):
        assert(meta['omero']['channels'][i]['label'] == chan_names[i])
        assert(meta['omero']['channels'][i]['window']['start'] == clims[i][0])
        assert(meta['omero']['channels'][i]['window']['end'] == clims[i][1])

def test_write(setup_folder):

    folder = setup_folder
    writer = WaveorderWriter(folder + '/Test', 'physical')
    writer.create_zarr_root('test_zarr_root')
    writer.create_position(0)

    data = np.random.rand(3, 3, 65, 128, 128)

    data_shape = data.shape
    chunk_size = (1, 1, 1, 128, 128)
    chan_names = ['Phase3D', 'Retardance', 'Other']
    clims = [(-0.5, 0.5), (0, 25), (0, 10000)]

    writer.init_array(data_shape, chunk_size, chan_names, clims)

    # Write single index for each channel
    writer.write(data[0,0,0], t=0, c=0, z=0)

    # Write full data
    writer.write(data, t=[0,3], c=[0,3], z=[0,65])




















