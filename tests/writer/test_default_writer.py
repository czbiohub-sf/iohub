import pytest
import os
import zarr
import numpy as np
from waveorder.io.writer import WaveorderWriter
from waveorder.io.writer_structures import DefaultZarr, HCSZarr, WriterBase

def test_constructor(setup_writer_folder):
    """
    Test that constructor finds correct save directory

    Returns
    -------

    """
    folder = setup_writer_folder
    writer_def = WaveorderWriter(os.path.join(folder, 'Test'), hcs=False, hcs_meta=None, verbose=False)

    assert(isinstance(writer_def.sub_writer, DefaultZarr))
    assert(isinstance(writer_def.sub_writer, WriterBase))


def test_constructor_existing(setup_writer_folder):
    """
    Test isntantiating the writer into an existing zarr directory

    Parameters
    ----------
    setup_folder

    Returns
    -------

    """

    folder = setup_writer_folder

    writer = WaveorderWriter(os.path.join(folder, 'Test'))
    writer.create_zarr_root('existing.zarr')

    writer_existing = WaveorderWriter(os.path.join(folder, 'Test', 'existing.zarr'))

    assert(writer_existing.sub_writer.root_path == os.path.join(folder, 'Test', 'existing.zarr'))
    assert(writer_existing.sub_writer.store is not None)

def test_create_functions(setup_writer_folder):
    """
    Test create root zarr, create position subfolders, and switching between
    position substores

    Parameters
    ----------
    setup_folder

    Returns
    -------

    """

    folder = setup_writer_folder
    writer = WaveorderWriter(os.path.join(folder, 'Test'), hcs=False, hcs_meta=None, verbose=False)

    writer.create_zarr_root('test_zarr_root')

    assert(writer.sub_writer.root_path == os.path.join(folder, 'Test', 'test_zarr_root.zarr'))
    assert(writer.sub_writer.store is not None)
    assert(isinstance(writer.sub_writer.store['Row_0'], zarr.Group))

    # Check Plate Metadata
    assert('plate' in writer.sub_writer.plate_meta)
    assert('rows' in writer.sub_writer.plate_meta.get('plate').keys())
    assert('columns' in writer.sub_writer.plate_meta.get('plate').keys())
    assert('wells' in writer.sub_writer.plate_meta.get('plate').keys())
    assert(len(writer.sub_writer.plate_meta.get('plate').get('wells')) == 0)
    assert(len(writer.sub_writer.plate_meta.get('plate').get('columns')) == 0)
    assert(len(writer.sub_writer.plate_meta.get('plate').get('rows')) == 1)

    # Check Well metadata
    assert('well' in writer.sub_writer.well_meta)
    assert(len(writer.sub_writer.well_meta.get('well').get('images')) == 0)

def test_init_array(setup_writer_folder):
    """
    Test the correct initialization of desired array and the associated
    metadata

    Parameters
    ----------
    setup_folder

    Returns
    -------

    """

    folder = setup_writer_folder
    writer = WaveorderWriter(os.path.join(folder, 'Test'), hcs=False, hcs_meta=None, verbose=False)
    writer.create_zarr_root('test_zarr_root')

    data_shape = (3, 3, 21, 128, 128) # T, C, Z, Y, X
    chunk_size = (1, 1, 1, 128, 128)
    chan_names = ['State0', 'State1', 'State3']
    clims = [(-0.5, 0.5), (0, 25), (0, 10000)]
    dtype = 'uint16'

    writer.init_array(0, data_shape, chunk_size, chan_names, dtype, clims, position_name=None, overwrite=False)
    writer.init_array(1, data_shape, chunk_size, chan_names, dtype, clims, position_name='Test', overwrite=False)

    assert(isinstance(writer.sub_writer.store['Row_0']['Col_0']['Pos_000'], zarr.Group))
    meta_folder = writer.store['Row_0']['Col_0']['Pos_000']
    meta = meta_folder.attrs.asdict()
    array = meta_folder['arr_0']

    assert(meta_folder is not None)
    assert(array is not None)
    assert(array.shape == data_shape)
    assert(array.chunks == chunk_size)
    assert(array.dtype == dtype)

    assert(meta is not None)
    assert('multiscales' in meta)
    assert('omero' in meta)
    assert('rdefs' in meta['omero'])


    print(meta['omero']['channels'])
    # Test Chan Names and clims
    for i in range(len(meta['omero']['channels'])):
        assert(meta['omero']['channels'][i]['label'] == chan_names[i])
        assert(meta['omero']['channels'][i]['window']['start'] == clims[i][0])
        assert(meta['omero']['channels'][i]['window']['end'] == clims[i][1])

    assert(isinstance(writer.sub_writer.store['Row_0']['Col_1']['Test'], zarr.Group))
    meta_folder = writer.store['Row_0']['Col_1']['Test']
    meta = meta_folder.attrs.asdict()
    array = meta_folder['arr_0']

    assert(meta_folder is not None)
    assert(array is not None)
    assert(array.shape == data_shape)
    assert(array.chunks == chunk_size)
    assert(array.dtype == dtype)

    assert(meta is not None)
    assert('multiscales' in meta)
    assert('omero' in meta)
    assert('rdefs' in meta['omero'])

    # Test Chan Names and clims
    for i in range(len(meta['omero']['channels'])):
        assert(meta['omero']['channels'][i]['label'] == chan_names[i])
        assert(meta['omero']['channels'][i]['window']['start'] == clims[i][0])
        assert(meta['omero']['channels'][i]['window']['end'] == clims[i][1])

def test_write(setup_writer_folder):
    """
    Test the write function of the writer

    Parameters
    ----------
    setup_folder

    Returns
    -------

    """

    folder = setup_writer_folder
    writer = WaveorderWriter(os.path.join(folder, 'Test'), hcs=False, hcs_meta=None, verbose=False)
    writer.create_zarr_root('test_zarr_root')

    data = np.random.randint(1, 60000, size=(3, 3, 11, 128, 128), dtype='uint16')

    data_shape = data.shape
    chunk_size = (1, 1, 1, 128, 128)
    chan_names = ['State0', 'State1', 'State3']
    clims = [(-0.5, 0.5), (0, 25), (0, 10000)]
    dtype = 'uint16'

    writer.init_array(0, data_shape, chunk_size, chan_names, dtype, clims, position_name=None, overwrite=True)

    # Write single index for each channel
    writer.write(data[0, 0, 0], p=0, t=0, c=0, z=0)
    assert(np.array_equal(writer.sub_writer.store['Row_0']['Col_0']['Pos_000']['arr_0'][0, 0, 0], data[0, 0, 0]))

    # Write full data
    writer.write(data, p=0, t=slice(0, 3), c=slice(0, 3), z=slice(0, 11))
    assert(np.array_equal(writer.sub_writer.store['Row_0']['Col_0']['Pos_000']['arr_0'][:, :, :, :, :], data))

    # Write full data with alt method
    writer.write(data, p=0)
    assert (np.array_equal(writer.sub_writer.store['Row_0']['Col_0']['Pos_000']['arr_0'][:, :, :, :, :], data))
