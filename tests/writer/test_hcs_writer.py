import pytest
import os
import zarr
import numpy as np
from waveorder.io.writer import WaveorderWriter
from waveorder.io.writer_structures import DefaultZarr, HCSZarr, WriterBase

hcs_meta = {'plate': {
    'acquisitions': [{'id': 1,
                      'maximumfieldcount': 2,
                      'name': 'Dataset',
                      'starttime': 0}],
    'columns': [{'name': '1'},
                {'name': '2'},
                {'name': '3'},
                {'name': '4'}],
    'field_count': 2,
    'name': 'MultiWell_Plate_Example',
    'rows': [{'name': 'A'},
             {'name': 'B'},
             {'name': 'C'},
             {'name': 'D'}],
    'version': '0.1',
    'wells': [{'path': 'A/1'},
              {'path': 'A/2'},
              {'path': 'A/3'},
              {'path': 'A/4'},
              {'path': 'B/1'},
              {'path': 'B/2'},
              {'path': 'B/3'},
              {'path': 'B/4'},
              {'path': 'C/1'},
              {'path': 'C/2'},
              {'path': 'C/3'},
              {'path': 'C/4'},
              {'path': 'D/1'},
              {'path': 'D/2'},
              {'path': 'D/3'},
              {'path': 'D/4'}]},
    'well': [{'images': [{'path': 'FOV1'}, {'path': 'FOV2'}]},
             {'images': [{'path': 'FOV1'}, {'path': 'FOV2'}]},
             {'images': [{'path': 'FOV1'}, {'path': 'FOV2'}]},
             {'images': [{'path': 'FOV1'}, {'path': 'FOV2'}]},
             {'images': [{'path': 'FOV1'}, {'path': 'FOV2'}]},
             {'images': [{'path': 'FOV1'}, {'path': 'FOV2'}]},
             {'images': [{'path': 'FOV1'}, {'path': 'FOV2'}]},
             {'images': [{'path': 'FOV1'}, {'path': 'FOV2'}]},
             {'images': [{'path': 'FOV1'}, {'path': 'FOV2'}]},
             {'images': [{'path': 'FOV1'}, {'path': 'FOV2'}]},
             {'images': [{'path': 'FOV1'}, {'path': 'FOV2'}]},
             {'images': [{'path': 'FOV1'}, {'path': 'FOV2'}]},
             {'images': [{'path': 'FOV1'}, {'path': 'FOV2'}]},
             {'images': [{'path': 'FOV1'}, {'path': 'FOV2'}]},
             {'images': [{'path': 'FOV1'}, {'path': 'FOV2'}]},
             {'images': [{'path': 'FOV1'}, {'path': 'FOV2'}]}]
}

def test_constructor(setup_writer_folder):
    """
    Test that constructor finds correct save directory

    Returns
    -------

    """
    folder = setup_writer_folder
    writer_def = WaveorderWriter(os.path.join(folder, 'Test'), hcs=True, hcs_meta=hcs_meta, verbose=False)

    assert(isinstance(writer_def.sub_writer, HCSZarr))
    assert(isinstance(writer_def.sub_writer, WriterBase))

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
    writer = WaveorderWriter(os.path.join(folder, 'Test'), hcs=True, hcs_meta=hcs_meta, verbose=False)

    writer.create_zarr_root('test_zarr_root')

    assert(writer.sub_writer.root_path == os.path.join(folder, 'Test', 'test_zarr_root.zarr'))
    assert(writer.sub_writer.store is not None)

    # Check that the correct hierarchy was initialized
    cnt = 0
    for row in hcs_meta['plate']['rows']:
        for col in hcs_meta['plate']['columns']:
            for fov in range(1,3):
                path = f'{row["name"]}/{col["name"]}/FOV{fov}'
                assert(isinstance(writer.sub_writer.store[path], zarr.Group))
                assert(writer.sub_writer.positions[cnt] == {'name': f'FOV{fov}',
                                                            'row': row["name"],
                                                            'col': col["name"]})
                cnt += 1

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
    writer = WaveorderWriter(os.path.join(folder, 'Test'), hcs=True, hcs_meta=hcs_meta, verbose=False)
    writer.create_zarr_root('test_zarr_root')

    data_shape = (3, 3, 21, 128, 128) # T, C, Z, Y, X
    chunk_size = (1, 1, 1, 128, 128)
    chan_names = ['State0', 'State1', 'State3']
    clims = [(-0.5, 0.5), (0, 25), (0, 10000)]
    dtype = 'uint16'

    writer.init_array(0, data_shape, chunk_size, chan_names, dtype, clims, position_name=None, overwrite=False)
    writer.init_array(11, data_shape, chunk_size, chan_names, dtype, clims, position_name='Test', overwrite=False)

    assert(isinstance(writer.sub_writer.store['A']['1']['FOV1'], zarr.Group))
    meta_folder = writer.store['A']['1']['FOV1']
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

    assert(isinstance(writer.sub_writer.store['B']['2']['FOV2'], zarr.Group))
    meta_folder = writer.store['B']['2']['FOV2']
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
    writer = WaveorderWriter(os.path.join(folder, 'Test'), hcs=True, hcs_meta=hcs_meta, verbose=False)
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
    assert(np.array_equal(writer.sub_writer.store['A']['1']['FOV1']['arr_0'][0, 0, 0], data[0, 0, 0]))

    # Write full data
    writer.write(data, p=0)
    assert(np.array_equal(writer.sub_writer.store['A']['1']['FOV1']['arr_0'][:, :, :, :, :], data))

    # Write full data with alt method
    writer.write(data, p=0, t=slice(0, 3), c=slice(0, 3), z=slice(0, 11))
    assert(np.array_equal(writer.sub_writer.store['A']['1']['FOV1']['arr_0'][:, :, :, :, :], data))
