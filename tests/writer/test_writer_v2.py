import pytest
import zarr
from waveorder.io.writer import WaveorderWriter

def test_default_writer():


    data_path = '/Users/cameron.foltz/Desktop/Test_Data/Writer_Testing_OME/B1_small_noRawData.zarr'
    out_path = '/Users/cameron.foltz/Desktop/Test_Data/Writer_Testing_OME/Writer_Output/'
    store = zarr.open(data_path)
    array = store['Fake_Row']['Fake_Col_0']['Pos_000']['array']
    omero_meta = store['Fake_Row']['Fake_Col_0']['Pos_000'].attrs.asdict()
    chan_names = []
    clims = []
    for i in range(len(omero_meta['omero']['channels'])):
        chan_names.append(omero_meta['omero']['channels'][i]['label'])
        clims.append((omero_meta['omero']['channels'][i]['window']['start'],
                      omero_meta['omero']['channels'][i]['window']['end']))


    writer = WaveorderWriter(out_path)
    writer.create_zarr_root('B1_small_noRawData.zarr')
    for i in range(4):
        writer.create_position(i)
        writer.init_array(data_shape=array.shape, chunk_size=(1,1,1,2048,2048), dtype='uint16',
                          chan_names=chan_names, clims=clims)

        data = store['Fake_Row'][f'Fake_Col_{i}']['Pos_000']['array']
        writer.write(data)

def test_hcs_writer():

    data_path = '/Users/cameron.foltz/Desktop/Test_Data/Writer_Testing_OME/B1_small_noRawDataError.zarr'
    out_path = '/Users/cameron.foltz/Desktop/Test_Data/Writer_Testing_OME/Writer_Output/'
    store = zarr.open(data_path)
    array = store['Fake_Row']['Fake_Col_0']['Pos_000']['array']

    hcs_meta = dict()
    hcs_meta['plate'] = store.attrs.asdict()['plate']
    well_meta = []
    for i in range(4):
        well_meta.append(store['Fake_Row'][f'Fake_Col_{i}'].attrs.asdict()['well'])

    hcs_meta['well'] = well_meta
    omero_meta = store['Fake_Row']['Fake_Col_0']['Pos_000'].attrs.asdict()
    chan_names = []
    clims = []
    for i in range(len(omero_meta['omero']['channels'])):
        chan_names.append(omero_meta['omero']['channels'][i]['label'])
        clims.append((omero_meta['omero']['channels'][i]['window']['start'],
                      omero_meta['omero']['channels'][i]['window']['end']))

    writer = WaveorderWriter(out_path, hcs=True, hcs_meta=hcs_meta)
    writer.create_zarr_root('B1_small_noRawDataHCSError.zarr')
    for i in range(4):
        writer.create_position(i)
        writer.init_array(data_shape=array.shape, chunk_size=(1,1,1,2048,2048), dtype='uint16',
                          chan_names=chan_names, clims=clims)

        data = store['Fake_Row'][f'Fake_Col_{i}'][f'Pos_00{i}']['array']
        writer.write(data)



