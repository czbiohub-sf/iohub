import pytest
import os
import shutil
import zarr
import tifffile as tiff
from waveorder.io import WaveorderReader, WaveorderWriter
import glob
import numpy as np
from recOrder.io.zarr_converter import ZarrConverter

def test_converter_initialize(setup_data_save_folder, setup_test_data):

    folder, ometiff_data, zarr_data, bf_data = setup_test_data
    save_folder = setup_data_save_folder

    input = ometiff_data
    output = os.path.join(save_folder,'2T_3P_81Z_231Y_498X_Kazansky.zarr')

    if os.path.exists(output):
        shutil.rmtree(output)

    converter = ZarrConverter(input, output, 'ometiff')
    tf = tiff.TiffFile(os.path.join(ometiff_data, '2T_3P_81Z_231Y_498X_Kazansky_2_MMStack.ome.tif'))

    assert(converter.files == glob.glob(os.path.join(ometiff_data, '*.ome.tif')))
    assert(converter.dtype == 'uint16')
    assert(isinstance(converter.reader, WaveorderReader))
    assert(isinstance(converter.writer, WaveorderWriter))
    assert(converter.summary_metadata == tf.micromanager_metadata['Summary'])

    converter._gen_coordset()
    coords = []
    for t in range(2):
        for p in range(3):
            for c in range(4):
                for z in range(81):
                    coords.append((t, p, c, z))

    assert(converter.coords == coords)
    assert(converter.p_dim == 1)
    assert(converter.t_dim == 0)
    assert(converter.c_dim == 2)
    assert(converter.z_dim == 3)
    assert(converter.p == 3)
    assert(converter.t == 2)
    assert(converter.c == 4)
    assert(converter.z == 81)

def test_converter_run(setup_data_save_folder, setup_test_data):

    folder, ometiff_data, zarr_data, bf_data = setup_test_data
    save_folder = setup_data_save_folder

    input = ometiff_data
    output = os.path.join(save_folder, '2T_3P_81Z_231Y_498X_Kazansky.zarr')

    if os.path.exists(output):
        shutil.rmtree(output)

    converter = ZarrConverter(input, output, 'ometiff')
    tf = tiff.TiffFile(os.path.join(ometiff_data, '2T_3P_81Z_231Y_498X_Kazansky_2_MMStack.ome.tif'))

    converter.run_conversion()

    zs = zarr.open(output, 'r')

    assert(os.path.exists(os.path.join(save_folder, '2T_3P_81Z_231Y_498X_Kazansky_ImagePlaneMetadata.txt')))

    coords = []
    for t in range(2):
        for p in range(3):
            for c in range(4):
                for z in range(81):
                    coords.append((t, p, c, z))

    assert(converter.coords == coords)

    cnt = 0
    for t in range(2):
        for p in range(3):
            for c in range(4):
                for z in range(81):
                    image = zs['Row_0'][f'Col_{p}'][f'Pos_00{p}']['arr_0'][t, c, z]
                    tiff_image = tf.pages.get(cnt).asarray()
                    assert(np.array_equal(image, tiff_image))
                    cnt += 1


# def test_converter_upti():
#
#     input = '/Users/cameron.foltz/Desktop/Test_Data/upti/CM_FOV1/data'
#     output = '/Users/cameron.foltz/Desktop/Test_Data/converter_test/CM_FOV1_upti.zarr'
#     if os.path.exists(output):
#         shutil.rmtree(output)
#     data_type = 'upti'
#     converter = ZarrConverter(input, output, data_type)
#     converter.run_conversion()