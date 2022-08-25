import os
import shutil
import zarr
from tifffile import TiffFile
from waveorder.io import WaveorderReader, WaveorderWriter
import numpy as np
from recOrder.io.zarr_converter import ZarrConverter

def test_ometiff_converter_initialize(setup_data_save_folder, get_ometiff_data_dir):

    folder, ometiff_data = get_ometiff_data_dir
    save_folder = setup_data_save_folder

    input = ometiff_data
    output = os.path.join(save_folder, '2T_3P_16Z_128Y_256X_Kazansky.zarr')

    if os.path.exists(output):
        shutil.rmtree(output)

    converter = ZarrConverter(input, output)

    with TiffFile(os.path.join(ometiff_data, '2T_3P_16Z_128Y_256X_Kazansky_1_MMStack_Pos0.ome.tif')) as tf:
        assert (converter.summary_metadata == tf.micromanager_metadata['Summary'])

    assert(converter.dtype == 'uint16')
    assert(isinstance(converter.reader, WaveorderReader))
    assert(isinstance(converter.writer, WaveorderWriter))

    assert(converter.dim_order == ['time', 'position', 'z', 'channel'])
    assert(converter.p_dim == 1)
    assert(converter.t_dim == 0)
    assert(converter.c_dim == 3)
    assert(converter.z_dim == 2)
    assert(converter.p == 3)
    assert(converter.t == 2)
    assert(converter.c == 4)
    assert(converter.z == 16)

def test_ometiff_converter_run(setup_data_save_folder, get_ometiff_data_dir):

    folder, ometiff_data = get_ometiff_data_dir
    save_folder = setup_data_save_folder

    input = ometiff_data
    output = os.path.join(save_folder, '2T_3P_16Z_128Y_256X_Kazansky.zarr')

    if os.path.exists(output):
        shutil.rmtree(output)

    converter = ZarrConverter(input, output)

    converter.run_conversion()
    zs = zarr.open(output, 'r')

    assert(os.path.exists(os.path.join(save_folder, '2T_3P_16Z_128Y_256X_Kazansky_ImagePlaneMetadata.txt')))

    cnt = 0
    for t in range(2):
        for p in range(3):
            cnt = t*16*4
            tf = TiffFile(os.path.join(ometiff_data, f'2T_3P_16Z_128Y_256X_Kazansky_1_MMStack_Pos{p}.ome.tif'))
            for z in range(16):
                for c in range(4):
                    image = zs['Row_0'][f'Col_{p}'][f'Pos_00{p}']['arr_0'][t, c, z]
                    tiff_image = tf.pages.get(cnt).asarray()
                    assert(np.array_equal(image, tiff_image))
                    cnt += 1
            tf.close()

def test_pycromanager_converter_initialize(setup_data_save_folder, get_pycromanager_data_dir):

    folder, pm_data = get_pycromanager_data_dir
    save_folder = setup_data_save_folder

    input = pm_data
    output = os.path.join(save_folder, 'mm2.0-20210713_pm0.13.2_2p_3t_2c_7z_1.zarr')

    if os.path.exists(output):
        shutil.rmtree(output)

    converter = ZarrConverter(input, output)

    assert(isinstance(converter.reader, WaveorderReader))
    assert(isinstance(converter.writer, WaveorderWriter))
    assert converter.summary_metadata is not None

    assert(converter.dim_order == ['position', 'time', 'channel', 'z'])
    assert(converter.p_dim == 0)
    assert(converter.t_dim == 1)
    assert(converter.c_dim == 2)
    assert(converter.z_dim == 3)
    assert(converter.p == 2)
    assert(converter.t == 3)
    assert(converter.c == 2)
    assert(converter.z == 7)

def test_pycromanager_converter_run(setup_data_save_folder, get_pycromanager_data_dir):

    folder, pm_data = get_pycromanager_data_dir
    save_folder = setup_data_save_folder

    input = pm_data
    output = os.path.join(save_folder, 'mm2.0-20210713_pm0.13.2_2p_3t_2c_7z_1.zarr')

    if os.path.exists(output):
        shutil.rmtree(output)

    converter = ZarrConverter(input, output)
    wo_dataset = WaveorderReader(input)

    converter.run_conversion()
    zs = zarr.open(output, 'r')

    assert(os.path.exists(os.path.join(save_folder, 'mm2.0-20210713_pm0.13.2_2p_3t_2c_7z_1_ImagePlaneMetadata.txt')))

    cnt = 0
    for t in range(3):
        for p in range(2):
            for c in range(2):
                for z in range(7):
                    image = zs['Row_0'][f'Col_{p}'][f'Pos_00{p}']['arr_0'][t, c, z]
                    wo_image = wo_dataset.get_image(p, t, c, z)
                    assert(np.array_equal(image, wo_image))
                    cnt += 1
