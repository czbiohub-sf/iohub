import pytest
import shutil
import os
import random
from wget import download

MM2GAMMA_OMETIFF_SUBFOLDERS = \
    {
        'mm2.0-20201209_4p_2t_5z_1c_512k_1',
        'mm2.0-20201209_4p_2t_5z_512k_1',
        'mm2.0-20201209_4p_2t_3c_512k_1',
        'mm2.0-20201209_4p_5z_3c_512k_1',
        'mm2.0-20201209_4p_2t_512k_1',
        'mm2.0-20201209_4p_3c_512k_1',
        'mm2.0-20201209_4p_5z_512k_1',
        'mm2.0-20201209_1t_5z_3c_512k_1',
        'mm2.0-20201209_1t_5z_512k_1',
        'mm2.0-20201209_2t_3c_512k_1',
        'mm2.0-20201209_5z_3c_512k_1',
        'mm2.0-20201209_4p_512k_1',
        'mm2.0-20201209_2t_512k_1',
        'mm2.0-20201209_3c_512k_1',
        'mm2.0-20201209_5z_512k_1',
        'mm2.0-20201209_100t_5z_3c_512k_1_nopositions',
        'mm2.0-20201209_4p_20t_5z_3c_512k_1_multipositions',
    }

join = os.path.join


@pytest.fixture(scope='function')
def setup_folder():
    temp_folder = os.getcwd() + '/pytest_temp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")

    yield temp_folder

    try:
        # remove temp folder
        shutil.rmtree(temp_folder)
    except OSError as e:
        print(f"Error while deleting temp folder: {e.strerror}")


@pytest.fixture(scope="session")
def setup_test_data():

    temp_folder = os.getcwd() + '/pytest_temp'
    temp_data = os.path.join(temp_folder, 'rawdata')
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")
    if not os.path.isdir(temp_data):
        os.mkdir(temp_data)

    # Zenodo URL
    url = 'https://zenodo.org/record/6249285/files/waveOrder_testData.zip?download=1'

    # download files to temp folder
    output = temp_data + "/waveOrder_testData.zip"
    download(url, out=output)
    shutil.unpack_archive(output, extract_dir=temp_data)

    yield join(temp_data, 'waveOrder')

    # breakdown files
    try:
        # remove zip file
        os.remove(output)

        # remove unzipped folder
        shutil.rmtree(temp_data)

        # remove temp folder
        shutil.rmtree(temp_folder)
    except OSError as e:
        print(f"Error while deleting temp folder: {e.strerror}")

@pytest.fixture(scope="function")
def setup_mm2gamma_ome_tiffs():

    temp_folder = os.getcwd() + '/pytest_temp/rawdata/waveOrder'
    temp_data = os.path.join(temp_folder, 'MM20_ome-tiffs')

    subfolders = [f for f in os.listdir(temp_data) if os.path.isdir(join(temp_data, f))]

    # specific folder
    one_folder = join(temp_data, subfolders[0])
    # random folder
    rand_folder = join(temp_data, random.choice(subfolders))
    # return path to unzipped folder containing test images as well as specific folder paths
    yield temp_data, one_folder, rand_folder

@pytest.fixture(scope="function")
def setup_mm2gamma_ome_tiffs_incomplete():
    """
    This fixture returns a dataset with 11 timepoints
    """
    temp_folder = os.getcwd() + '/pytest_temp/rawdata/waveOrder'
    temp_data = os.path.join(temp_folder, 'MM20_ometiff_incomplete')

    src = os.path.join(temp_data, 'mm2.0-20201209_20t_5z_3c_512k_incomplete_1')

    yield src

@pytest.fixture(scope="function")
def setup_mm2gamma_singlepage_tiffs():
    temp_folder = os.getcwd() + '/pytest_temp/rawdata/waveOrder'
    temp_data = os.path.join(temp_folder, 'MM20_singlepage-tiffs')

    subfolders = [f for f in os.listdir(temp_data) if os.path.isdir(join(temp_data, f))]

    # specific folder
    one_folder = join(temp_data, subfolders[0])
    # random folder
    rand_folder = join(temp_data, random.choice(subfolders))
    # return path to unzipped folder containing test images as well as specific folder paths
    yield temp_data, one_folder, rand_folder

@pytest.fixture(scope="function")
def setup_mm2gamma_singlepage_tiffs_incomplete():
    """
    This fixture returns a dataset with 11 timepoints
    The MDA definition at start of the experiment specifies 20 timepoints
    """
    temp_folder = os.getcwd() + '/pytest_temp/rawdata/waveOrder'
    temp_data = os.path.join(temp_folder, 'MM20_singlepage_incomplete')

    src = os.path.join(temp_data, 'mm2.0-20201209_20t_5z_3c_512k_incomplete_1 2')

    yield src

@pytest.fixture(scope="function")
def setup_mm1422_ome_tiffs():
    temp_folder = os.getcwd() + '/pytest_temp/rawdata/waveOrder'
    temp_data = os.path.join(temp_folder, 'MM1422_ome-tiffs')

    subfolders = [f for f in os.listdir(temp_data) if os.path.isdir(join(temp_data, f))]

    # specific folder
    one_folder = join(temp_data, subfolders[0])
    # random folder
    rand_folder = join(temp_data, random.choice(subfolders))
    # return path to unzipped folder containing test images as well as specific folder paths
    yield temp_data, one_folder, rand_folder

@pytest.fixture(scope="function")
def setup_mm1422_singlepage_tiffs():

    temp_folder = os.getcwd() + '/pytest_temp/rawdata/waveOrder'
    temp_data = os.path.join(temp_folder, 'MM1422_singlepage-tiffs')

    subfolders = [f for f in os.listdir(temp_data) if os.path.isdir(join(temp_data, f))]

    # specific folder
    one_folder = join(temp_data, subfolders[0])
    # random folder
    rand_folder = join(temp_data, random.choice(subfolders))
    # return path to unzipped folder containing test images as well as specific folder paths
    yield temp_data, one_folder, rand_folder

@pytest.fixture(scope="function")
def setup_mm2gamma_zarr():

    temp_folder = os.getcwd() + '/pytest_temp/rawdata/waveOrder'
    temp_data = os.path.join(temp_folder, 'MM20_zarr')

    zp = os.path.join(temp_data, 'mm2.0-20201209_4p_2t_5z_1c_512k_1.zarr')

    # return path to unzipped folder containing test images as well as specific folder paths
    yield zp
