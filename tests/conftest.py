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


@pytest.fixture(scope="session")
def setup_test_data():

    temp_folder = join(os.getcwd(), 'pytest_temp')
    test_data = os.path.join(temp_folder, 'test_data')
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")
    if not os.path.isdir(test_data):
        os.mkdir(test_data)

    # Zenodo URL
    url = 'https://zenodo.org/record/6983916/files/waveOrder_test_data.zip?download=1'

    # download files to temp folder
    output = join(test_data, "waveOrder_test_data.zip")
    if not os.listdir(test_data):
        print("Downloading test files...")
        download(url, out=output)
        shutil.unpack_archive(output, extract_dir=test_data)

    yield test_data

    # TODO: single page tiff don't clean up properly
    try:
        # remove temp folder
        shutil.rmtree(temp_folder)
    except OSError as e:
        print(f"Error while deleting temp folder: {e.strerror}")

@pytest.fixture(scope='function')
def setup_writer_folder():
    temp_folder = join(os.getcwd(), 'pytest_temp', 'writer_temp')
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")

    yield temp_folder

    try:
        # remove temp folder
        shutil.rmtree(temp_folder)
    except OSError as e:
        print(f"Error while deleting temp folder: {e.strerror}")

@pytest.fixture(scope="function")
def setup_mm2gamma_ome_tiffs():

    test_data = os.path.join(os.getcwd(), 'pytest_temp', 'test_data', 'MM20_ome-tiffs')

    subfolders = [f for f in os.listdir(test_data) if os.path.isdir(join(test_data, f))]

    # specific folder
    one_folder = join(test_data, subfolders[0])
    # random folder
    rand_folder = join(test_data, random.choice(subfolders))
    # return path to unzipped folder containing test images as well as specific folder paths
    yield test_data, one_folder, rand_folder

@pytest.fixture(scope="function")
def setup_mm2gamma_ome_tiffs_incomplete():
    """
    This fixture returns a dataset with 11 timepoints
    """

    test_data = os.path.join(os.getcwd(), 'pytest_temp', 'test_data', 'MM20_ometiff_incomplete')

    src = os.path.join(test_data, 'mm2.0-20201209_20t_5z_3c_512k_incomplete_1')

    yield src

@pytest.fixture(scope="function")
def setup_mm2gamma_singlepage_tiffs():

    test_data = os.path.join(os.getcwd(), 'pytest_temp', 'test_data', 'MM20_singlepage-tiffs')

    subfolders = [f for f in os.listdir(test_data) if os.path.isdir(join(test_data, f))]

    # specific folder
    one_folder = join(test_data, subfolders[0])
    # random folder
    rand_folder = join(test_data, random.choice(subfolders))
    # return path to unzipped folder containing test images as well as specific folder paths
    yield test_data, one_folder, rand_folder

@pytest.fixture(scope="function")
def setup_mm2gamma_singlepage_tiffs_incomplete():
    """
    This fixture returns a dataset with 11 timepoints
    The MDA definition at start of the experiment specifies 20 timepoints
    """

    test_data = os.path.join(os.getcwd(), 'pytest_temp', 'test_data', 'MM20_singlepage_incomplete')

    src = os.path.join(test_data, 'mm2.0-20201209_20t_5z_3c_512k_incomplete_1 2')

    yield src

@pytest.fixture(scope="function")
def setup_mm1422_ome_tiffs():

    test_data = os.path.join(os.getcwd(), 'pytest_temp', 'test_data', 'MM1422_ome-tiffs')

    subfolders = [f for f in os.listdir(test_data) if os.path.isdir(join(test_data, f))]

    # specific folder
    one_folder = join(test_data, subfolders[0])
    # random folder
    rand_folder = join(test_data, random.choice(subfolders))
    # return path to unzipped folder containing test images as well as specific folder paths
    yield test_data, one_folder, rand_folder

@pytest.fixture(scope="function")
def setup_mm1422_singlepage_tiffs():

    test_data = os.path.join(os.getcwd(), 'pytest_temp', 'test_data', 'MM1422_singlepage-tiffs')

    subfolders = [f for f in os.listdir(test_data) if os.path.isdir(join(test_data, f))]

    # specific folder
    one_folder = join(test_data, subfolders[0])
    # random folder
    rand_folder = join(test_data, random.choice(subfolders))
    # return path to unzipped folder containing test images as well as specific folder paths
    yield test_data, one_folder, rand_folder

@pytest.fixture(scope="function")
def setup_mm2gamma_zarr():

    test_data = os.path.join(os.getcwd(), 'pytest_temp', 'test_data', 'MM20_zarr')

    zp = os.path.join(test_data, 'mm2.0-20201209_4p_2t_5z_1c_512k_1.zarr')

    # return path to unzipped folder containing test images as well as specific folder paths
    yield zp

@pytest.fixture(scope="function")
def setup_pycromanager_test_data():
    test_data = os.path.join(os.getcwd(), 'pytest_temp', 'test_data', 'MM20_pycromanager')
    datasets = ['mm2.0-20210713_pm0.13.2_2c_1',
                'mm2.0-20210713_pm0.13.2_2c_7z_1',
                'mm2.0-20210713_pm0.13.2_2p_3t_2c_1',
                'mm2.0-20210713_pm0.13.2_2p_3t_2c_7z_1',
                'mm2.0-20210713_pm0.13.2_3t_2c_1',
                'mm2.0-20210713_pm0.13.2_3t_2c_7z_1',
                'mm2.0-20210713_pm0.13.2_3t_7z_1',
                'mm2.0-20210713_pm0.13.2_5t_1',
                'mm2.0-20210713_pm0.13.2_7z_1']

    dataset_dirs = [os.path.join(test_data, ds) for ds in datasets]
    first_dir, rand_dir, ptcz_dir = (dataset_dirs[0], random.choice(dataset_dirs), dataset_dirs[3])

    yield first_dir, rand_dir, ptcz_dir
