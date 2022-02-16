import pytest
import shutil
import os
import random
# import gdown
from google_drive_downloader import GoogleDriveDownloader as gdd

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
def setup_mm2gamma_ome_tiffs():
    temp_folder = os.getcwd() + '/pytest_temp'
    temp_gamma = os.path.join(temp_folder, 'mm2gamma')
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")

    # shared gdrive
    # 'https://drive.google.com/file/d/1UWSr4GQ6Kpj5irq2TicvDLULfWjKhh0b/view?usp=sharing'

    # DO NOT ADJUST THIS VALUE
    mm2gamma_ometiffs = '1UWSr4GQ6Kpj5irq2TicvDLULfWjKhh0b'

    # download files to temp folder
    output = temp_gamma + "/mm2gamma_ometiffs.zip"
    gdd.download_file_from_google_drive(file_id=mm2gamma_ometiffs,
                                        dest_path=output,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    src = os.path.join(temp_gamma, 'ome-tiffs')
    subfolders = [f for f in os.listdir(src) if os.path.isdir(join(src, f))]

    # specific folder
    one_folder = join(src, subfolders[0])
    # random folder
    rand_folder = join(src, random.choice(subfolders))
    # return path to unzipped folder containing test images as well as specific folder paths
    yield src, one_folder, rand_folder

    # breakdown files
    try:
        # remove zip file
        os.remove(output)

        # remove unzipped folder
        shutil.rmtree(os.path.join(temp_gamma, 'ome-tiffs'))

        # remove temp folder
        shutil.rmtree(temp_folder)
    except OSError as e:
        print(f"Error while deleting temp folder: {e.strerror}")


@pytest.fixture(scope="session")
def setup_mm2gamma_ome_tiffs_incomplete():
    """
    This fixture returns a dataset with 11 timepoints
    """
    temp_folder = os.getcwd() + '/pytest_temp'
    temp_gamma = os.path.join(temp_folder, 'mm2gamma')
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")

    # shared gdrive
    # 'https://drive.google.com/file/d/1OeKmJcZ_Pb8q36q9u_FIRd2AOvdn2_MP/view?usp=sharing'

    # DO NOT ADJUST THIS VALUE
    mm2gamma_singlepage_tiffs = '1OeKmJcZ_Pb8q36q9u_FIRd2AOvdn2_MP'

    # download files to temp folder
    output = temp_gamma + "/mm2.0-20201209_20t_5z_3c_512k_incomplete_1.zip"
    gdd.download_file_from_google_drive(file_id=mm2gamma_singlepage_tiffs,
                                        dest_path=output,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    src = os.path.join(temp_gamma, 'mm2.0-20201209_20t_5z_3c_512k_incomplete_1')

    yield src

    # breakdown files
    try:
        # remove zip file
        os.remove(output)

        # remove unzipped folder
        shutil.rmtree(os.path.join(temp_gamma, 'mm2.0-20201209_20t_5z_3c_512k_incomplete_1'))

        # remove temp folder
        shutil.rmtree(temp_folder)
    except OSError as e:
        print(f"Error while deleting temp folder: {e.strerror}")


@pytest.fixture(scope="session")
def setup_mm2gamma_singlepage_tiffs():
    temp_folder = os.getcwd() + '/pytest_temp'
    temp_gamma = os.path.join(temp_folder, 'mm2gamma')
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")

    # shared gdrive
    # 'https://drive.google.com/file/d/1UpslH393sJ2GodaM6XgqKBzWwQ39-Dc-/view?usp=sharing'

    # DO NOT ADJUST THIS VALUE
    mm2gamma_singlepage_tiffs = '1UpslH393sJ2GodaM6XgqKBzWwQ39-Dc-'

    # download files to temp folder
    output = temp_gamma + "/mm2gamma_singlepage_tiffs.zip"
    gdd.download_file_from_google_drive(file_id=mm2gamma_singlepage_tiffs,
                                        dest_path=output,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    src = os.path.join(temp_gamma, 'singlepage-tiffs')
    subfolders = [f for f in os.listdir(src) if os.path.isdir(join(src, f))]

    # specific folder
    one_folder = join(src, subfolders[0])
    # random folder
    rand_folder = join(src, random.choice(subfolders))
    # return path to unzipped folder containing test images as well as specific folder paths
    yield src, one_folder, rand_folder

    # breakdown files
    try:
        # remove zip file
        os.remove(output)

        # remove unzipped folder
        shutil.rmtree(os.path.join(temp_gamma, 'singlepage-tiffs'))

        # remove temp folder
        shutil.rmtree(temp_folder)
    except OSError as e:
        print(f"Error while deleting temp folder: {e.strerror}")


@pytest.fixture(scope="session")
def setup_mm2gamma_singlepage_tiffs_incomplete():
    """
    This fixture returns a dataset with 11 timepoints
    The MDA definition at start of the experiment specifies 20 timepoints
    """
    temp_folder = os.getcwd() + '/pytest_temp'
    temp_gamma = os.path.join(temp_folder, 'mm2gamma')
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")

    # shared gdrive
    # 'https://drive.google.com/file/d/1CXNpohEUb4IUjAATGw5Wr_2Q8JzaXcaj/view?usp=sharing'

    # DO NOT ADJUST THIS VALUE
    mm2gamma_singlepage_tiffs = '1CXNpohEUb4IUjAATGw5Wr_2Q8JzaXcaj'

    # download files to temp folder
    output = temp_gamma + "/mm2.0-20201209_20t_5z_3c_512k_incomplete_1.zip"
    gdd.download_file_from_google_drive(file_id=mm2gamma_singlepage_tiffs,
                                        dest_path=output,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    src = os.path.join(temp_gamma, 'mm2.0-20201209_20t_5z_3c_512k_incomplete_1')

    yield src

    # breakdown files
    try:
        # remove zip file
        os.remove(output)

        # remove unzipped folder
        shutil.rmtree(os.path.join(temp_gamma, 'mm2.0-20201209_20t_5z_3c_512k_incomplete_1'))

        # remove temp folder
        shutil.rmtree(temp_folder)
    except OSError as e:
        print(f"Error while deleting temp folder: {e.strerror}")


@pytest.fixture(scope="session")
def setup_mm1422_ome_tiffs():
    temp_folder = os.getcwd() + '/pytest_temp'
    temp_1422 = os.path.join(temp_folder, 'mm1422')
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")

    # shared gdrive
    # 'https://drive.google.com/file/d/1N94AT2YGJDxgjTVIgYA048SQkKvbPqyo/view?usp=sharing'

    # DO NOT ADJUST THIS VALUE
    mm1422_ometiffs = '1N94AT2YGJDxgjTVIgYA048SQkKvbPqyo'

    # download files to temp folder
    output = temp_1422 + "/mm1422_ometiffs.zip"
    gdd.download_file_from_google_drive(file_id=mm1422_ometiffs,
                                        dest_path=output,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    # return path to unzipped folder containing test images
    src = os.path.join(temp_1422, 'ome-tiffs')
    subfolders = [f for f in os.listdir(src) if os.path.isdir(join(src, f))]

    # specific folder
    one_folder = join(src, subfolders[0])
    # random folder
    rand_folder = join(src, random.choice(subfolders))
    # return path to unzipped folder containing test images as well as specific folder paths
    yield src, one_folder, rand_folder

    # breakdown files
    try:
        # remove zip file
        os.remove(output)

        # remove unzipped folder
        shutil.rmtree(os.path.join(temp_1422, 'ome-tiffs'))

        # remove temp folder
        shutil.rmtree(temp_folder)
    except OSError as e:
        print(f"Error while deleting temp folder: {e.strerror}")


@pytest.fixture(scope="session")
def setup_mm1422_singlepage_tiffs():
    temp_folder = os.getcwd() + '/pytest_temp'
    temp_1422 = os.path.join(temp_folder, 'mm1422')
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")

    # shared gdrive
    # 'https://drive.google.com/file/d/1HWx8cja3cBeGPJJoyK3SMYRE6wcjSsnb/view?usp=sharing'

    # DO NOT ADJUST THIS VALUE
    mm1422_singlepage_tiffs = '1HWx8cja3cBeGPJJoyK3SMYRE6wcjSsnb'

    # download files to temp folder
    output = temp_1422 + "/mm1422_singlepage_tiffs.zip"
    gdd.download_file_from_google_drive(file_id=mm1422_singlepage_tiffs,
                                        dest_path=output,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    src = os.path.join(temp_1422, 'singlepage-tiffs')
    subfolders = [f for f in os.listdir(src) if os.path.isdir(join(src, f))]

    # specific folder
    one_folder = join(src, subfolders[0])
    # random folder
    rand_folder = join(src, random.choice(subfolders))
    # return path to unzipped folder containing test images as well as specific folder paths
    yield src, one_folder, rand_folder

    # breakdown files
    try:
        # remove zip file
        os.remove(output)

        # remove unzipped folder
        shutil.rmtree(os.path.join(temp_1422, 'singlepage-tiffs'))

        # remove temp folder
        shutil.rmtree(temp_folder)
    except OSError as e:
        print(f"Error while deleting temp folder: {e.strerror}")

@pytest.fixture(scope="session")
def setup_mm2gamma_zarr():
    temp_folder = os.getcwd() + '/pytest_temp'
    temp_2gamma = os.path.join(temp_folder, 'mm2gamma')
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")

    # shared gdrive
    # https://drive.google.com/file/d/1mCMjKxoQ3D09j0xBJFP0Rg_p2TBY6pAq/view?usp=sharing

    # DO NOT ADJUST THIS VALUE
    mm2gamma_zarr = '1mCMjKxoQ3D09j0xBJFP0Rg_p2TBY6pAq'

    # download files to temp folder
    output = temp_2gamma + "/2021_06_11_recOrder_pytest_20x_04NA_zarr_test.zip"
    gdd.download_file_from_google_drive(file_id=mm2gamma_zarr,
                                        dest_path=output,
                                        unzip=False,
                                        showsize=True,
                                        overwrite=True)

    shutil.unpack_archive(output, temp_2gamma)

    print(os.listdir(temp_2gamma))

    src = os.path.join(temp_2gamma, '2021_06_11_recOrder_pytest_20x_04NA_zarr')
    zp = os.path.join(src, '2T_3P_81Z_231Y_498X_Kazansky.zarr')

    # return path to unzipped folder containing test images as well as specific folder paths
    yield zp

    # breakdown files
    try:
        # remove zip file
        os.remove(output)

        # remove unzipped folder
        shutil.rmtree(src)

        # remove temp folder
        shutil.rmtree(temp_folder)
    except OSError as e:
        print(f"Error while deleting temp folder: {e.strerror}")
