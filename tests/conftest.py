import pytest
import os
import shutil

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


@pytest.fixture(scope="session")
def setup_mm2gamma_ome_tiffs():
    temp_folder = os.getcwd() + '/pytest_temp'
    temp_gamma = os.path.join(temp_folder, 'mm2gamma')
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")

    # 'https://drive.google.com/file/d/1WSu1CaFQKIipMproxPInYs6ljh17JM7K/view?usp=sharing'

    # DO NOT ADJUST THIS VALUE
    mm2gamma_ometiffs = '1WSu1CaFQKIipMproxPInYs6ljh17JM7K'

    # download files to temp folder
    output = temp_gamma + "/mm2gamma_ometiffs.zip"
    gdd.download_file_from_google_drive(file_id=mm2gamma_ometiffs,
                                        dest_path=output,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    # return path to unzipped folder containing test images
    yield os.path.join(temp_gamma, 'ome-tiffs')

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
def setup_mm2gamma_singlepage_tiffs():
    temp_folder = os.getcwd() + '/pytest_temp'
    temp_gamma = os.path.join(temp_folder, 'mm2gamma')
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")

    # 'https://drive.google.com/file/d/1qQys8r0_HIsVqLrMfUzsndTgTtyPhZes/view?usp=sharing'

    # DO NOT ADJUST THIS VALUE
    mm2gamma_singlepage_tiffs = '1qQys8r0_HIsVqLrMfUzsndTgTtyPhZes'

    # download files to temp folder
    output = temp_gamma + "/mm2gamma_singlepage_tiffs.zip"
    gdd.download_file_from_google_drive(file_id=mm2gamma_singlepage_tiffs,
                                        dest_path=output,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    # return path to folder with images
    yield os.path.join(temp_gamma, 'singlepage-tiffs')

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
