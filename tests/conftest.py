import pytest
import os
import shutil

from google_drive_downloader import GoogleDriveDownloader as gdd


@pytest.fixture(scope="session")
def setup_mm2gamma_ome_tiffs():
    temp_folder = os.getcwd() + '/pytest_temp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")

    # 'https://drive.google.com/file/d/1WSu1CaFQKIipMproxPInYs6ljh17JM7K/view?usp=sharing'

    # DO NOT ADJUST THIS VALUE
    mm2gamma_ometiffs = '1WSu1CaFQKIipMproxPInYs6ljh17JM7K'

    # download files to temp folder
    output = temp_folder + "/mm2gamma_ometiffs.zip"
    gdd.download_file_from_google_drive(file_id=mm2gamma_ometiffs,
                                        dest_path=output,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    # return path to unzipped folder containing test images
    yield output.strip('.zip')

    # breakdown files
    try:
        # remove zip file
        os.remove(output)

        # remove unzipped folder
        shutil.rmtree(output.strip('.zip'))

        # remove temp folder
        shutil.rmtree(temp_folder)
    except OSError as e:
        print(f"Error while deleting temp folder: {e.strerror}")


@pytest.fixture(scope="session")
def setup_mm2gamma_singlepage_tiffs():
    temp_folder = os.getcwd() + '/pytest_temp'
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")

    # 'https://drive.google.com/file/d/1qQys8r0_HIsVqLrMfUzsndTgTtyPhZes/view?usp=sharing'

    # DO NOT ADJUST THIS VALUE
    mm2gamma_singlepage_tiffs = '1qQys8r0_HIsVqLrMfUzsndTgTtyPhZes'

    # download files to temp folder
    output = temp_folder + "/mm2gamma_singlepage_tiffs.zip"
    gdd.download_file_from_google_drive(file_id=mm2gamma_singlepage_tiffs,
                                        dest_path=output,
                                        unzip=True,
                                        showsize=True,
                                        overwrite=True)

    # return path to folder with images
    yield output.strip('.zip')

    # breakdown files
    try:
        # remove zip file
        os.remove(output)

        # remove unzipped folder
        shutil.rmtree(output.strip('.zip'))

        # remove temp folder
        shutil.rmtree(temp_folder)
    except OSError as e:
        print(f"Error while deleting temp folder: {e.strerror}")
