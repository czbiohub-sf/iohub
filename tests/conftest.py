import os
import random
import shutil
from os.path import join as pjoin

import fsspec
import pytest
from wget import download

MM2GAMMA_OMETIFF_SUBFOLDERS = {
    "mm2.0-20201209_4p_2t_5z_1c_512k_1",
    "mm2.0-20201209_4p_2t_5z_512k_1",
    "mm2.0-20201209_4p_2t_3c_512k_1",
    "mm2.0-20201209_4p_5z_3c_512k_1",
    "mm2.0-20201209_4p_2t_512k_1",
    "mm2.0-20201209_4p_3c_512k_1",
    "mm2.0-20201209_4p_5z_512k_1",
    "mm2.0-20201209_1t_5z_3c_512k_1",
    "mm2.0-20201209_1t_5z_512k_1",
    "mm2.0-20201209_2t_3c_512k_1",
    "mm2.0-20201209_5z_3c_512k_1",
    "mm2.0-20201209_4p_512k_1",
    "mm2.0-20201209_2t_512k_1",
    "mm2.0-20201209_3c_512k_1",
    "mm2.0-20201209_5z_512k_1",
    "mm2.0-20201209_100t_5z_3c_512k_1_nopositions",
    "mm2.0-20201209_4p_20t_5z_3c_512k_1_multipositions",
}


@pytest.fixture(scope="session")
def setup_test_data():
    temp_folder = pjoin(os.getcwd(), ".pytest_temp")
    test_data = pjoin(temp_folder, "test_data")
    if not os.path.isdir(temp_folder):
        os.mkdir(temp_folder)
        print("\nsetting up temp folder")
    if not os.path.isdir(test_data):
        os.mkdir(test_data)

    # Zenodo URL
    custom_url = (
        "https://zenodo.org/record/6983916/files/waveOrder_test_data.zip"
    )
    # Reference v0.4 HCS dataset from OME
    # See the last line of
    # https://github.com/ome/ngff/issues/140#issuecomment-1309972511
    ome_hcs_url = "https://zenodo.org/record/8091756/files/20200812-CardiomyocyteDifferentiation14-Cycle1.zarr.zip"  # noqa

    # download files to temp folder
    if not os.listdir(test_data):
        print("Downloading test files...")
        for url in (custom_url, ome_hcs_url):
            output = pjoin(test_data, os.path.basename(url))
            download(url, out=output)
            shutil.unpack_archive(output, extract_dir=test_data)
        ghfs = fsspec.filesystem(
            "github", org="micro-manager", repo="NDTiffStorage"
        )
        v3_lp = pjoin(test_data, "ndtiff_v3_labeled_positions")
        os.mkdir(v3_lp)
        ghfs.get(
            ghfs.ls("test_data/v3/labeled_positions_1"),
            v3_lp,
            recursive=True,
        )
    yield test_data


@pytest.fixture(scope="function")
def setup_mm2gamma_ome_tiffs():
    test_data = pjoin(
        os.getcwd(), ".pytest_temp", "test_data", "MM20_ome-tiffs"
    )

    subfolders = [
        f for f in os.listdir(test_data) if os.path.isdir(pjoin(test_data, f))
    ]

    # specific folder
    one_folder = pjoin(test_data, subfolders[0])
    # random folder
    rand_folder = pjoin(test_data, random.choice(subfolders))
    # return path to unzipped folder containing test images
    # as well as specific folder paths
    yield test_data, one_folder, rand_folder


@pytest.fixture(scope="function")
def setup_mm2gamma_ome_tiff_hcs():
    test_data = pjoin(
        os.getcwd(), ".pytest_temp", "test_data", "MM20_ome-tiffs"
    )

    subfolders = [
        f for f in os.listdir(test_data) if os.path.isdir(pjoin(test_data, f))
    ]
    # select datasets with multiple positioons; here they all have 4 positions
    hcs_subfolders = [f for f in subfolders if '4p' in f]

    # specific folder
    one_folder = pjoin(test_data, hcs_subfolders[0])
    # random folder
    rand_folder = pjoin(test_data, random.choice(hcs_subfolders))
    # return path to unzipped folder containing test images
    # as well as specific folder paths
    yield test_data, one_folder, rand_folder


@pytest.fixture(scope="function")
def setup_mm2gamma_ome_tiffs_incomplete():
    """
    This fixture returns a dataset with 11 timepoints
    """

    test_data = pjoin(
        os.getcwd(), ".pytest_temp", "test_data", "MM20_ometiff_incomplete"
    )

    src = pjoin(test_data, "mm2.0-20201209_20t_5z_3c_512k_incomplete_1")

    yield src


@pytest.fixture(scope="function")
def setup_mm2gamma_singlepage_tiffs():
    test_data = pjoin(
        os.getcwd(), ".pytest_temp", "test_data", "MM20_singlepage-tiffs"
    )

    subfolders = [
        f for f in os.listdir(test_data) if os.path.isdir(pjoin(test_data, f))
    ]

    # specific folder
    one_folder = pjoin(test_data, subfolders[0])
    # random folder
    rand_folder = pjoin(test_data, random.choice(subfolders))
    # return path to unzipped folder containing test images
    # as well as specific folder paths
    yield test_data, one_folder, rand_folder


@pytest.fixture(scope="function")
def setup_mm2gamma_singlepage_tiffs_incomplete():
    """
    This fixture returns a dataset with 11 timepoints
    The MDA definition at start of the experiment specifies 20 timepoints
    """

    test_data = pjoin(
        os.getcwd(), ".pytest_temp", "test_data", "MM20_singlepage_incomplete"
    )

    src = pjoin(test_data, "mm2.0-20201209_20t_5z_3c_512k_incomplete_1 2")

    yield src


@pytest.fixture(scope="function")
def setup_mm1422_ome_tiffs():
    test_data = pjoin(
        os.getcwd(), ".pytest_temp", "test_data", "MM1422_ome-tiffs"
    )

    subfolders = [
        f for f in os.listdir(test_data) if os.path.isdir(pjoin(test_data, f))
    ]

    # specific folder
    one_folder = pjoin(test_data, subfolders[0])
    # random folder
    rand_folder = pjoin(test_data, random.choice(subfolders))
    # return path to unzipped folder containing test images
    # as well as specific folder paths
    yield test_data, one_folder, rand_folder


@pytest.fixture(scope="function")
def setup_mm1422_singlepage_tiffs():
    test_data = pjoin(
        os.getcwd(), ".pytest_temp", "test_data", "MM1422_singlepage-tiffs"
    )

    subfolders = [
        f for f in os.listdir(test_data) if os.path.isdir(pjoin(test_data, f))
    ]

    # specific folder
    one_folder = pjoin(test_data, subfolders[0])
    # random folder
    rand_folder = pjoin(test_data, random.choice(subfolders))
    # return path to unzipped folder containing test images
    # as well as specific folder paths
    yield test_data, one_folder, rand_folder


@pytest.fixture(scope="function")
def setup_mm2gamma_zarr():
    test_data = pjoin(os.getcwd(), ".pytest_temp", "test_data", "MM20_zarr")

    zp = pjoin(test_data, "mm2.0-20201209_4p_2t_5z_1c_512k_1.zarr")

    # return path to unzipped folder containing test images
    # as well as specific folder paths
    yield zp


@pytest.fixture(scope="session")
def setup_hcs_ref():
    yield pjoin(
        os.getcwd(),
        ".pytest_temp",
        "test_data",
        "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
    )


@pytest.fixture(scope="function")
def setup_pycromanager_test_data():
    test_data = pjoin(
        os.getcwd(), ".pytest_temp", "test_data", "MM20_pycromanager"
    )
    datasets = [
        "mm2.0-20210713_pm0.13.2_2c_1",
        "mm2.0-20210713_pm0.13.2_2c_7z_1",
        "mm2.0-20210713_pm0.13.2_2p_3t_2c_1",
        "mm2.0-20210713_pm0.13.2_2p_3t_2c_7z_1",
        "mm2.0-20210713_pm0.13.2_3t_2c_1",
        "mm2.0-20210713_pm0.13.2_3t_2c_7z_1",
        "mm2.0-20210713_pm0.13.2_3t_7z_1",
        "mm2.0-20210713_pm0.13.2_5t_1",
        "mm2.0-20210713_pm0.13.2_7z_1",
    ]

    dataset_dirs = [pjoin(test_data, ds) for ds in datasets]
    first_dir, rand_dir, ptcz_dir = (
        dataset_dirs[0],
        random.choice(dataset_dirs),
        dataset_dirs[3],
    )

    yield first_dir, rand_dir, ptcz_dir


@pytest.fixture(scope="function")
def ndtiff_v3_labeled_positions(setup_test_data):
    yield pjoin(setup_test_data, "ndtiff_v3_labeled_positions")
