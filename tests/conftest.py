import os
import csv
import shutil
from pathlib import Path

import fsspec
import pytest
from wget import download


def _download_ndtiff_v3_labeled_positions(test_data: Path) -> None:
    ghfs = fsspec.filesystem(
        "github",
        org="micro-manager",
        repo="NDTiffStorage",
        username=os.environ.get("GITHUB_ACTOR"),
        token=os.environ.get("GITHUB_TOKEN"),
    )
    v3_lp = test_data / "ndtiff_v3_labeled_positions"
    Path.mkdir(v3_lp)
    ghfs.get(
        ghfs.ls("test_data/v3/labeled_positions_1"),
        str(v3_lp),
        recursive=True,
    )


def download_data():
    """Download test datasets."""
    test_data = Path.cwd() / ".pytest_temp" / "test_data"
    if not test_data.is_dir():
        Path.mkdir(test_data, parents=True)
        print("\nsetting up temp folder")

    # Zenodo URL
    custom_url = (
        "https://zenodo.org/record/6983916/files/waveOrder_test_data.zip"
    )
    # Reference v0.4 HCS dataset from OME
    # See the last line of
    # https://github.com/ome/ngff/issues/140#issuecomment-1309972511
    ome_hcs_url = "https://zenodo.org/record/8091756/files/20200812-CardiomyocyteDifferentiation14-Cycle1.zarr.zip"  # noqa

    # download files to temp folder
    if not any(test_data.iterdir()):
        print("Downloading test files...")
        for url in (custom_url, ome_hcs_url):
            output = test_data / Path(url).name
            download(url, out=str(output))
            shutil.unpack_archive(output, extract_dir=test_data)
        _download_ndtiff_v3_labeled_positions(test_data)
    return test_data


def subdirs(parent: Path, name: str) -> list[Path]:
    return [d for d in (parent / name).iterdir() if d.is_dir()]


test_datasets = download_data()


mm2gamma_ome_tiffs = subdirs(test_datasets, "MM20_ome-tiffs")


mm2gamma_ome_tiffs_hcs = [p for p in mm2gamma_ome_tiffs if "4p" in p.name]


# This is a dataset with 11 timepoints
# The MDA definition at start of the experiment specifies 20 timepoints
mm2gamma_ome_tiffs_incomplete = (
    test_datasets
    / "MM20_ometiff_incomplete"
    / "mm2.0-20201209_20t_5z_3c_512k_incomplete_1"
)


mm2gamma_singlepage_tiffs = subdirs(test_datasets, "MM20_singlepage-tiffs")


# This is a dataset with 11 timepoints
# The MDA definition at start of the experiment specifies 20 timepoints
mm2gamma_singlepage_tiffs_incomplete = (
    test_datasets
    / "MM20_singlepage_incomplete"
    / "mm2.0-20201209_20t_5z_3c_512k_incomplete_1 2"
)


mm1422_ome_tiffs = subdirs(test_datasets, "MM1422_ome-tiffs")


mm1422_singlepage_tiffs = subdirs(test_datasets, "MM1422_singlepage-tiffs")


mm2gamma_zarr_v01 = (
    test_datasets / "MM20_zarr" / "mm2.0-20201209_4p_2t_5z_1c_512k_1.zarr"
)


hcs_ref = test_datasets / "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr"


ndtiff_v2_datasets = subdirs(test_datasets, "MM20_pycromanager")


ndtiff_v2_ptcz = (
    test_datasets
    / "MM20_pycromanager"
    / "mm2.0-20210713_pm0.13.2_2p_3t_2c_7z_1"
)


ndtiff_v3_labeled_positions = test_datasets / "ndtiff_v3_labeled_positions"


@pytest.fixture
def csv_data_file_1(tmpdir):
    test_csv_1 = tmpdir / "well_names_1.csv"
    csv_data_1 = [
        ["B/03", "D/4"],
    ]
    with open(test_csv_1, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data_1)
    return test_csv_1


@pytest.fixture
def csv_data_file_2(tmpdir):
    test_csv_2 = tmpdir / "well_names_2.csv"
    csv_data_2 = [
        ["D/4", "B/03"],
    ]
    with open(test_csv_2, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data_2)
    return test_csv_2
