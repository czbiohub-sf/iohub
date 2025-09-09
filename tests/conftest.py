import csv
import os
import shutil
from pathlib import Path

import fsspec
import numpy as np
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


@pytest.fixture
def empty_ome_zarr_hcs_v05(tmpdir) -> tuple[Path, tuple[tuple[str, ...], ...]]:
    """Create an empty HCS OME-Zarr v0.5 dataset."""
    example_json_dir = Path(__file__).parent / "ngff" / "static_data" / "v05"
    empty_zarr = tmpdir / "v05.hcs.ome.zarr"
    empty_zarr.mkdir()
    TARGET_FILENAME = "zarr.json"
    shutil.copy(example_json_dir / "plate.json", empty_zarr / TARGET_FILENAME)
    ROWS = ("A", "B")
    COLS = ("1", "2", "3")
    FOVS = ("0", "1", "2", "3")
    RESOLUTIONS = ("0", "1", "2")
    for row in ROWS:
        row_dir = empty_zarr / row
        row_dir.mkdir()
        shutil.copy(example_json_dir / "row.json", row_dir / TARGET_FILENAME)
        for col in COLS:
            col_dir = row_dir / col
            col_dir.mkdir()
            shutil.copy(
                example_json_dir / "well.json", col_dir / TARGET_FILENAME
            )
            for fov in FOVS:
                fov_dir = col_dir / fov
                fov_dir.mkdir()
                shutil.copy(
                    example_json_dir / "image.json", fov_dir / TARGET_FILENAME
                )
                for res in RESOLUTIONS:
                    res_dir = fov_dir / res
                    res_dir.mkdir()
                    shutil.copy(
                        example_json_dir / "array.json",
                        res_dir / TARGET_FILENAME,
                    )
    return empty_zarr, (ROWS, COLS, FOVS, RESOLUTIONS)


@pytest.fixture()
def aqz_ome_zarr_05(tmpdir):
    pytest.importorskip("acquire_zarr")
    import acquire_zarr as aqz

    store_path = tmpdir / "ome_zarr_v0.5.zarr"

    settings = aqz.StreamSettings(
        arrays=[
            aqz.ArraySettings(
                data_type=np.uint16,
                compression=aqz.CompressionSettings(
                    codec=aqz.CompressionCodec.BLOSC_LZ4,
                    compressor=aqz.Compressor.BLOSC1,
                    level=1,
                    shuffle=0,
                ),
                dimensions=[
                    aqz.Dimension(
                        name="t",
                        kind=aqz.DimensionType.TIME,
                        array_size_px=0,
                        chunk_size_px=16,
                        shard_size_chunks=1,
                    ),
                    aqz.Dimension(
                        name="c",
                        kind=aqz.DimensionType.CHANNEL,
                        array_size_px=4,
                        chunk_size_px=1,
                        shard_size_chunks=1,
                    ),
                    aqz.Dimension(
                        name="z",
                        kind=aqz.DimensionType.SPACE,
                        array_size_px=10,
                        chunk_size_px=10,
                        shard_size_chunks=1,
                    ),
                    aqz.Dimension(
                        name="y",
                        kind=aqz.DimensionType.SPACE,
                        array_size_px=48,
                        chunk_size_px=16,
                        shard_size_chunks=3,
                    ),
                    aqz.Dimension(
                        name="x",
                        kind=aqz.DimensionType.SPACE,
                        array_size_px=64,
                        chunk_size_px=16,
                        shard_size_chunks=2,
                    ),
                ],
                downsampling_method=aqz.DownsamplingMethod.MEAN,
            )
        ],
        store_path=str(store_path),
        version=aqz.ZarrVersion.V3,
        max_threads=1,
    )

    stream = aqz.ZarrStream(settings)
    data = np.random.randint(
        0, 2**16 - 1, (32, 4, 10, 48, 64), dtype=np.uint16
    )
    stream.append(data)
    del stream

    return store_path
