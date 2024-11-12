import re

import pytest
from xarray import DataArray

from iohub.ndtiff import NDTiffDataset, NDTiffFOV
from tests.conftest import ndtiff_v2_datasets, ndtiff_v3_labeled_positions


def pytest_generate_tests(metafunc):
    if "ndtiff_dataset" in metafunc.fixturenames:
        metafunc.parametrize(
            "ndtiff_dataset",
            ndtiff_v2_datasets + [ndtiff_v3_labeled_positions],
        )


def test_dataset_ctx(ndtiff_dataset):
    with NDTiffDataset(ndtiff_dataset) as dataset:
        assert isinstance(dataset, NDTiffDataset)
        assert len(dataset) > 0
        assert "NDTiffDataset" in dataset.__repr__()


def test_dataset_nonexisting(tmpdir):
    with pytest.raises(FileNotFoundError):
        NDTiffDataset(tmpdir / "nonexisting")


def test_dataset_metadata(ndtiff_dataset):
    with NDTiffDataset(ndtiff_dataset) as dataset:
        assert isinstance(dataset.micromanager_metadata, dict)
        assert dataset.micromanager_metadata["Summary"]
        assert isinstance(dataset.micromanager_summary, dict)
        if dataset.stage_positions:
            assert "DefaultXYStage" in dataset.stage_positions[0]


@pytest.mark.parametrize("ndtiff_v2", ndtiff_v2_datasets)
def test_dataset_getitem_v2(ndtiff_v2):
    with NDTiffDataset(ndtiff_v2) as dataset:
        assert isinstance(dataset["0"], NDTiffFOV)
        assert isinstance(dataset[0], NDTiffFOV)


def test_dataset_v3_labeled_positions():
    dataset = NDTiffDataset(ndtiff_v3_labeled_positions)
    assert len(dataset) == 3
    positions = ["Pos0", "Pos1", "Pos2"]
    for (key, fov), name in zip(dataset, positions):
        assert key == name
        assert isinstance(fov, NDTiffFOV)
        assert name in dataset.__repr__()
        assert key in fov.__repr__()
    with pytest.raises(KeyError):
        dataset["0"]
        dataset[0]
    dataset.close()


def test_dataset_iter(ndtiff_dataset):
    with NDTiffDataset(ndtiff_dataset) as dataset:
        for key, fov in dataset:
            assert isinstance(key, str)
            assert isinstance(fov, NDTiffFOV)


def test_dataset_v2_num_positions(ndtiff_dataset):
    with NDTiffDataset(ndtiff_dataset) as dataset:
        p_match = re.search(r"_(\d+)p_", str(ndtiff_dataset))
        assert len(dataset) == int(p_match.group(1)) if p_match else 1


def test_fov_getitem(ndtiff_dataset):
    with NDTiffDataset(ndtiff_dataset) as dataset:
        for _, fov in dataset:
            img = fov[:]
            assert isinstance(img, DataArray)
            assert img.ndim == 5
            assert img[0, 0, 0, 0, 0] >= 0
            for ch in fov.channel_names:
                assert img.sel(time=0, channel=ch, z=0, y=0, x=0) >= 0
