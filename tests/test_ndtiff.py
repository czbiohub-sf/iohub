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


def test_dataset_nonexisting(tmpdir):
    with pytest.raises(FileNotFoundError):
        NDTiffDataset(tmpdir / "nonexisting")


@pytest.mark.parametrize("ndtiff_v2", ndtiff_v2_datasets)
def test_dataset_getitem_v2(ndtiff_v2):
    with NDTiffDataset(ndtiff_v2) as dataset:
        assert isinstance(dataset["0"], NDTiffFOV)
        assert isinstance(dataset[0], NDTiffFOV)


def test_dataset_v3_labeled_positions():
    dataset = NDTiffDataset(ndtiff_v3_labeled_positions)
    assert len(dataset) == 3
    for (key, fov), name in zip(dataset, ["Pos0", "Pos1", "Pos2"]):
        assert key == name
        assert isinstance(fov, NDTiffFOV)
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
