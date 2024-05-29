from pathlib import Path

import numpy as np
import pytest

from iohub.daxi import create_mock_daxi_dataset, DaXiFOV
from iohub.clearcontrol import ArrayIndex


@pytest.fixture
def mock_daxi_dataset_path(tmp_path: Path) -> Path:
    ds_path = tmp_path / "daxi_dataset_raw"
    create_mock_daxi_dataset(path=ds_path)
    return ds_path


@pytest.mark.parametrize(
    "key",
    [1,
     (slice(None), 1),
     (0, [1, 2]),
     (-1, np.asarray([0, 3])),
     (slice(1), -2),
     (np.asarray(0),),
     (0, 1, 57, slice(32), slice(32, None)),
     (0, 0, slice(32)),
    ],
)
def test_DaXiFOV_indexing(
    mock_daxi_dataset_path: Path,
    key: ArrayIndex,
) -> None:
    fov = DaXiFOV(mock_daxi_dataset_path)

    # copy of whole array
    similar_arr = fov[slice(None)]

    # checking if indexing works as in numpy
    assert np.array_equal(similar_arr[key], fov[key])


def test_DaXiFOV_cache(
    mock_daxi_dataset_path: Path,
) -> None:
    fov = DaXiFOV(mock_daxi_dataset_path, cache=True)

    volume = fov[0, 0]
    assert np.array_equal(fov._cache_array, volume)

    fov[0, 0, 0]
    assert np.array_equal(fov._cache_array, volume)
    assert fov._cache_key == (0, 0)

    array = fov[1]
    assert np.array_equal(fov._cache_array, array)

    new_array = fov[1]
    assert id(new_array) == id(array)   # same reference, so cache worked

    fov.cache = False
    assert fov._cache_key is None
    assert fov._cache_array is None
