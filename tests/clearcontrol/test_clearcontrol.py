from pathlib import Path

import numpy as np
import pytest

from iohub.clearcontrol import (
    ArrayIndex,
    ClearControlFOV,
    _array_to_blosc_buffer,
    blosc_buffer_to_array,
    create_mock_clear_control_dataset,
)


@pytest.fixture
def mock_clear_control_dataset_path(tmp_path: Path) -> Path:
    ds_path = tmp_path / "dataset.cc"
    create_mock_clear_control_dataset(path=ds_path)
    return ds_path


def test_blosc_buffer(tmp_path: Path) -> None:
    buffer_path = tmp_path / "buffer.blc"
    in_array = np.random.randint(0, 5_000, size=(32, 32))

    _array_to_blosc_buffer(in_array, buffer_path)
    out_array = blosc_buffer_to_array(
        buffer_path, in_array.shape, in_array.dtype
    )

    assert np.allclose(in_array, out_array)


@pytest.mark.parametrize(
    "key",
    [
        1,
        (slice(None), 1),
        (0, [1, 2]),
        (-1, np.asarray([0, 3])),
        (slice(1), -2),
        (np.asarray(0),),
        (0, 0, slice(32)),
    ],
)
def test_CCFOV_indexing(
    mock_clear_control_dataset_path: Path,
    key: ArrayIndex,
) -> None:
    cc = ClearControlFOV(mock_clear_control_dataset_path)

    # copy of whole array
    similar_arr = cc[slice(None)]

    # checking if indexing works as in numpy
    assert np.array_equal(similar_arr[key], cc[key])


def test_CCFOV_metadata(
    mock_clear_control_dataset_path: Path,
) -> None:
    cc = ClearControlFOV(mock_clear_control_dataset_path)
    expected_metadata = {
        "voxel_size_z": 1.0,
        "voxel_size_y": 0.25,
        "voxel_size_x": 0.25,
        "acquisition_type": "NA",
        "time_delta": 45.0,
    }
    metadata = cc.metadata()
    assert metadata == expected_metadata


def test_CCFOV_scales(
    mock_clear_control_dataset_path: Path,
) -> None:
    cc = ClearControlFOV(mock_clear_control_dataset_path)
    zyx_scale = (1.0, 0.25, 0.25)
    time_delta = 45.0
    assert zyx_scale == cc.zyx_scale
    assert time_delta == cc.t_scale


def test_CCFOV_cache(
    mock_clear_control_dataset_path: Path,
) -> None:
    cc = ClearControlFOV(mock_clear_control_dataset_path, cache=True)

    volume = cc[0, 0]
    assert np.array_equal(cc._cache_array, volume)

    cc[0, 0, 0]
    assert np.array_equal(cc._cache_array, volume)
    assert cc._cache_key == (0, 0)

    array = cc[1]
    assert np.array_equal(cc._cache_array, array)

    new_array = cc[1]
    assert id(new_array) == id(array)  # same reference, so cache worked

    cc.cache = False
    assert cc._cache_key is None
    assert cc._cache_array is None
