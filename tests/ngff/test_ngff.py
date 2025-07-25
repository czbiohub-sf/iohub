from __future__ import annotations

import os
import platform
import shutil
import string
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import pytest
import zarr
from hypothesis import HealthCheck, assume, given, settings
from numpy.testing import assert_array_almost_equal, assert_array_equal
from numpy.typing import NDArray
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

if TYPE_CHECKING:
    from _typeshed import StrPath

from iohub.ngff.nodes import (
    TO_DICT_SETTINGS,
    NGFFNode,
    Plate,
    TransformationMeta,
    _case_insensitive_local_fs,
    _open_store,
    _pad_shape,
    open_ome_zarr,
)
from tests.conftest import hcs_ref

short_text_st = st.text(min_size=1, max_size=16)
t_dim_st = st.integers(1, 4)
c_dim_st = st.integers(1, 4)
z_dim_st = st.integers(1, 4)
y_dim_st = st.integers(1, 32)
x_dim_st = st.integers(1, 32)
channel_names_st = c_dim_st.flatmap(
    (
        lambda c_dim: st.lists(
            short_text_st, min_size=c_dim, max_size=c_dim, unique=True
        )
    )
)
short_alpha_numeric = st.text(
    alphabet=list(
        string.ascii_lowercase + string.ascii_uppercase + string.digits
    ),
    min_size=1,
    max_size=16,
)
if os.name == "nt":
    # Windows does not allow certain file names
    _INVALID_NT_FILE_NAMES = (
        ["CON", "PRN", "AUX", "NUL"]
        + ["COM" + str(i) for i in range(10)]
        + ["LPT" + str(i) for i in range(10)]
    )
    short_alpha_numeric = short_alpha_numeric.filter(
        lambda x: x not in _INVALID_NT_FILE_NAMES
    )
tiles_rc_st = st.tuples(t_dim_st, t_dim_st)
plate_axis_names_st = st.lists(
    short_alpha_numeric,
    min_size=1,
    max_size=8,
    unique_by=(lambda x: x.lower()),
)


@st.composite
def _random_array_shape_and_dtype_with_channels(draw, c_dim: int):
    shape = (
        draw(t_dim_st),
        c_dim,
        draw(z_dim_st),
        draw(y_dim_st),
        draw(x_dim_st),
    )
    dtype = draw(
        st.one_of(
            npst.integer_dtypes(),
            npst.unsigned_integer_dtypes(),
            npst.floating_dtypes(),
            npst.boolean_dtypes(),
        )
    )
    return shape, dtype


@st.composite
def _channels_and_random_5d_shape_and_dtype(draw):
    channel_names = draw(channel_names_st)
    shape, dtype = draw(
        _random_array_shape_and_dtype_with_channels(c_dim=len(channel_names))
    )
    return channel_names, shape, dtype


@st.composite
def _channels_and_random_5d(draw):
    channel_names, shape, dtype = draw(
        _channels_and_random_5d_shape_and_dtype()
    )
    random_5d = draw(npst.arrays(dtype, shape=shape))
    return channel_names, random_5d


@given(shape=st.lists(x_dim_st, min_size=1, max_size=10), target=x_dim_st)
@settings(max_examples=16, deadline=1000)
def test_pad_shape(shape, target):
    """Test `iohub.ngff._pad_shape()`"""
    shape = tuple(shape)
    assume(len(shape) <= target)
    new_shape = _pad_shape(shape=shape, target=target)
    assert len(new_shape) == target
    assert new_shape[-len(shape) :] == shape


def test_open_store_create():
    """Test `iohub.ngff._open_store()"""
    for mode in ("a", "w", "w-"):
        with TemporaryDirectory() as temp_dir:
            store_path = os.path.join(temp_dir, "new.zarr")
            root = _open_store(store_path, mode=mode, version="0.4")
            assert isinstance(root, zarr.Group)
            assert isinstance(root.store, zarr.DirectoryStore)
            assert root.store._dimension_separator == "/"
            assert root.store.path == store_path


def test_open_store_create_existing():
    """Test `iohub.ngff._open_store()"""
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "new.zarr")
        g = zarr.open_group(store_path, mode="w-")
        g.store.close()
        with pytest.raises(RuntimeError):
            _ = _open_store(store_path, mode="w-", version="0.4")
        assert _open_store(store_path, mode="w", version="0.4") is not None


def test_open_store_read_nonexist():
    """Test `iohub.ngff._open_store()"""
    for mode in ("r", "r+"):
        with TemporaryDirectory() as temp_dir:
            store_path = os.path.join(temp_dir, "new.zarr")
            with pytest.raises(FileNotFoundError):
                _ = _open_store(store_path, mode=mode, version="0.4")


def test_case_insensitive_local_fs():
    """Test `iohub.ngff._case_insensitive_local_fs()`"""
    match platform.system():
        case "Windows":
            assert _case_insensitive_local_fs() is True
        case "Darwin":
            assert _case_insensitive_local_fs() is True
        case "Linux":
            assert _case_insensitive_local_fs() is False
        case _:
            _ = _case_insensitive_local_fs()


@given(channel_names=channel_names_st)
@settings(max_examples=16)
def test_init_ome_zarr(channel_names):
    """Test `iohub.ngff.open_ome_zarr()`"""
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "ome.zarr")
        dataset = open_ome_zarr(
            store_path, layout="fov", mode="w-", channel_names=channel_names
        )
        assert os.path.isdir(store_path)
        assert dataset.channel_names == channel_names


@pytest.mark.parametrize(
    "basename",
    ["some.zarr", "other.zarr/0/0/0", "random_dir", "napari_ome_zarr"],
)
def test_init_ome_zarr_overwrite_non_zarr(tmp_path, basename):
    """Test `iohub.ngff.open_ome_zarr()`"""
    store_path = tmp_path / basename
    store_path.mkdir(parents=True)
    some_child_directory = store_path / "some_other_directory"
    some_child_directory.mkdir()
    if ".zarr" not in basename:
        with pytest.raises(ValueError):
            _ = open_ome_zarr(
                store_path, layout="fov", mode="w", channel_names=["channel"]
            )
        assert some_child_directory.exists()
    assert (
        open_ome_zarr(
            store_path,
            layout="fov",
            mode="w",
            channel_names=["channel"],
            disable_path_checking=True,
        )
        is not None
    )
    assert not some_child_directory.exists()


@contextmanager
def _temp_ome_zarr(
    image_5d: NDArray, channel_names: list[str], arr_name: str, **kwargs
):
    """Helper function to generate a temporary OME-Zarr store.

    Parameters
    ----------
    image_5d : NDArray
    channel_names : list[str]
    arr_name : str

    Yields
    ------
    Position
    """
    try:
        temp_dir = TemporaryDirectory()
        dataset = open_ome_zarr(
            os.path.join(temp_dir.name, "ome.zarr"),
            layout="fov",
            mode="a",
            channel_names=channel_names,
        )
        dataset.create_image(arr_name, image_5d, **kwargs)
        yield dataset
    finally:
        dataset.close()
        temp_dir.cleanup()


@contextmanager
def _temp_ome_zarr_plate(
    image_5d: NDArray,
    channel_names: list[str],
    arr_name: str,
    position_list: list[tuple[str, str, str]],
    **kwargs,
):
    """Helper function to generate a temporary OME-Zarr store.

    Parameters
    ----------
    image_5d : NDArray
    channel_names : list[str]
    arr_name : str
    position_list : list[tuple[str, str, str]]

    Yields
    ------
    Position
    """
    try:
        temp_dir = TemporaryDirectory()
        dataset = open_ome_zarr(
            os.path.join(temp_dir.name, "ome.zarr"),
            layout="hcs",
            mode="a",
            channel_names=channel_names,
        )
        for position in position_list:
            pos = dataset.create_position(
                position[0], position[1], position[2]
            )
            pos.create_image(arr_name, image_5d, **kwargs)
        yield dataset
    finally:
        dataset.close()
        temp_dir.cleanup()


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    arr_name=short_alpha_numeric,
)
@settings(
    max_examples=16,
    deadline=2000,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_write_ome_zarr(channels_and_random_5d, arr_name):
    """Test `iohub.ngff.Position.__setitem__()`"""
    channel_names, random_5d = channels_and_random_5d
    with _temp_ome_zarr(random_5d, channel_names, arr_name) as dataset:
        assert_array_almost_equal(dataset[arr_name][:], random_5d)
        # round-trip test with the offical reader implementation
        ext_reader = Reader(parse_url(dataset.zgroup.store.path))
        node = list(ext_reader())[0]
        assert node.metadata["channel_names"] == channel_names
        assert node.specs[0].datasets == [arr_name]
        assert node.data[0].shape == random_5d.shape
        assert node.data[0].dtype == random_5d.dtype


@given(
    ch_shape_dtype=_channels_and_random_5d_shape_and_dtype(),
    arr_name=short_alpha_numeric,
)
@settings(
    max_examples=16,
    deadline=2000,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_create_zeros(ch_shape_dtype, arr_name):
    """Test `iohub.ngff.Position.create_zeros()`"""
    channel_names, shape, dtype = ch_shape_dtype
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "ome.zarr")
        dataset = open_ome_zarr(
            store_path, layout="fov", mode="w-", channel_names=channel_names
        )
        dataset.create_zeros(name=arr_name, shape=shape, dtype=dtype)
        assert os.listdir(os.path.join(store_path, arr_name)) == [".zarray"]
        assert not dataset[arr_name][:].any()
        assert dataset[arr_name].shape == shape
        assert dataset[arr_name].dtype == dtype


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    arr_name=short_alpha_numeric,
)
@settings(
    max_examples=16,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_ome_zarr_to_dask(channels_and_random_5d, arr_name):
    """Test `iohub.ngff.Position.data` to dask"""
    channel_names, random_5d = channels_and_random_5d
    with _temp_ome_zarr(random_5d, channel_names, "0") as dataset:
        assert_array_almost_equal(
            dataset.data.dask_array().compute(), random_5d
        )
    with _temp_ome_zarr(random_5d, channel_names, arr_name) as dataset:
        assert_array_almost_equal(
            dataset[arr_name].dask_array().compute(), random_5d
        )


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    arr_name=short_alpha_numeric,
)
@settings(
    max_examples=16,
    deadline=2000,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_position_data(channels_and_random_5d, arr_name):
    """Test `iohub.ngff.Position.data`"""
    channel_names, random_5d = channels_and_random_5d
    assume(arr_name != "0")
    with _temp_ome_zarr(random_5d, channel_names, "0") as dataset:
        assert_array_almost_equal(dataset.data.numpy(), random_5d)
    with pytest.raises(KeyError):
        with _temp_ome_zarr(random_5d, channel_names, arr_name) as dataset:
            _ = dataset.data


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    arr_name=short_alpha_numeric,
)
@settings(
    max_examples=16,
    deadline=2000,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_ome_zarr_to_tensorstore(channels_and_random_5d, arr_name):
    """Test `iohub.ngff.Position.data` to tensorstore"""
    channel_names, random_5d = channels_and_random_5d
    with _temp_ome_zarr(random_5d, channel_names, arr_name) as dataset:
        tstore = dataset[arr_name].tensorstore()
        assert_array_equal(tstore, random_5d)
        zeros = np.zeros_like(random_5d)
        tstore[...].write(zeros).result()
        with open_ome_zarr(
            dataset.zgroup.store.path, mode="r"
        ) as read_only_dataset:
            assert_array_equal(read_only_dataset[arr_name].numpy(), zeros)
            read_only_tstore = read_only_dataset[arr_name].tensorstore()
            with pytest.raises(ValueError):
                read_only_tstore[0] = 1


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    arr_name=short_alpha_numeric,
)
@settings(
    max_examples=16,
    deadline=2000,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_append_channel(channels_and_random_5d, arr_name):
    """Test `iohub.ngff.Position.append_channel()`"""
    channel_names, random_5d = channels_and_random_5d
    assume(len(channel_names) > 1)
    with _temp_ome_zarr(
        random_5d[:, :-1], channel_names[:-1], arr_name
    ) as dataset:
        dataset.append_channel(channel_names[-1], resize_arrays=True)
        dataset[arr_name][:, -1] = random_5d[:, -1]
        assert_array_almost_equal(dataset[arr_name][:], random_5d)


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    arr_name=short_alpha_numeric,
    new_channel=short_text_st,
)
@settings(
    max_examples=16,
    deadline=2000,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_rename_channel(channels_and_random_5d, arr_name, new_channel):
    """Test `iohub.ngff.Position.rename_channel()`"""
    channel_names, random_5d = channels_and_random_5d
    assume(new_channel not in channel_names)
    with _temp_ome_zarr(random_5d, channel_names, arr_name) as dataset:
        dataset.rename_channel(old=channel_names[0], new=new_channel)
        assert new_channel in dataset.channel_names
        assert dataset.metadata.omero.channels[0].label == new_channel


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    arr_name=short_alpha_numeric,
)
@settings(deadline=None)
def test_rename_well(channels_and_random_5d, arr_name):
    """Test `iohub.ngff.Position.rename_well()`"""
    channel_names, random_5d = channels_and_random_5d

    position_list = [["A", "1", "0"], ["C", "4", "0"]]
    with _temp_ome_zarr_plate(
        random_5d, channel_names, arr_name, position_list
    ) as dataset:
        assert dataset.zgroup["A/1"]
        with pytest.raises(KeyError):
            dataset.zgroup["B/2"]
        assert "A" in [r[0] for r in dataset.rows()]
        assert "B" not in [r[0] for r in dataset.rows()]
        assert "A" in [row.name for row in dataset.metadata.rows]
        assert "B" not in [row.name for row in dataset.metadata.rows]
        assert "1" in [col.name for col in dataset.metadata.columns]
        assert "2" not in [col.name for col in dataset.metadata.columns]
        assert "C" in [row.name for row in dataset.metadata.rows]
        assert "4" in [col.name for col in dataset.metadata.columns]

        dataset.rename_well("A/1", "B/2")

        assert dataset.zgroup["B/2"]
        with pytest.raises(KeyError):
            dataset.zgroup["A/1"]
        assert "A" not in [r[0] for r in dataset.rows()]
        assert "B" in [r[0] for r in dataset.rows()]
        assert "A" not in [row.name for row in dataset.metadata.rows]
        assert "B" in [row.name for row in dataset.metadata.rows]
        assert "1" not in [col.name for col in dataset.metadata.columns]
        assert "2" in [col.name for col in dataset.metadata.columns]
        assert "C" in [row.name for row in dataset.metadata.rows]
        assert "4" in [col.name for col in dataset.metadata.columns]

        # destination exists
        with pytest.raises(ValueError):
            dataset.rename_well("B/2", "C/4")

        # source doesn't exist
        with pytest.raises(ValueError):
            dataset.rename_well("Q/1", "Q/2")

        # invalid well names
        with pytest.raises(ValueError):
            dataset.rename_well("B/2", " A/1")
        with pytest.raises(ValueError):
            dataset.rename_well("B/2", "A/?")


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    arr_name=short_alpha_numeric,
)
@settings(
    max_examples=16,
    deadline=2000,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_update_channel(channels_and_random_5d, arr_name):
    """Test `iohub.ngff.Position.update_channel()`"""
    channel_names, random_5d = channels_and_random_5d
    assume(len(channel_names) > 1)
    with _temp_ome_zarr(
        random_5d[:, :-1], channel_names[:-1], arr_name
    ) as dataset:
        for i, ch in enumerate(dataset.channel_names):
            dataset.update_channel(
                chan_name=ch, target=arr_name, data=random_5d[:, -1]
            )
            assert_array_almost_equal(
                dataset[arr_name][:, i], random_5d[:, -1]
            )


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    arr_name=short_alpha_numeric,
)
@settings(
    max_examples=16,
    deadline=2000,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_write_more_channels(channels_and_random_5d, arr_name):
    """Test `iohub.ngff.Position.create_image()`"""
    channel_names, random_5d = channels_and_random_5d
    assume(len(channel_names) > 1)
    with pytest.raises(ValueError):
        with _temp_ome_zarr(random_5d, channel_names[:-1], arr_name) as _:
            pass


@given(
    ch_shape_dtype=_channels_and_random_5d_shape_and_dtype(),
    arr_name=short_alpha_numeric,
)
def test_set_transform_image(ch_shape_dtype, arr_name):
    """Test `iohub.ngff.Position.set_transform()`"""
    channel_names, shape, dtype = ch_shape_dtype
    transform = [
        TransformationMeta(type="translation", translation=(1, 2, 3, 4, 5))
    ] * len(channel_names)
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "ome.zarr")
        with open_ome_zarr(
            store_path, layout="fov", mode="w-", channel_names=channel_names
        ) as dataset:
            dataset.create_zeros(name=arr_name, shape=shape, dtype=dtype)
            assert dataset.metadata.multiscales[0].datasets[
                0
            ].coordinate_transformations == [
                TransformationMeta(type="identity")
            ]
            dataset.set_transform(image=arr_name, transform=transform)
            assert (
                dataset.metadata.multiscales[0]
                .datasets[0]
                .coordinate_transformations
                == transform
            )
        # read data with an external reader
        ext_reader = Reader(parse_url(dataset.zgroup.store.path))
        node = list(ext_reader())[0]
        assert node.metadata["coordinateTransformations"][0] == [
            translate.model_dump(**TO_DICT_SETTINGS) for translate in transform
        ]


input_transformations = [
    ([TransformationMeta(type="identity")], []),
    ([TransformationMeta(type="scale", scale=(1.0, 2.0, 3.0, 4.0, 5.0))], []),
    (
        [
            TransformationMeta(
                type="translation", translation=(1.0, 2.0, 3.0, 4.0, 5.0)
            )
        ],
        [],
    ),
    (
        [
            TransformationMeta(type="scale", scale=(2.0, 2.0, 2.0, 2.0, 2.0)),
            TransformationMeta(
                type="translation", translation=(1.0, 1.0, 1.0, 1.0, 1.0)
            ),
        ],
        [
            TransformationMeta(type="scale", scale=(2.0, 2.0, 2.0, 2.0, 2.0)),
            TransformationMeta(
                type="translation", translation=(1.0, 1.0, 1.0, 1.0, 1.0)
            ),
        ],
    ),
]
target_scales = [
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [4.0, 4.0, 4.0, 4.0, 4.0],
]
target_translations = [
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [2.0, 2.0, 2.0, 2.0, 2.0],
]


@pytest.mark.parametrize(
    "transforms",
    [
        (saved, target)
        for saved, target in zip(input_transformations, target_scales)
    ],
)
@given(
    ch_shape_dtype=_channels_and_random_5d_shape_and_dtype(),
    arr_name=short_alpha_numeric,
)
def test_get_effective_scale_image(transforms, ch_shape_dtype, arr_name):
    """Test `iohub.ngff.Position.get_effective_scale()`"""
    (fov_transform, img_transform), expected_scale = transforms
    channel_names, shape, dtype = ch_shape_dtype
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "ome.zarr")
        with open_ome_zarr(
            store_path, layout="fov", mode="w-", channel_names=channel_names
        ) as dataset:
            dataset.create_zeros(name=arr_name, shape=shape, dtype=dtype)
            dataset.set_transform(image="*", transform=fov_transform)
            dataset.set_transform(image=arr_name, transform=img_transform)
            scale = dataset.get_effective_scale(image=arr_name)
            assert scale == expected_scale


@pytest.mark.parametrize(
    "transforms",
    [
        (saved, target)
        for saved, target in zip(input_transformations, target_translations)
    ],
)
@given(
    ch_shape_dtype=_channels_and_random_5d_shape_and_dtype(),
    arr_name=short_alpha_numeric,
)
def test_get_effective_translation_image(transforms, ch_shape_dtype, arr_name):
    """Test `iohub.ngff.Position.get_effective_translation()`"""
    (fov_transform, img_transform), expected_translation = transforms
    channel_names, shape, dtype = ch_shape_dtype
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "ome.zarr")
        with open_ome_zarr(
            store_path, layout="fov", mode="w-", channel_names=channel_names
        ) as dataset:
            dataset.create_zeros(name=arr_name, shape=shape, dtype=dtype)
            dataset.set_transform(image="*", transform=fov_transform)
            dataset.set_transform(image=arr_name, transform=img_transform)
            translation = dataset.get_effective_translation(image=arr_name)
            assert translation == expected_translation


@given(
    ch_shape_dtype=_channels_and_random_5d_shape_and_dtype(),
    arr_name=short_alpha_numeric,
)
def test_set_transform_fov(ch_shape_dtype, arr_name):
    """Test `iohub.ngff.Position.set_transform()`"""
    channel_names, shape, dtype = ch_shape_dtype
    transform = [
        TransformationMeta(type="translation", translation=(1, 2, 3, 4, 5))
    ] * len(channel_names)
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "ome.zarr")
        with open_ome_zarr(
            store_path, layout="fov", mode="w-", channel_names=channel_names
        ) as dataset:
            dataset.create_zeros(name=arr_name, shape=shape, dtype=dtype)
            assert dataset.metadata.multiscales[
                0
            ].coordinate_transformations == [
                TransformationMeta(type="identity")
            ]
            dataset.set_transform(image="*", transform=transform)
            assert (
                dataset.metadata.multiscales[0].coordinate_transformations
                == transform
            )
        # read data with plain zarr
        group = zarr.open(store_path)
        assert group.attrs["multiscales"][0]["coordinateTransformations"] == [
            translate.model_dump(**TO_DICT_SETTINGS) for translate in transform
        ]


@pytest.mark.parametrize("image_name", ["0", "1", "a", "*"])
def test_set_scale(image_name):
    """Test `iohub.ngff.Position.set_scale()`"""
    translation = [float(t) for t in range(1, 6)]
    scale = [float(s) for s in range(5, 0, -1)]
    array_name = "0" if image_name == "*" else image_name
    new_scale = 10.0
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "ome.zarr")
        with open_ome_zarr(
            store_path, layout="fov", mode="w-", channel_names=["a", "b"]
        ) as dataset:
            dataset.create_zeros(
                name=array_name,
                shape=(1, 2, 4, 8, 16),
                dtype=int,
                transform=[
                    TransformationMeta(
                        type="translation", translation=translation
                    ),
                    TransformationMeta(type="scale", scale=scale),
                ],
            )
            with pytest.raises(ValueError):
                dataset.set_scale(
                    image=image_name, axis_name="z", new_scale=-1.0
                )
            with pytest.raises(KeyError):
                dataset.set_scale(
                    image="nonexistent", axis_name="z", new_scale=9.0
                )
            assert dataset.scale[-3] == 3.0
            dataset.set_scale(
                image=image_name, axis_name="z", new_scale=new_scale
            )
            if image_name == "*":
                assert dataset.scale[-3] == new_scale * 3.0
            else:
                assert dataset.scale[-3] == new_scale
            assert dataset.get_effective_translation(array_name) == translation
            for tf in dataset.zattrs["iohub"]["previous_transforms"][0][
                "transforms"
            ]:
                if tf["type"] == "scale":
                    assert tf["scale"] == scale


@given(channel_names=channel_names_st)
@settings(max_examples=16)
def test_set_contrast_limits(channel_names):
    """Test `iohub.ngff.Position.set_contrast_limits()`"""
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "ome.zarr")
        dataset = open_ome_zarr(
            store_path, layout="fov", mode="a", channel_names=channel_names
        )
        # Create a simple small array - exact shape/dtype doesn't matter
        dataset.create_zeros(
            "data", shape=(1, len(channel_names), 1, 4, 4), dtype=float
        )

        # Store the initial window settings for all channels
        initial_windows = {}
        for ch_name in channel_names:
            ch_idx = dataset.get_channel_index(ch_name)
            initial_windows[ch_name] = dataset.metadata.omero.channels[
                ch_idx
            ].window

        # Test setting contrast limits for the first channel only
        target_channel = channel_names[0]
        window = {"start": 10.0, "end": 100.0, "min": 0.0, "max": 255.0}

        # Set contrast limits
        dataset.set_contrast_limits(target_channel, window)

        # Check that the contrast limits
        # were set correctly for the target channel
        channel_index = dataset.get_channel_index(target_channel)
        channel_window = dataset.metadata.omero.channels[channel_index].window
        assert channel_window is not None
        assert channel_window["start"] == window["start"]
        assert channel_window["end"] == window["end"]
        assert channel_window["min"] == window["min"]
        assert channel_window["max"] == window["max"]

        # Check that other channels were not affected (if any exist)
        for ch_name in channel_names[1:]:
            ch_idx = dataset.get_channel_index(ch_name)
            assert (
                dataset.metadata.omero.channels[ch_idx].window
                == initial_windows[ch_name]
            )


@given(channel_names=channel_names_st)
@settings(max_examples=16)
def test_create_tiled(channel_names):
    """Test that `iohub.ngff.open_ome_zarr()` can create
    an empty OME-Zarr store with 'tiled' layout."""
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "tiled.zarr")
        dataset = open_ome_zarr(
            store_path, layout="tiled", mode="a", channel_names=channel_names
        )
        assert os.path.isdir(store_path)
        assert dataset.channel_names == channel_names


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    grid_shape=tiles_rc_st,
    arr_name=short_alpha_numeric,
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_make_tiles(channels_and_random_5d, grid_shape, arr_name):
    """Test `iohub.ngff.TiledPosition.make_tiles()` and  `...get_tile()`"""
    with TemporaryDirectory() as temp_dir:
        channel_names, random_5d = channels_and_random_5d
        store_path = os.path.join(temp_dir, "tiled.zarr")
        with open_ome_zarr(
            store_path, layout="tiled", mode="a", channel_names=channel_names
        ) as dataset:
            tiles = dataset.make_tiles(
                name=arr_name,
                grid_shape=grid_shape,
                tile_shape=random_5d.shape,
                dtype=random_5d.dtype,
                chunk_dims=2,
            )
            assert tiles.rows == grid_shape[0]
            assert tiles.columns == grid_shape[1]
            assert tiles.tiles == grid_shape
            assert tiles.shape[-2:] == (
                grid_shape[-2] * random_5d.shape[-2],
                grid_shape[-1] * random_5d.shape[-1],
            )
            assert tiles.tile_shape == _pad_shape(
                random_5d.shape[-2:], target=5
            )
            assert tiles.dtype == random_5d.dtype
            for args in ((1.01, 1), (0, 0, 0)):
                with pytest.raises(TypeError):
                    tiles.get_tile(*args)
            for args in ((0, 0, (0,) * 2), (0, 0, (0,) * 4)):
                with pytest.raises(IndexError):
                    tiles.get_tile(*args)


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    grid_shape=tiles_rc_st,
    arr_name=short_alpha_numeric,
)
@settings(
    max_examples=16,
    deadline=2000,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_write_read_tiles(channels_and_random_5d, grid_shape, arr_name):
    """Test `iohub.ngff.TiledPosition.write_tile()` and `...get_tile()`"""
    channel_names, random_5d = channels_and_random_5d

    def _tile_data(tiles):
        for row in range(tiles.rows):
            for column in range(tiles.columns):
                yield (
                    (
                        random_5d
                        / (tiles.rows * tiles.columns + 1)
                        * (row * column + 1)
                    ).astype(random_5d.dtype),
                    row,
                    column,
                )

    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "tiled.zarr")
        with open_ome_zarr(
            store_path, layout="tiled", mode="w-", channel_names=channel_names
        ) as dataset:
            tiles = dataset.make_tiles(
                name=arr_name,
                grid_shape=grid_shape,
                tile_shape=random_5d.shape,
                dtype=random_5d.dtype,
                chunk_dims=2,
            )
            for data, row, column in _tile_data(tiles):
                tiles.write_tile(data, row, column)
        with open_ome_zarr(
            store_path, layout="tiled", mode="r", channel_names=channel_names
        ) as dataset:
            for data, row, column in _tile_data(tiles):
                read = tiles.get_tile(row, column)
                assert_array_almost_equal(data, read)


@given(channel_names=channel_names_st)
@settings(max_examples=16)
def test_create_hcs(channel_names):
    """Test `iohub.ngff.open_ome_zarr()`"""
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "hcs.zarr")
        dataset = open_ome_zarr(
            store_path, layout="hcs", mode="a", channel_names=channel_names
        )
        assert os.path.isdir(store_path)
        assert dataset.channel_names == channel_names


def test_open_hcs_create_empty():
    """Test `iohub.ngff.open_ome_zarr()`"""
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "hcs.zarr")
        dataset = open_ome_zarr(
            store_path, layout="hcs", mode="a", channel_names=["GFP"]
        )
        assert dataset.zgroup.store.path == store_path
        dataset.close()
        with pytest.raises(FileExistsError):
            _ = open_ome_zarr(
                store_path, layout="hcs", mode="w-", channel_names=["GFP"]
            )
        with pytest.raises(ValueError):
            _ = open_ome_zarr(store_path, layout="hcs", mode="x")
        with pytest.raises(FileNotFoundError):
            _ = open_ome_zarr("do-not-exist.zarr", layout="hcs", mode="r+")
        with pytest.raises(ValueError):
            dataset = open_ome_zarr(store_path, layout="hcs", mode="r+")


@contextmanager
def _temp_copy(src: StrPath):
    """Create a temporary copy of data on disk."""
    try:
        temp_dir = TemporaryDirectory()
        yield shutil.copytree(src, temp_dir.name, dirs_exist_ok=True)
    finally:
        temp_dir.cleanup()


@given(wrong_channel_name=channel_names_st)
def test_get_channel_index(wrong_channel_name):
    """Test `iohub.ngff.NGFFNode.get_channel_axis()`"""
    assume(wrong_channel_name != "DAPI")
    with open_ome_zarr(hcs_ref, layout="hcs", mode="r+") as dataset:
        assert dataset.get_channel_index("DAPI") == 0
        with pytest.raises(ValueError):
            _ = dataset.get_channel_index(wrong_channel_name)


def test_get_axis_index():
    with open_ome_zarr(hcs_ref, layout="hcs", mode="r+") as dataset:
        position = dataset["B/03/0"]

        assert position.axis_names == ["c", "z", "y", "x"]

        assert position.get_axis_index("z") == 1
        assert position.get_axis_index("Z") == 1

        with pytest.raises(ValueError):
            _ = position.get_axis_index("t")

        with pytest.raises(ValueError):
            _ = position.get_axis_index("DOG")


def test_ngff_node_contains_cross_platform(caplog):
    """Test `iohub.ngff.NGFFNode.__contains__()` on multiple platforms."""
    with open_ome_zarr(hcs_ref, layout="hcs", mode="r") as dataset:
        assert "B" in dataset
        match platform.system():
            case "Linux":
                assert "b" not in dataset
            case "Windows" | "Darwin":
                assert "b" in dataset
                assert any(
                    "Key 'b' matched" in r.message for r in caplog.records
                )


@given(
    row=short_alpha_numeric, col=short_alpha_numeric, pos=short_alpha_numeric
)
@settings(max_examples=16, deadline=2000)
def test_modify_hcs_ref(row: str, col: str, pos: str):
    """Test `iohub.ngff.open_ome_zarr()`"""
    assume((row.lower() != "b"))
    with _temp_copy(hcs_ref) as store_path:
        with open_ome_zarr(store_path, layout="hcs", mode="r+") as dataset:
            assert dataset.axes[0].name == "c"
            assert dataset.channel_names == ["DAPI"]
            position = dataset["B/03/0"]
            assert position[0].shape == (1, 2, 2160, 5120)
            position.append_channel("GFP", resize_arrays=True)
            assert position.channel_names == ["DAPI", "GFP"]
            assert position[0].shape == (2, 2, 2160, 5120)
            new_pos_path = "/".join([row, col, pos])
            assume(new_pos_path not in dataset)
            new_pos = dataset.create_position(row, col, pos)
            new_pos.create_zeros("0", position[0].shape, position[0].dtype)
            assert not dataset[f"{new_pos_path}/0"][:].any()


@given(row_names=plate_axis_names_st, col_names=plate_axis_names_st)
@settings(max_examples=16, deadline=2000)
def test_create_well(row_names: list[str], col_names: list[str]):
    """Test `iohub.ngff.Plate.create_well()`"""
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "hcs.zarr")
        dataset = open_ome_zarr(
            store_path, layout="hcs", mode="a", channel_names=["GFP"]
        )
        for row_name in row_names:
            for col_name in col_names:
                dataset.create_well(row_name, col_name)
        assert [
            c["name"] for c in dataset.zattrs["plate"]["columns"]
        ] == col_names
        assert [
            r["name"] for r in dataset.zattrs["plate"]["rows"]
        ] == row_names


def test_create_case_sensitive_well(tmp_path):
    """Test `iohub.ngff.Plate.create_well()` with case-sensitive names."""
    store_path = tmp_path / "hcs.zarr"
    with open_ome_zarr(
        store_path, layout="hcs", mode="w-", channel_names=["1", "2"]
    ) as dataset:
        well = dataset.create_well("A", "B")
        fov = well.create_position("0")
        fov.create_zeros("0", shape=(1, 2, 3, 4, 5), dtype=int)
        match platform.system():
            case "Windows" | "Darwin":
                with pytest.raises(FileExistsError):
                    dataset.create_well("a", "B")
                with pytest.raises(FileExistsError):
                    dataset.create_well("A", "b")
                new_well = dataset.create_well("a", "1")
                expected_rows = 1
            case "Linux":
                new_well = dataset.create_well("a", "b")
                expected_rows = 2
        new_fov = new_well.create_position("0")
        new_fov.create_zeros("0", shape=(1, 2, 3, 4, 5), dtype=int)
    with open_ome_zarr(store_path) as dataset:
        assert len(dataset.metadata.rows) == expected_rows
        assert len(list(dataset.rows())) == expected_rows
        assert len(dataset.metadata.columns) == 2


@given(
    row=short_alpha_numeric, col=short_alpha_numeric, pos=short_alpha_numeric
)
def test_create_position(row, col, pos):
    """Test `iohub.ngff.Plate.create_position()`"""
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "hcs.zarr")
        dataset = open_ome_zarr(
            store_path, layout="hcs", mode="a", channel_names=["GFP"]
        )
        _ = dataset.create_position(row_name=row, col_name=col, pos_name=pos)
        assert [c["name"] for c in dataset.zattrs["plate"]["columns"]] == [col]
        assert [r["name"] for r in dataset.zattrs["plate"]["rows"]] == [row]
        assert os.path.isdir(os.path.join(store_path, row, col, pos))
        assert dataset[row][col].metadata.images[0].path == pos


@given(channels_and_random_5d=_channels_and_random_5d())
def test_position_scale(channels_and_random_5d):
    """Test `iohub.ngff.Position.scale`"""
    channel_names, random_5d = channels_and_random_5d
    scale = list(range(1, 6))
    transform = [TransformationMeta(type="scale", scale=scale)]
    with _temp_ome_zarr(
        random_5d, channel_names, "0", transform=transform
    ) as dataset:
        assert dataset.scale == scale


def test_combine_fovs_to_hcs():
    fovs = {}
    fov_paths = ("A/1/0", "B/1/0", "H/12/9")
    with open_ome_zarr(hcs_ref) as hcs_store:
        fov = hcs_store["B/03/0"]
        array = fov[0].numpy()
        channel_names = fov.channel_names
        old_omero_name = fov.metadata.omero.name
        for path in fov_paths:
            fovs[path] = fov
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "combined.zarr")
        Plate.from_positions(store_path, fovs).close()
        with open_ome_zarr(store_path, layout="hcs", mode="r") as dataset:
            assert len(dataset.metadata.rows) == 3
            assert len(dataset.metadata.columns) == 2
            for fov_path in fov_paths:
                copied_fov = dataset[fov_path]
                assert copied_fov.channel_names == channel_names
                new_omero_name = copied_fov.metadata.omero.name
                assert new_omero_name != old_omero_name
                assert new_omero_name == fov_path[-1]
                assert_array_equal(dataset[fov_path]["0"].numpy(), array)


def test_hcs_external_reader(tmp_path):
    store_path = tmp_path / "hcs.zarr"
    fov_name_parts = (("A", "1", "7"), ("B", "1", "7"), ("H", "12", "7"))
    y_size, x_size = (128, 100)
    with open_ome_zarr(
        store_path, layout="hcs", mode="a", channel_names=["A", "B"]
    ) as dataset:
        for name in fov_name_parts:
            fov = dataset.create_position(*name)
            fov.create_zeros("0", shape=(1, 2, 3, y_size, x_size), dtype=int)
        n_rows = len(dataset.metadata.rows)
        n_cols = len(dataset.metadata.columns)
    plate = list(Reader(parse_url(store_path))())[0]
    assert plate.data[0].shape == (1, 2, 3, y_size * n_rows, x_size * n_cols)
    assert plate.data[0].dtype == int
    assert not plate.data[0].any()
    assert plate.metadata["channel_names"] == ["A", "B"]
