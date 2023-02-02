from __future__ import annotations

import pytest
import logging
import os
import string
import shutil
from contextlib import contextmanager
from tempfile import TemporaryDirectory
import zarr
import numpy as np
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from numpy.testing import assert_array_almost_equal
from hypothesis import given, settings, assume, strategies as st
import hypothesis.extra.numpy as npst
from typing import TYPE_CHECKING
from numpy.typing import NDArray

if TYPE_CHECKING:
    from _typeshed import StrPath

from iohub.ngff import _pad_shape, open_store, OMEZarr, HCSZarr


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
row_names_st = st.lists(
    short_alpha_numeric,
    min_size=1,
    max_size=8,
    unique_by=(lambda x: x.lower()),
)
col_names_st = st.lists(
    short_alpha_numeric,
    min_size=1,
    max_size=8,
    unique_by=(lambda x: x.lower()),
)


@st.composite
def _random_5d_with_channels(draw, c_dim: int):
    arr_shape = (
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
    return draw(npst.arrays(dtype, shape=arr_shape))


@st.composite
def _channels_and_random_5d(draw):
    channel_names = draw(channel_names_st)
    random_5d = draw(_random_5d_with_channels(c_dim=len(channel_names)))
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
    """Test `iohub.ngff.open_store()"""
    for mode in ("a", "w", "w-"):
        with TemporaryDirectory() as temp_dir:
            store_path = os.path.join(temp_dir, "new.zarr")
            root = open_store(store_path, mode=mode, version="0.4")
            assert isinstance(root, zarr.Group)
            assert isinstance(root.store, zarr.DirectoryStore)
            assert root.store._dimension_separator == "/"
            assert root.store.path == store_path


def test_open_store_create_existing():
    """Test `iohub.ngff.open_store()"""
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "new.zarr")
        g = zarr.open_group(store_path, mode="w")
        g.store.close()
        with pytest.raises(RuntimeError):
            _ = open_store(store_path, mode="w-", version="0.4")


def test_open_store_read_nonexist():
    """Test `iohub.ngff.open_store()"""
    for mode in ("r", "r+"):
        with TemporaryDirectory() as temp_dir:
            store_path = os.path.join(temp_dir, "new.zarr")
            with pytest.raises(FileNotFoundError):
                _ = open_store(store_path, mode=mode, version="0.4")


@given(channel_names=channel_names_st)
@settings(max_examples=16)
def test_init_ome_zarr(channel_names):
    """Test `iohub.ngff.OMEZarr.open()`"""
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "ome.zarr")
        dataset = OMEZarr.open(
            store_path, mode="w-", channel_names=channel_names
        )
        assert os.path.isdir(store_path)
        assert dataset.channel_names == channel_names


@contextmanager
def _temp_ome_zarr(image_5d: NDArray, channel_names: list[str], arr_name):
    try:
        temp_dir = TemporaryDirectory()
        dataset = OMEZarr.open(
            os.path.join(temp_dir.name, "ome.zarr"),
            mode="a",
            channel_names=channel_names,
        )
        dataset[arr_name] = image_5d
        yield dataset
    finally:
        dataset.close()
        temp_dir.cleanup()


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    arr_name=short_alpha_numeric,
)
@settings(max_examples=16, deadline=2000)
def test_write_ome_zarr(channels_and_random_5d, arr_name):
    """Test `iohub.ngff.OMEZarr.__setitem__()`"""
    channel_names, random_5d = channels_and_random_5d
    with _temp_ome_zarr(random_5d, channel_names, arr_name) as dataset:
        assert_array_almost_equal(dataset[arr_name][:], random_5d)
        # round-trip test with the offical reader implementation
        ext_reader = Reader(parse_url(dataset.zgroup.store.path))
        node = list(ext_reader())[0]
        assert node.metadata["name"] == channel_names
        assert node.specs[0].datasets == [arr_name]
        assert node.data[0].shape == random_5d.shape
        assert node.data[0].dtype == random_5d.dtype


@given(channel_names=channel_names_st)
@settings(max_examples=16)
def test_create_hcs(channel_names):
    """Test `iohub.ngff.HCSZarr.open()`"""
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "hcs.zarr")
        dataset = HCSZarr.open(
            store_path, mode="a", channel_names=channel_names
        )
        assert os.path.isdir(store_path)
        assert dataset.channel_names == channel_names


def test_open_hcs_create_empty(caplog):
    """Test `iohub.ngff.HCSZarr.open()`"""
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "hcs.zarr")
        dataset = HCSZarr.open(store_path, mode="a", channel_names=["GFP"])
        assert dataset.zgroup.store.path == store_path
        dataset.close()
        with pytest.raises(FileExistsError):
            _ = HCSZarr.open(store_path, mode="w-", channel_names=["GFP"])
        with pytest.raises(ValueError):
            _ = HCSZarr.open(store_path, mode="x")
        with pytest.raises(FileNotFoundError):
            _ = HCSZarr.open("do-not-exist.zarr", mode="r+")
        # avoid capturing warning in test report with fixture
        with caplog.at_level(logging.INFO):
            dataset = HCSZarr.open(store_path, mode="r+")
            assert "Cannot determine" in caplog.text
            dataset.close()


@contextmanager
def _temp_copy(src: StrPath):
    """Create a temporary copy of data on disk."""
    try:
        temp_dir = TemporaryDirectory()
        yield shutil.copytree(src, temp_dir.name, dirs_exist_ok=True)
    finally:
        temp_dir.cleanup()


def test_modify_hcs_ref(setup_test_data, setup_hcs_ref):
    """Test `iohub.writer.HCSZarr.open()`"""
    with _temp_copy(setup_hcs_ref) as store_path:
        with HCSZarr.open(store_path, mode="r+") as dataset:
            assert dataset.axes[0].name == "c"
            assert dataset.channel_names == ["DAPI"]
            position = dataset["B/03/0"]
            assert position[0].shape == (1, 2, 2160, 5120)
            position.append_channel("GFP", resize_arrays=True)
            assert position.channel_names == ["DAPI", "GFP"]
            assert position[0].shape == (2, 2, 2160, 5120)


@given(row_names=row_names_st, col_names=col_names_st)
@settings(max_examples=32, deadline=2000)
def test_create_well(row_names: list[str], col_names: list[str]):
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "hcs.zarr")
        dataset = HCSZarr.open(store_path, mode="a", channel_names=["GFP"])
        for row_name in row_names:
            for col_name in col_names:
                dataset.create_well(row_name, col_name)
        assert [
            c["name"] for c in dataset.zattrs["plate"]["columns"]
        ] == col_names
        assert [
            r["name"] for r in dataset.zattrs["plate"]["rows"]
        ] == row_names
