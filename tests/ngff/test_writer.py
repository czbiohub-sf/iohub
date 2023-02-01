from __future__ import annotations

import pytest
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
from hypothesis import given, settings, strategies as st
import hypothesis.extra.numpy as npst
from typing import TYPE_CHECKING
from numpy.typing import NDArray

if TYPE_CHECKING:
    from _typeshed import StrPath

from iohub.ngff import new_zarr, OMEZarrWriter, HCSWriter


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
    short_alpha_numeric, min_size=1, max_size=8, unique=True
)
col_names_st = st.lists(
    short_alpha_numeric, min_size=1, max_size=8, unique=True
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


def test_new_zarr():
    """Test `iohub.writer.new_zarr()"""
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "new.zarr")
        root = new_zarr(store_path)
        assert isinstance(root, zarr.Group)
        assert isinstance(root.store, zarr.DirectoryStore)
        assert root.store._dimension_separator == "/"
        assert root.store.path == store_path


def test_new_zarr_same_path():
    """Test `iohub.writer.new_zarr()"""
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "new.zarr")
        os.mkdir(store_path)
        with pytest.raises(FileExistsError):
            _ = new_zarr(store_path)


@given(channel_names=channel_names_st, arr_name=short_text_st)
@settings(max_examples=16)
def test_init_ome_zarr(channel_names, arr_name):
    """Test `iohub.writer.OMEZarrWriter.__call__()`"""
    with TemporaryDirectory() as temp_dir:
        root = new_zarr(os.path.join(temp_dir, "ome.zarr"))
        writer = OMEZarrWriter(root, channel_names, arr_name=arr_name)
        assert writer.channel_names == channel_names
        assert writer.arr_name == arr_name
        assert writer._rel_keys("/") == []


@contextmanager
def _temp_ome_zarr_writer(image_5d: NDArray, channel_names: list[str]):
    try:
        temp_dir = TemporaryDirectory()
        writer = OMEZarrWriter.open(
            os.path.join(temp_dir.name, "ome.zarr"),
            mode="a",
            channel_names=channel_names,
        )
        for t, time_point in enumerate(image_5d):
            for c, zstack in enumerate(time_point):
                writer.write_zstack(zstack, writer.root, t, c)
        yield writer
    finally:
        writer.close()
        temp_dir.cleanup()


@given(channels_and_random_5d=_channels_and_random_5d())
@settings(max_examples=16, deadline=2000)
def test_write_ome_zarr(channels_and_random_5d):
    """Test `iohub.writer.OMEZarrWriter.write_zstack()`"""
    channel_names, random_5d = channels_and_random_5d
    with _temp_ome_zarr_writer(random_5d, channel_names) as writer:
        assert_array_almost_equal(writer.root["0"][:], random_5d)
        # round-trip test with the offical reader implementation
        ext_reader = Reader(parse_url(writer.root.store.path))
        node = list(ext_reader())[0]
        assert node.metadata["name"] == channel_names
        assert node.specs[0].datasets == [writer.arr_name]
        assert node.data[0].shape == random_5d.shape
        assert node.data[0].dtype == random_5d.dtype


@given(channel_names=channel_names_st, arr_name=short_text_st)
@settings(max_examples=16)
def test_init_hcs(channel_names, arr_name):
    with TemporaryDirectory() as temp_dir:
        root = new_zarr(os.path.join(temp_dir, "hcs.zarr"))
        writer = HCSWriter(root, channel_names, arr_name=arr_name)
        assert writer.channel_names == channel_names
        assert writer.arr_name == arr_name
        assert writer._rel_keys("/") == []


def test_open_hcs_create_empty():
    """Test `iohub.writer.HCSWriter.open()`"""
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "hcs.zarr")
        writer = HCSWriter.open(store_path, mode="a", channel_names=["GFP"])
        assert writer.root.store.path == store_path
        writer.close()
        with pytest.raises(FileExistsError):
            _ = HCSWriter.open(store_path, mode="w-", channel_names=["GFP"])
        with pytest.raises(ValueError):
            _ = HCSWriter.open(store_path, mode="a", channel_names=["GFP"])
        with pytest.raises(FileNotFoundError):
            _ = HCSWriter.open(store_path, mode="r+")
        with pytest.raises(FileNotFoundError):
            _ = HCSWriter.open("do-not-exist.zarr", mode="r+")


@contextmanager
def _temp_copy(src: StrPath):
    """Create a temporary copy of data on disk."""
    try:
        temp_dir = TemporaryDirectory()
        yield shutil.copytree(src, temp_dir.name, dirs_exist_ok=True)
    finally:
        temp_dir.cleanup()


def test_modify_hcs_ref(setup_hcs_ref):
    """Test `iohub.writer.HCSWriter.open()`"""
    with _temp_copy(setup_hcs_ref) as store_path:
        writer = HCSWriter.open(store_path, mode="r+")
        assert writer.axes[0].name == "c"
        assert writer.channel_names == ["DAPI"]
        writer.append_channel("GFP")
        assert writer.channel_names == ["DAPI", "GFP"]


@given(row_names=row_names_st)
def test_require_row(row_names: list[str]):
    with TemporaryDirectory() as temp_dir:
        store_path = os.path.join(temp_dir, "hcs.zarr")
        writer = HCSWriter.open(store_path, mode="a", channel_names=["GFP"])
        for row_name in row_names:
            writer.require_row(row_name)
        assert list(writer.rows.keys()) == row_names
