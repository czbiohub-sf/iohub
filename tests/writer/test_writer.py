import pytest
import os
from contextlib import contextmanager
from tempfile import TemporaryDirectory
import zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from numpy.testing import assert_array_almost_equal
from hypothesis import given, settings, strategies as st
import hypothesis.extra.numpy as npst
from typing import List
from numpy.typing import NDArray

from iohub.writer import new_zarr, OMEZarrWriter


short_text_st = st.text(min_size=1, max_size=16)
t_dim_st = st.shared(st.integers(1, 4))
c_dim_st = st.shared(st.integers(1, 4))
z_dim_st = st.shared(st.integers(1, 4))
y_dim_st = st.shared(st.integers(1, 32))
x_dim_st = st.shared(st.integers(1, 32))
channel_names_st = c_dim_st.flatmap(
    (
        lambda c_dim: st.lists(
            short_text_st, min_size=c_dim, max_size=c_dim, unique=True
        )
    )
)


@st.composite
def _random_5d_with_channels(draw):
    channel_names = draw(channel_names_st)
    arr_shape = (
        draw(t_dim_st),
        draw(c_dim_st),
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
    random_5d = draw(npst.arrays(dtype, shape=arr_shape))
    return (channel_names, random_5d)


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
def _temp_ome_zarr_writer(image_5d: NDArray, channel_names: List[str]):
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


@given(random_5d=_random_5d_with_channels())
@settings(max_examples=16, deadline=2000)
def test_write_ome_zarr(random_5d):
    """Test `iohub.writer.OMEZarrWriter.write_zstack()`"""
    channel_names, random_5d = random_5d
    with _temp_ome_zarr_writer(random_5d, channel_names) as writer:
        assert_array_almost_equal(writer.root["0"][:], random_5d)
        # round-trip test with the offical reader implementation
        ext_reader = Reader(parse_url(writer.root.store.path))
        node = list(ext_reader())[0]
        assert node.metadata["name"] == channel_names
        assert node.specs[0].datasets == [writer.arr_name]
        assert node.data[0].shape == random_5d.shape
        assert node.data[0].dtype == random_5d.dtype
