from __future__ import annotations

import os
import platform
import shutil
import string
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import ome_zarr.io
import ome_zarr.reader
import pytest
import zarr.storage
from hypothesis import HealthCheck, assume, given, settings
from ngff_zarr import from_ngff_zarr
from numpy.testing import assert_allclose, assert_array_equal
from numpy.typing import NDArray

if TYPE_CHECKING:
    from _typeshed import StrPath

from iohub.core.compat import get_ome_attrs
from iohub.core.utils import pad_shape
from iohub.ngff.models import TO_DICT_SETTINGS
from iohub.ngff.nodes import (
    Plate,
    Position,
    TransformationMeta,
    _case_insensitive_local_fs,
    _open_store,
    _scale_dims,
    open_ome_zarr,
)
from tests.conftest import hcs_ref

short_text_st = st.text(min_size=1, max_size=16)
t_dim_st = st.integers(1, 4)
c_dim_st = st.integers(1, 4)
z_dim_st = st.integers(1, 4)
y_dim_st = st.integers(1, 32)
x_dim_st = st.integers(1, 32)
channel_names_st = c_dim_st.flatmap(lambda c_dim: st.lists(short_text_st, min_size=c_dim, max_size=c_dim, unique=True))
ngff_versions_st = st.sampled_from(["0.4", "0.5"])
short_alpha_numeric = st.text(
    alphabet=list(string.ascii_lowercase + string.ascii_uppercase + string.digits),
    min_size=1,
    max_size=16,
)
if os.name == "nt":
    # Windows does not allow certain file names
    _INVALID_NT_FILE_NAMES = (
        ["CON", "PRN", "AUX", "NUL"] + ["COM" + str(i) for i in range(10)] + ["LPT" + str(i) for i in range(10)]
    )
    short_alpha_numeric = short_alpha_numeric.filter(lambda x: x not in _INVALID_NT_FILE_NAMES)
tiles_rc_st = st.tuples(t_dim_st, t_dim_st)
plate_axis_names_st = st.lists(
    short_alpha_numeric,
    min_size=1,
    max_size=8,
    unique_by=(lambda x: x.lower()),
)
_dims_st = st.frozensets(
    st.sampled_from(["z", "y", "x"]),
    min_size=1,
    max_size=3,
)


@st.composite
def _pyramid_config(draw):
    """Random TCZYX shape (including odd dims), dims subset, and level count."""
    shape = (
        draw(t_dim_st),
        draw(c_dim_st),
        draw(z_dim_st),
        draw(y_dim_st),
        draw(x_dim_st),
    )
    dims = draw(_dims_st)
    levels = draw(st.integers(2, 4))
    return shape, dims, levels


@st.composite
def _random_array_shape_and_dtype_with_channels(draw, c_dim: int):
    shape = (
        draw(t_dim_st),
        c_dim,
        draw(z_dim_st),
        draw(y_dim_st),
        draw(x_dim_st),
    )
    # zarr-python 3 broke big-endian support:
    # https://github.com/zarr-developers/zarr-python/issues/3005
    dtype = draw(
        st.one_of(
            npst.integer_dtypes(endianness="<"),
            npst.unsigned_integer_dtypes(endianness="<"),
            npst.floating_dtypes(endianness="<"),
            npst.boolean_dtypes(),
        )
    )
    return shape, dtype


@st.composite
def _channels_and_random_5d_shape_and_dtype(draw):
    channel_names = draw(channel_names_st)
    shape, dtype = draw(_random_array_shape_and_dtype_with_channels(c_dim=len(channel_names)))
    return channel_names, shape, dtype


@st.composite
def _channels_and_random_5d(draw):
    channel_names, shape, dtype = draw(_channels_and_random_5d_shape_and_dtype())
    random_5d = draw(npst.arrays(dtype, shape=shape))
    return channel_names, random_5d


@pytest.mark.parametrize(
    ("values", "axes", "expected"),
    [
        # all axes downsampled
        ((4, 8, 16), {0, 1, 2}, (2, 4, 8)),
        # subset of axes
        ((4, 8, 16), {1, 2}, (4, 4, 8)),
        # no axes — identity
        ((4, 8, 16), set(), (4, 8, 16)),
        # odd values round up
        ((3, 7, 5), {0, 1, 2}, (2, 4, 3)),
        # single element
        ((1,), {0}, (1,)),
    ],
)
def test_scale_dims(values, axes, expected):
    """Test `iohub.ngff._scale_dims()`"""
    assert _scale_dims(values, axes) == expected


@given(
    values=st.tuples(*[st.integers(1, 64)] * 5),
    axes=st.frozensets(st.integers(0, 4)),
)
@settings(max_examples=64, deadline=None)
def test_scale_dims_properties(values, axes):
    """Property tests for _scale_dims."""
    import math

    result = _scale_dims(values, axes)
    assert len(result) == len(values)
    for i, (v, r) in enumerate(zip(values, result, strict=False)):
        if i in axes:
            assert r == math.ceil(v / 2)
        else:
            assert r == v


@given(shape=st.lists(x_dim_st, min_size=1, max_size=10), target=x_dim_st)
@settings(max_examples=16, deadline=None)
def testpad_shape(shape, target):
    """Test `iohub.ngff.pad_shape()`"""
    shape = tuple(shape)
    assume(len(shape) <= target)
    new_shape = pad_shape(shape=shape, target=target)
    assert len(new_shape) == target
    assert new_shape[-len(shape) :] == shape


@given(version=ngff_versions_st)
def test_open_store_create(version):
    """Test `iohub.ngff._open_store()"""
    for mode in ("a", "w", "w-"):
        with TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "new.zarr"
            root, _impl = _open_store(store_path, mode=mode, version=version)
            assert isinstance(root, zarr.Group)
            assert isinstance(root.store, zarr.storage.LocalStore)
            # assert root.store._dimension_separator == "/"
            assert root.store.root.resolve() == Path(store_path).resolve()


@given(version=ngff_versions_st)
def test_open_store_create_existing(version):
    """Test `iohub.ngff._open_store()"""
    with TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "new.zarr"
        g = zarr.open_group(store_path, mode="w-")
        g.store.close()
        with pytest.raises(FileExistsError):
            _ = _open_store(store_path, mode="w-", version=version)
        root, _impl = _open_store(store_path, mode="w", version=version)
        assert root is not None


@given(version=ngff_versions_st)
def test_open_store_read_nonexist(version):
    """Test `iohub.ngff._open_store()"""
    for mode in ("r", "r+"):
        with TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "new.zarr"
            with pytest.raises(FileNotFoundError):
                _ = _open_store(store_path, mode=mode, version=version)


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


@given(channel_names=channel_names_st, version=ngff_versions_st)
@settings(max_examples=16)
def test_init_ome_zarr(channel_names, version):
    """Test `iohub.ngff.open_ome_zarr()`"""
    with TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "ome.zarr"
        dataset = open_ome_zarr(
            store_path,
            layout="fov",
            mode="w-",
            channel_names=channel_names,
            version=version,
        )
        assert Path(store_path).is_dir()
        assert dataset.channel_names == channel_names


@pytest.mark.parametrize("mode", ["w", "w-"])
def test_open_ome_zarr_v04_write_succeeds(tmp_path, mode):
    """Creating new v0.4 stores must succeed."""
    store_path = tmp_path / "out.zarr"
    with open_ome_zarr(
        store_path,
        layout="fov",
        mode=mode,
        channel_names=["DAPI"],
        version="0.4",
    ) as ds:
        assert ds.version == "0.4"
        assert (store_path / ".zgroup").exists()
        assert not (store_path / "zarr.json").exists()


def test_open_ome_zarr_v04_append_new_path_succeeds(tmp_path):
    """mode='a' on a nonexistent path should create a v0.4 store."""
    store_path = tmp_path / "nonexistent.zarr"
    with open_ome_zarr(
        store_path,
        layout="fov",
        mode="a",
        channel_names=["DAPI"],
        version="0.4",
    ) as ds:
        assert ds.version == "0.4"
        assert (store_path / ".zgroup").exists()


@pytest.mark.parametrize("version", ["0.5"])
@pytest.mark.parametrize(
    "basename",
    ["some.zarr", "other.zarr/0/0/0", "random_dir", "napari_ome_zarr"],
)
def test_init_ome_zarr_overwrite_non_zarr(tmp_path, basename, version):
    """Test `iohub.ngff.open_ome_zarr()`"""
    store_path = tmp_path / basename
    store_path.mkdir(parents=True)
    some_child_directory = store_path / "some_other_directory"
    some_child_directory.mkdir()
    if ".zarr" not in basename:
        with pytest.raises(ValueError, match=r"."):
            _ = open_ome_zarr(
                store_path,
                layout="fov",
                mode="w",
                channel_names=["channel"],
                version=version,
            )
        assert some_child_directory.exists()
    assert (
        open_ome_zarr(
            store_path,
            layout="fov",
            mode="w",
            channel_names=["channel"],
            disable_path_checking=True,
            version=version,
        )
        is not None
    )
    assert not some_child_directory.exists()


@contextmanager
def _temp_ome_zarr(
    image_5d: NDArray,
    channel_names: list[str],
    arr_name: str,
    version: Literal["0.4", "0.5"],
    **kwargs,
):
    """Helper function to generate a temporary OME-Zarr store.

    Parameters
    ----------
    image_5d : NDArray
    channel_names : list[str]
    arr_name : str
    version : str
    **kwargs : dict
        Additional keyword arguments to pass to `create_image()`.

    Yields
    ------
    Position
    """
    try:
        temp_dir = TemporaryDirectory()
        dataset = open_ome_zarr(
            Path(temp_dir.name) / "ome.zarr",
            layout="fov",
            mode="a",
            version=version,
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
    version: Literal["0.4", "0.5"],
    **kwargs,
):
    """Helper function to generate a temporary OME-Zarr store.

    Parameters
    ----------
    image_5d : NDArray
    channel_names : list[str]
    arr_name : str
    position_list : list[tuple[str, str, str]]
    version : Literal["0.4", "0.5"]

    Yields
    ------
    Position
    """
    try:
        temp_dir = TemporaryDirectory()
        dataset = open_ome_zarr(
            Path(temp_dir.name) / "ome.zarr",
            layout="hcs",
            mode="a",
            channel_names=channel_names,
            version=version,
        )
        for position in position_list:
            pos = dataset.create_position(position[0], position[1], position[2])
            pos.create_image(arr_name, image_5d, **kwargs)
        yield dataset
    finally:
        dataset.close()
        temp_dir.cleanup()


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    arr_name=short_alpha_numeric,
    version=ngff_versions_st,
)
@settings(
    max_examples=16,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_write_ome_zarr(channels_and_random_5d, arr_name, version):
    """Test `iohub.ngff.Position.__setitem__()`"""
    channel_names, random_5d = channels_and_random_5d
    with _temp_ome_zarr(random_5d, channel_names, arr_name, version=version) as dataset:
        assert_allclose(dataset[arr_name][:], random_5d)
        if version == "0.5":
            # round-trip test with the official reader implementation
            # ome-zarr-py reader requires zarr-python Group with .store.root
            ext_reader = ome_zarr.reader.Reader(ome_zarr.io.parse_url(dataset.zgroup.store.root))
            node = next(iter(ext_reader()))
            assert node.metadata["channel_names"] == channel_names
            assert node.specs[0].datasets == [arr_name]
            assert_allclose(node.data[0], random_5d)


@given(
    ch_shape_dtype=_channels_and_random_5d_shape_and_dtype(),
    arr_name=short_alpha_numeric,
    version=ngff_versions_st,
)
@settings(
    max_examples=16,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_create_zeros(ch_shape_dtype, arr_name, version):
    """Test `iohub.ngff.Position.create_zeros()`"""
    channel_names, shape, dtype = ch_shape_dtype
    with TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "ome.zarr"
        dataset = open_ome_zarr(
            store_path,
            layout="fov",
            mode="w-",
            channel_names=channel_names,
            version=version,
        )
        dataset.create_zeros(name=arr_name, shape=shape, dtype=dtype)
        if version == "0.5":
            assert (Path(store_path) / arr_name / "zarr.json").exists()
        else:
            assert (Path(store_path) / arr_name / ".zarray").exists()
        if version == "0.5":
            assert dataset[arr_name].metadata.dimension_names == (
                "T",
                "C",
                "Z",
                "Y",
                "X",
            )
        assert not dataset[arr_name][:].any()
        assert dataset[arr_name].shape == shape
        assert dataset[arr_name].dtype == dtype


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    arr_name=short_alpha_numeric,
    version=ngff_versions_st,
)
@settings(
    max_examples=16,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_ome_zarr_to_dask(channels_and_random_5d, arr_name, version):
    """Test `iohub.ngff.Position.data` to dask"""
    channel_names, random_5d = channels_and_random_5d
    with _temp_ome_zarr(random_5d, channel_names, "0", version=version) as dataset:
        assert_allclose(dataset.data.dask_array().compute(), random_5d)
    with _temp_ome_zarr(random_5d, channel_names, arr_name, version=version) as dataset:
        assert_allclose(dataset[arr_name].dask_array().compute(), random_5d)


@given(channels_and_random_5d=_channels_and_random_5d())
@settings(
    max_examples=16,
    deadline=None,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_writing_sharded(channels_and_random_5d):
    """Test `iohub.ngff.Position.data`"""
    channel_names, random_5d = channels_and_random_5d
    chunks = (
        1,
        max(1, random_5d.shape[1] // 3),
        max(1, random_5d.shape[2] // 4),
        max(1, random_5d.shape[3] // 5),
        max(1, random_5d.shape[4] // 6),
    )
    shards_ratio = (3, 4, 5, 6, 7)
    with _temp_ome_zarr(
        random_5d,
        channel_names,
        arr_name="0",
        version="0.5",
        chunks=chunks,
        shards_ratio=shards_ratio,
    ) as dataset:
        assert_array_equal(dataset["0"].numpy(), random_5d)
        assert dataset["0"].chunks == chunks
        assert dataset["0"].shards == tuple(c * s for c, s in zip(chunks, shards_ratio, strict=False))


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    arr_name=short_alpha_numeric,
    version=ngff_versions_st,
)
@settings(
    max_examples=16,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_position_data(channels_and_random_5d, arr_name, version):
    """Test `iohub.ngff.Position.data`"""
    channel_names, random_5d = channels_and_random_5d
    assume(arr_name != "0")
    with _temp_ome_zarr(random_5d, channel_names, "0", version=version) as dataset:
        assert_allclose(dataset.data.numpy(), random_5d)
    with pytest.raises(KeyError):
        with _temp_ome_zarr(random_5d, channel_names, arr_name, version=version) as dataset:
            _ = dataset.data


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    arr_name=short_alpha_numeric,
    version=ngff_versions_st,
    concurrency=st.one_of(st.just(None), st.integers(1, 2)),
)
@settings(
    max_examples=16,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_ome_zarr_to_tensorstore(channels_and_random_5d, arr_name, version, concurrency):
    """Test round-trip write/read via TensorStore implementation."""
    pytest.importorskip("tensorstore")
    from pathlib import Path

    from iohub.core.config import TensorStoreConfig

    channel_names, random_5d = channels_and_random_5d
    zeros = np.zeros_like(random_5d)
    ts_config = TensorStoreConfig(data_copy_concurrency=concurrency) if concurrency else None

    with _temp_ome_zarr(random_5d, channel_names, arr_name, version=version) as dataset:
        store_path = Path(dataset.zgroup.store.root)

        with open_ome_zarr(
            store_path,
            layout="fov",
            mode="r+",
            implementation="tensorstore",
            implementation_config=ts_config,
        ) as ts_dataset:
            ts_arr = ts_dataset[arr_name]
            assert_array_equal(ts_arr.numpy(), random_5d)
            ts_arr[...] = zeros

        with open_ome_zarr(store_path, layout="fov", mode="r") as read_only:
            assert_array_equal(read_only[arr_name].numpy(), zeros)


@pytest.mark.parametrize("recheck_cached_data", [None, "open", True, False])
def test_tensorstore_recheck_cached_data(monkeypatch, recheck_cached_data):
    """``TensorStoreConfig.recheck_cached_data`` propagates into ``ts.open``.

    When the option is ``None`` (default) the kwarg must not be forwarded,
    so the TensorStore driver falls back to its own default. For any other
    value (``"open"``, ``True``, ``False``) the exact value must reach the
    ``ts.open`` call — this is the knob used to suppress per-chunk
    revalidation on networked filesystems during read-heavy training.
    """
    pytest.importorskip("tensorstore")
    import tensorstore as ts

    from iohub.core.config import TensorStoreConfig
    from iohub.core.implementations import tensorstore as ts_impl

    captured_kwargs: list[dict] = []
    real_ts_open = ts_impl._ts_open

    def spy_ts_open(spec, **kwargs):
        captured_kwargs.append(kwargs)
        return real_ts_open(spec, **kwargs)

    monkeypatch.setattr(ts_impl, "_ts_open", spy_ts_open)

    channel_names = ["DAPI"]
    random_5d = np.random.default_rng(0).random((1, 1, 1, 4, 4), dtype=np.float32)
    ts_config = TensorStoreConfig(recheck_cached_data=recheck_cached_data)

    with _temp_ome_zarr(random_5d, channel_names, arr_name="0", version="0.5") as dataset:
        store_path = Path(dataset.zgroup.store.root)
        with open_ome_zarr(
            store_path,
            layout="fov",
            mode="r",
            implementation="tensorstore",
            implementation_config=ts_config,
        ) as ts_dataset:
            handle = ts_dataset["0"].native
            assert isinstance(handle, ts.TensorStore)
            assert_array_equal(np.asarray(handle.read().result()), random_5d)

    open_calls = [k for k in captured_kwargs if k.get("open") is True]
    assert open_calls, "Expected at least one ts.open(open=True) call"
    last = open_calls[-1]
    if recheck_cached_data is None:
        assert "recheck_cached_data" not in last
    else:
        assert last.get("recheck_cached_data") == recheck_cached_data


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    arr_name=short_alpha_numeric,
    version=ngff_versions_st,
)
@settings(
    max_examples=16,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_append_channel(channels_and_random_5d, arr_name, version):
    """Test `iohub.ngff.Position.append_channel()`"""
    channel_names, random_5d = channels_and_random_5d
    assume(len(channel_names) > 1)
    with _temp_ome_zarr(random_5d[:, :-1], channel_names[:-1], arr_name, version=version) as dataset:
        dataset.append_channel(channel_names[-1], resize_arrays=True)
        dataset[arr_name][:, -1] = random_5d[:, -1]
        assert_allclose(dataset[arr_name][:], random_5d)


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    arr_name=short_alpha_numeric,
    version=ngff_versions_st,
    new_channel=short_text_st,
)
@settings(
    max_examples=16,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_rename_channel(channels_and_random_5d, arr_name, new_channel, version):
    """Test `iohub.ngff.Position.rename_channel()`"""
    channel_names, random_5d = channels_and_random_5d
    assume(new_channel not in channel_names)
    with _temp_ome_zarr(random_5d, channel_names, arr_name, version=version) as dataset:
        dataset.rename_channel(old=channel_names[0], new=new_channel)
        assert new_channel in dataset.channel_names
        assert dataset.metadata.omero.channels[0].label == new_channel


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    arr_name=short_alpha_numeric,
    version=ngff_versions_st,
)
@settings(deadline=None)
def test_rename_well(channels_and_random_5d, arr_name, version):
    """Test `iohub.ngff.Position.rename_well()`"""
    channel_names, random_5d = channels_and_random_5d

    position_list = [("A", "1", "0"), ("C", "4", "0")]
    with _temp_ome_zarr_plate(random_5d, channel_names, arr_name, position_list, version) as dataset:
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
        with pytest.raises(FileExistsError):
            dataset.rename_well("B/2", "C/4")

        # source doesn't exist
        with pytest.raises(FileNotFoundError):
            dataset.rename_well("Q/1", "Q/2")

        # invalid well names
        with pytest.raises(ValueError, match=r"."):
            dataset.rename_well("B/2", " A/1")
        with pytest.raises(ValueError, match=r"."):
            dataset.rename_well("B/2", "A/?")


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    arr_name=short_alpha_numeric,
    version=ngff_versions_st,
)
@settings(
    max_examples=16,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_update_channel(channels_and_random_5d, arr_name, version):
    """Test `iohub.ngff.Position.update_channel()`"""
    channel_names, random_5d = channels_and_random_5d
    assume(len(channel_names) > 1)
    with _temp_ome_zarr(random_5d[:, :-1], channel_names[:-1], arr_name, version=version) as dataset:
        for i, ch in enumerate(dataset.channel_names):
            dataset.update_channel(chan_name=ch, target=arr_name, data=random_5d[:, -1])
            assert_allclose(dataset[arr_name][:, i], random_5d[:, -1])


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    arr_name=short_alpha_numeric,
    version=ngff_versions_st,
)
@settings(
    max_examples=16,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_write_more_channels(channels_and_random_5d, arr_name, version):
    """Test `iohub.ngff.Position.create_image()`"""
    channel_names, random_5d = channels_and_random_5d
    assume(len(channel_names) > 1)
    with pytest.raises(ValueError, match=r"."):
        with _temp_ome_zarr(random_5d, channel_names[:-1], arr_name, version=version) as _:
            pass


@pytest.mark.parametrize("implementation", ["zarr", "tensorstore"])
@given(
    ch_shape_dtype=_channels_and_random_5d_shape_and_dtype(),
    arr_name=short_alpha_numeric,
)
def test_set_transform_image(implementation, ch_shape_dtype, arr_name):
    if implementation == "tensorstore":
        pytest.importorskip("tensorstore")
    """Test `iohub.ngff.Position.set_transform()`"""
    channel_names, shape, dtype = ch_shape_dtype
    transform = [TransformationMeta(type="translation", translation=(1, 2, 3, 4, 5))] * len(channel_names)
    with TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "ome.zarr"
        with open_ome_zarr(
            store_path, layout="fov", mode="w-", channel_names=channel_names, implementation=implementation
        ) as dataset:
            dataset.create_zeros(name=arr_name, shape=shape, dtype=dtype)
            assert dataset.metadata.multiscales[0].datasets[0].coordinate_transformations == [
                TransformationMeta(type="scale", scale=(1.0, 1.0, 1.0, 1.0, 1.0))
            ]
            dataset.set_transform(image=arr_name, transform=transform)
            assert dataset.metadata.multiscales[0].datasets[0].coordinate_transformations == transform
        # read data with an external reader
        ext_reader = ome_zarr.reader.Reader(ome_zarr.io.parse_url(dataset.zgroup.store.root))
        node = next(iter(ext_reader()))
        assert node.metadata["coordinateTransformations"][0] == [
            translate.model_dump(**TO_DICT_SETTINGS) for translate in transform
        ]


input_transformations = [
    ([TransformationMeta(type="identity")], []),
    ([TransformationMeta(type="scale", scale=(1.0, 2.0, 3.0, 4.0, 5.0))], []),
    (
        [TransformationMeta(type="translation", translation=(1.0, 2.0, 3.0, 4.0, 5.0))],
        [],
    ),
    (
        [
            TransformationMeta(type="scale", scale=(2.0, 2.0, 2.0, 2.0, 2.0)),
            TransformationMeta(type="translation", translation=(1.0, 1.0, 1.0, 1.0, 1.0)),
        ],
        [
            TransformationMeta(type="scale", scale=(2.0, 2.0, 2.0, 2.0, 2.0)),
            TransformationMeta(type="translation", translation=(1.0, 1.0, 1.0, 1.0, 1.0)),
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
    [(saved, target) for saved, target in zip(input_transformations, target_scales, strict=False)],
)
@given(
    ch_shape_dtype=_channels_and_random_5d_shape_and_dtype(),
    arr_name=short_alpha_numeric,
    version=ngff_versions_st,
)
def test_get_effective_scale_image(transforms, ch_shape_dtype, arr_name, version):
    """Test `iohub.ngff.Position.get_effective_scale()`"""
    (fov_transform, img_transform), expected_scale = transforms
    channel_names, shape, dtype = ch_shape_dtype
    with TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "ome.zarr"
        with open_ome_zarr(
            store_path,
            layout="fov",
            mode="w-",
            channel_names=channel_names,
            version=version,
        ) as dataset:
            dataset.create_zeros(name=arr_name, shape=shape, dtype=dtype)
            dataset.set_transform(image="*", transform=fov_transform)
            dataset.set_transform(image=arr_name, transform=img_transform)
            scale = dataset.get_effective_scale(image=arr_name)
            assert scale == expected_scale


@pytest.mark.parametrize(
    "transforms",
    [(saved, target) for saved, target in zip(input_transformations, target_translations, strict=False)],
)
@given(
    ch_shape_dtype=_channels_and_random_5d_shape_and_dtype(),
    arr_name=short_alpha_numeric,
    version=ngff_versions_st,
)
def test_get_effective_translation_image(transforms, ch_shape_dtype, arr_name, version):
    """Test `iohub.ngff.Position.get_effective_translation()`"""
    (fov_transform, img_transform), expected_translation = transforms
    channel_names, shape, dtype = ch_shape_dtype
    with TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "ome.zarr"
        with open_ome_zarr(
            store_path,
            layout="fov",
            mode="w-",
            channel_names=channel_names,
            version=version,
        ) as dataset:
            dataset.create_zeros(name=arr_name, shape=shape, dtype=dtype)
            dataset.set_transform(image="*", transform=fov_transform)
            dataset.set_transform(image=arr_name, transform=img_transform)
            translation = dataset.get_effective_translation(image=arr_name)
            assert translation == expected_translation


@given(
    ch_shape_dtype=_channels_and_random_5d_shape_and_dtype(),
    arr_name=short_alpha_numeric,
    version=ngff_versions_st,
)
def test_set_transform_fov(ch_shape_dtype, arr_name, version):
    """Test `iohub.ngff.Position.set_transform()`"""
    channel_names, shape, dtype = ch_shape_dtype
    transform = [TransformationMeta(type="translation", translation=(1, 2, 3, 4, 5))] * len(channel_names)
    with TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "ome.zarr"
        with open_ome_zarr(
            store_path,
            layout="fov",
            mode="w-",
            channel_names=channel_names,
            version=version,
        ) as dataset:
            dataset.create_zeros(name=arr_name, shape=shape, dtype=dtype)
            assert dataset.metadata.multiscales[0].coordinate_transformations is None
            dataset.set_transform(image="*", transform=transform)
            assert dataset.metadata.multiscales[0].coordinate_transformations == transform
        # read data with plain zarr
        group = zarr.open(store_path)
        maybe_ome = get_ome_attrs(group.attrs)
        assert maybe_ome["multiscales"][0]["coordinateTransformations"] == [
            translate.model_dump(**TO_DICT_SETTINGS) for translate in transform
        ]


@pytest.mark.parametrize("version", ["0.5"])
@pytest.mark.parametrize("image_name", ["0", "1", "a", "*"])
def test_set_scale(image_name, version):
    """Test `iohub.ngff.Position.set_scale()`"""
    translation = [float(t) for t in range(1, 6)]
    scale = [float(s) for s in range(5, 0, -1)]
    array_name = "0" if image_name == "*" else image_name
    new_scale = 10.0
    with TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "ome.zarr"
        with open_ome_zarr(
            store_path,
            layout="fov",
            mode="w-",
            channel_names=["a", "b"],
            version=version,
        ) as dataset:
            dataset.create_zeros(
                name=array_name,
                shape=(1, 2, 4, 8, 16),
                dtype=int,
                transform=[
                    TransformationMeta(type="translation", translation=translation),
                    TransformationMeta(type="scale", scale=scale),
                ],
            )
            with pytest.raises(ValueError, match=r"."):
                dataset.set_scale(image=image_name, axis_name="z", new_scale=-1.0)
            with pytest.raises(KeyError):
                dataset.set_scale(image="nonexistent", axis_name="z", new_scale=9.0)
            assert dataset.scale[-3] == 3.0
            dataset.set_scale(image=image_name, axis_name="z", new_scale=new_scale)
            if image_name == "*":
                assert dataset.scale[-3] == new_scale * 3.0
            else:
                assert dataset.scale[-3] == new_scale
            assert dataset.get_effective_translation(array_name) == translation
            for tf in dataset.zattrs["iohub"]["previous_transforms"][0]["transforms"]:
                if tf["type"] == "scale":
                    assert tf["scale"] == scale


@given(channel_names=channel_names_st, version=ngff_versions_st)
@settings(max_examples=16)
def test_set_contrast_limits(channel_names, version):
    """Test `iohub.ngff.Position.set_contrast_limits()`"""
    with TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "ome.zarr"
        dataset = open_ome_zarr(
            store_path,
            layout="fov",
            mode="a",
            channel_names=channel_names,
            version=version,
        )
        # Create a simple small array - exact shape/dtype doesn't matter
        dataset.create_zeros("data", shape=(1, len(channel_names), 1, 4, 4), dtype=float)

        # Store the initial window settings for all channels
        initial_windows = {}
        for ch_name in channel_names:
            ch_idx = dataset.get_channel_index(ch_name)
            initial_windows[ch_name] = dataset.metadata.omero.channels[ch_idx].window

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
            assert dataset.metadata.omero.channels[ch_idx].window == initial_windows[ch_name]


@given(channel_names=channel_names_st, version=ngff_versions_st)
@settings(max_examples=16)
def test_create_tiled(channel_names, version):
    """Test that `iohub.ngff.open_ome_zarr()` can create
    an empty OME-Zarr store with 'tiled' layout."""
    with TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "tiled.zarr"
        dataset = open_ome_zarr(
            store_path,
            layout="tiled",
            mode="a",
            channel_names=channel_names,
            version=version,
        )
        assert Path(store_path).is_dir()
        assert dataset.channel_names == channel_names


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    grid_shape=tiles_rc_st,
    arr_name=short_alpha_numeric,
)
@pytest.mark.parametrize("implementation", ["zarr", "tensorstore"])
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_make_tiles(implementation, channels_and_random_5d, grid_shape, arr_name):
    if implementation == "tensorstore":
        pytest.importorskip("tensorstore")
    """Test `iohub.ngff.TiledPosition.make_tiles()` and  `...get_tile()`"""
    with TemporaryDirectory() as temp_dir:
        channel_names, random_5d = channels_and_random_5d
        store_path = Path(temp_dir) / "tiled.zarr"
        with open_ome_zarr(
            store_path, layout="tiled", mode="a", channel_names=channel_names, implementation=implementation
        ) as dataset:
            tiles = dataset.make_tiles(
                name=arr_name,
                grid_shape=(int(grid_shape[0]), int(grid_shape[1])),
                tile_shape=tuple(int(i) for i in random_5d.shape),
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
            assert tiles.tile_shape == pad_shape(random_5d.shape[-2:], target=5)
            assert tiles.dtype == random_5d.dtype
            for args in ((1.01, 1), (0, 0, 0)):
                with pytest.raises(TypeError):
                    tiles.get_tile(*args)
            for args in ((0, 0, (0,) * 2), (0, 0, (0,) * 4)):
                with pytest.raises(IndexError):
                    tiles.get_tile(*args)


@pytest.mark.parametrize("implementation", ["zarr", "tensorstore"])
@given(
    channels_and_random_5d=_channels_and_random_5d(),
    grid_shape=tiles_rc_st,
    arr_name=short_alpha_numeric,
    version=ngff_versions_st,
)
@settings(
    max_examples=16,
    deadline=None,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_write_read_tiles(implementation, channels_and_random_5d, grid_shape, arr_name, version):
    if implementation == "tensorstore":
        pytest.importorskip("tensorstore")
    """Test `iohub.ngff.TiledPosition.write_tile()` and `...get_tile()`"""
    channel_names, random_5d = channels_and_random_5d

    def _tile_data(tiles):
        for row in range(tiles.rows):
            for column in range(tiles.columns):
                yield (
                    (random_5d / (tiles.rows * tiles.columns + 1) * (row * column + 1)).astype(random_5d.dtype),
                    row,
                    column,
                )

    with TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "tiled.zarr"
        with open_ome_zarr(
            store_path,
            layout="tiled",
            mode="w-",
            channel_names=channel_names,
            version=version,
            implementation=implementation,
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
            store_path, layout="tiled", mode="r", channel_names=channel_names, implementation=implementation
        ) as dataset:
            for data, row, column in _tile_data(tiles):
                read = tiles.get_tile(row, column)
                assert_allclose(data, read)


@pytest.mark.parametrize("implementation", ["zarr", "tensorstore"])
@given(channel_names=channel_names_st)
@settings(max_examples=16)
def test_create_hcs(implementation, channel_names):
    if implementation == "tensorstore":
        pytest.importorskip("tensorstore")
    """Test `iohub.ngff.open_ome_zarr()`"""
    with TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "hcs.zarr"
        dataset = open_ome_zarr(
            store_path, layout="hcs", mode="a", channel_names=channel_names, implementation=implementation
        )
        assert Path(store_path).is_dir()
        assert dataset.channel_names == channel_names


@pytest.mark.parametrize("implementation", ["zarr", "tensorstore"])
@pytest.mark.parametrize("version", ["0.5"])
def test_open_hcs_create_empty(implementation, version):
    if implementation == "tensorstore":
        pytest.importorskip("tensorstore")
    """Test `iohub.ngff.open_ome_zarr()`"""
    with TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "hcs.zarr"
        dataset = open_ome_zarr(
            store_path,
            layout="hcs",
            mode="a",
            channel_names=["GFP"],
            version=version,
            implementation=implementation,
        )
        assert dataset.zgroup.store.root.resolve() == store_path.resolve()
        dataset.close()
        with pytest.raises(FileExistsError):
            _ = open_ome_zarr(store_path, layout="hcs", mode="w-", channel_names=["GFP"])
        with pytest.raises(ValueError, match=r"."):
            _ = open_ome_zarr(store_path, layout="hcs", mode="x")
        with pytest.raises(FileNotFoundError):
            _ = open_ome_zarr("do-not-exist.zarr", layout="hcs", mode="r+")
        with pytest.raises(ValueError, match=r"."):
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
        with pytest.raises(ValueError, match=r"."):
            _ = dataset.get_channel_index(wrong_channel_name)


def test_get_axis_index():
    with open_ome_zarr(hcs_ref, layout="hcs", mode="r+") as dataset:
        position = dataset["B/03/0"]

        assert position.axis_names == ["c", "z", "y", "x"]

        assert position.get_axis_index("z") == 1
        assert position.get_axis_index("Z") == 1

        with pytest.raises(ValueError, match=r"."):
            _ = position.get_axis_index("t")

        with pytest.raises(ValueError, match=r"."):
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
                assert any("Key 'b' matched" in r.message for r in caplog.records)


@given(row=short_alpha_numeric, col=short_alpha_numeric, pos=short_alpha_numeric)
@settings(max_examples=16)
def test_modify_hcs_ref(row: str, col: str, pos: str):
    """Test `iohub.ngff.open_ome_zarr()`"""
    assume(row.lower() != "b")
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
@settings(max_examples=16)
def test_create_well(row_names: list[str], col_names: list[str]):
    """Test `iohub.ngff.Plate.create_well()`"""
    with TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "hcs.zarr"
        dataset = open_ome_zarr(store_path, layout="hcs", mode="a", channel_names=["GFP"])
        for row_name in row_names:
            for col_name in col_names:
                dataset.create_well(row_name, col_name)
        plate_meta = get_ome_attrs(dataset.zattrs)["plate"]
        assert [c["name"] for c in plate_meta["columns"]] == col_names
        assert [r["name"] for r in plate_meta["rows"]] == row_names


def test_create_case_sensitive_well(tmp_path):
    """Test `iohub.ngff.Plate.create_well()` with case-sensitive names."""
    store_path = tmp_path / "hcs.zarr"
    with open_ome_zarr(store_path, layout="hcs", mode="w-", channel_names=["1", "2"]) as dataset:
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
    row=short_alpha_numeric,
    col=short_alpha_numeric,
    pos=short_alpha_numeric,
    version=ngff_versions_st,
)
def test_create_position(row, col, pos, version):
    """Test `iohub.ngff.Plate.create_position()`"""
    with TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "hcs.zarr"
        dataset = open_ome_zarr(
            store_path,
            layout="hcs",
            mode="a",
            channel_names=["GFP"],
            version=version,
        )
        _ = dataset.create_position(row_name=row, col_name=col, pos_name=pos)
        ome = get_ome_attrs(dataset.zgroup.attrs)
        assert [c["name"] for c in ome["plate"]["columns"]] == [col]
        assert [r["name"] for r in ome["plate"]["rows"]] == [row]
        assert (store_path / row / col / pos).is_dir()
        assert dataset[row][col].metadata.images[0].path == pos


@pytest.mark.parametrize("version", ["0.5"])
def test_create_positions(tmp_path, version):
    positions = [
        x.split("/")
        for x in [
            "A/1/002026",
            "A/1/002027",
            "A/1/002028",
            "A/1/002029",
            "A/1/002030",
            "A/1/002031",
            "A/1/002032",
            "A/1/003021",
            "A/1/003022",
            "A/2/049031",
            "A/2/049032",
            "A/2/049033",
            "A/2/049034",
            "A/2/049035",
            "A/2/049036",
            "A/2/049037",
            "A/2/049038",
        ]
    ]
    single = open_ome_zarr(
        tmp_path / "single.zarr",
        layout="hcs",
        mode="a",
        channel_names=["GFP"],
        version=version,
    )
    batched = open_ome_zarr(
        tmp_path / "batched.zarr",
        layout="hcs",
        mode="a",
        channel_names=["GFP"],
        version=version,
    )
    for pos in positions:
        single.create_position(*pos)
    batched.create_positions(positions)

    # Collect positions and compare those

    get_metadata = lambda x: get_ome_attrs(x.zgroup.attrs)

    single_plate_metadata = get_metadata(single)
    batched_plate_metadata = get_metadata(batched)

    assert single_plate_metadata == batched_plate_metadata

    single_well_metadata = {k: get_metadata(v) for k, v in single.wells()}
    batched_well_metadata = {k: get_metadata(v) for k, v in batched.wells()}

    assert single_well_metadata == batched_well_metadata


@pytest.mark.parametrize("version", ["0.5"])
def test_create_positions_with_tuple_variations(tmp_path, version):
    """Test create_positions with various tuple lengths (3-6 elements).

    This tests the examples from the create_positions docstring, verifying that
    calling create_positions is equivalent to calling create_position multiple
    times with the same arguments.
    """
    # Mix of 3, 5, and 6 element tuples
    positions = [
        # 3-element tuples: automatic row/column indexing
        ("A", "1", "0"),
        ("A", "1", "1"),
        ("A", "2", "0"),
        # 5-element tuples: explicit row/column indices
        ("B", "3", "0", 1, 2),  # row_index=1, col_index=2
        ("B", "3", "1", 1, 2),  # same well indices
        # 6-element tuples: explicit indices with acquisition
        ("C", "4", "0", 0, 0, 0),  # acquisition 0
        ("C", "4", "1", 0, 0, 1),  # acquisition 1
        ("C", "5", "0", 0, 1, 0),  # different well, acquisition 0
    ]

    single = open_ome_zarr(
        tmp_path / "single.zarr",
        layout="hcs",
        mode="a",
        channel_names=["GFP"],
        version=version,
    )
    batched = open_ome_zarr(
        tmp_path / "batched.zarr",
        layout="hcs",
        mode="a",
        channel_names=["GFP"],
        version=version,
    )

    # Create positions individually
    for pos_spec in positions:
        single.create_position(*pos_spec)

    # Create positions in batch
    batched.create_positions(positions)

    # Verify metadata matches
    get_metadata = lambda x: get_ome_attrs(x.zgroup.attrs)

    single_meta = get_metadata(single)
    batched_meta = get_metadata(batched)
    assert single_meta == batched_meta

    # Verify well metadata matches
    single_well_meta = {k: get_metadata(v) for k, v in single.wells()}
    batched_well_meta = {k: get_metadata(v) for k, v in batched.wells()}
    assert single_well_meta == batched_well_meta

    # Verify acquisition indices were set correctly
    for plate in [single, batched]:
        well_c4 = plate["C/4"]
        well_c5 = plate["C/5"]
        assert len(well_c4.metadata.images) == 2  # Two positions in this well
        assert well_c4.metadata.images[0].acquisition == 0
        assert well_c4.metadata.images[1].acquisition == 1
        assert len(well_c5.metadata.images) == 1
        assert well_c5.metadata.images[0].acquisition == 0


@given(channels_and_random_5d=_channels_and_random_5d(), version=ngff_versions_st)
def test_position_scale(channels_and_random_5d, version):
    """Test `iohub.ngff.Position.scale`"""
    channel_names, random_5d = channels_and_random_5d
    scale = list(range(1, 6))
    transform = [TransformationMeta(type="scale", scale=scale)]
    with _temp_ome_zarr(random_5d, channel_names, "0", transform=transform, version=version) as dataset:
        assert dataset.scale == scale


@pytest.mark.skip(reason="https://github.com/zarr-developers/zarr-python/issues/2407")
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
        store_path = Path(temp_dir) / "combined.zarr"
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
    with open_ome_zarr(store_path, layout="hcs", mode="a", channel_names=["A", "B"]) as dataset:
        for name in fov_name_parts:
            fov = dataset.create_position(*name)
            fov.create_zeros("0", shape=(1, 2, 3, y_size, x_size), dtype=int)
        n_rows = len(dataset.metadata.rows)
        n_cols = len(dataset.metadata.columns)
    plate = next(iter(ome_zarr.reader.Reader(ome_zarr.io.parse_url(store_path))()))
    assert plate.data[0].shape == (1, 2, 3, y_size * n_rows, x_size * n_cols)
    assert plate.data[0].dtype == int
    assert not plate.data[0].any()
    assert plate.metadata["channel_names"] == ["A", "B"]


def test_read_empty_hcs_v05(empty_ome_zarr_hcs_v05):
    """Test reading an empty OME-Zarr v0.5 HCS store."""
    empty_zarr, (rows, cols, fovs, resolutions) = empty_ome_zarr_hcs_v05
    with open_ome_zarr(empty_zarr, layout="hcs", mode="r") as dataset:
        for row, col, fov in product(rows, cols, fovs):
            position: Position = dataset[f"{row}/{col}/{fov}"]
            assert position.version == "0.5"
            for resolution in resolutions:
                assert_array_equal(
                    position[resolution].numpy(),
                    np.zeros((50, 48, 64), dtype=np.uint16),
                )
        assert len(list(dataset.positions())) == len(rows) * len(cols) * len(fovs)


def test_acquire_zarr_ome_zarr_05(aqz_ome_zarr_05):
    """Test that `iohub.ngff.open_ome_zarr()` can read OME-Zarr 0.5."""
    pytest.importorskip("acquire_zarr")
    with open_ome_zarr(aqz_ome_zarr_05, layout="fov", mode="r", version="0.5") as dataset:
        assert dataset.version == "0.5"
        assert dataset.data.shape == (32, 4, 10, 48, 64)
        assert dataset.data.chunks == (16, 1, 10, 16, 16)
        assert dataset.data.shards == (16, 1, 10, 48, 32)
        assert "ome" in dataset.zattrs
        assert "multiscales" in dataset.zattrs["ome"]
        assert len(dataset.zattrs["ome"]["multiscales"]) == 1

        multiscale = dataset.zattrs["ome"]["multiscales"][0]
        assert len(multiscale["datasets"]) == 3
        assert multiscale["datasets"][0]["coordinateTransformations"][0]["scale"] == [1.0, 1.0, 1.0, 1.0, 1.0]
        assert multiscale["datasets"][1]["coordinateTransformations"][0]["scale"] == [1.0, 1.0, 1.0, 2.0, 2.0]
        assert multiscale["datasets"][2]["coordinateTransformations"][0]["scale"] == [1.0, 1.0, 1.0, 4.0, 4.0]
        assert 1 < dataset["0"].numpy().mean() < np.iinfo(np.uint16).max


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    arr_name=short_alpha_numeric,
    version=ngff_versions_st,
)
@settings(
    max_examples=16,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_ngff_zarr_read(channels_and_random_5d, arr_name, version):
    """Test that image written with iohub can be read with ngff-zarr."""
    channel_names, random_5d = channels_and_random_5d
    with _temp_ome_zarr(random_5d, channel_names, arr_name=arr_name, version=version) as dataset:
        nz_multiscales = from_ngff_zarr(dataset.zgroup.store.root, validate=False)
        assert_allclose(
            dataset[arr_name].dask_array().compute(),
            nz_multiscales.images[0].data,
        )


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    label_name=short_alpha_numeric,
    version=ngff_versions_st,
)
@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_create_labels(channels_and_random_5d, label_name, version):
    """Test `iohub.ngff.Position.create_label()`"""
    channel_names, random_5d = channels_and_random_5d
    # Create TZYX label data (no channel dimension)
    label_shape = (
        random_5d.shape[0],
        random_5d.shape[2],
        random_5d.shape[3],
        random_5d.shape[4],
    )
    label_data = np.random.default_rng().integers(0, 3, size=label_shape, dtype=np.uint16)

    with _temp_ome_zarr(random_5d, channel_names, "0", version=version) as dataset:
        # Test label creation
        label_image = dataset.create_label(
            name=label_name,
            data=label_data,
            pyramid_levels=1,
        )

        # Verify creation
        assert dataset.has_labels
        assert label_name in list(dataset.labels_group.group_keys())
        assert label_image.array_keys() == ["0"]
        assert_array_equal(label_image.data.numpy(), label_data)
        assert label_image.data.dtype == label_data.dtype

        # Verify TZYX format constraint
        assert len(label_data.shape) == 4  # TZYX
        assert len(random_5d.shape) == 5  # TCZYX


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    label_name=short_alpha_numeric,
    version=ngff_versions_st,
)
@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_labels_pyramid(channels_and_random_5d, label_name, version):
    """Test `iohub.ngff.Position.create_label()` multiscale pyramids"""
    channel_names, random_5d = channels_and_random_5d
    # Create TZYX label data
    label_shape = (
        random_5d.shape[0],
        random_5d.shape[2],
        random_5d.shape[3],
        random_5d.shape[4],
    )
    label_data = np.random.default_rng().integers(0, 3, size=label_shape, dtype=np.uint16)

    with _temp_ome_zarr(random_5d, channel_names, "0", version=version) as dataset:
        # Initialize image pyramid
        dataset.initialize_pyramid(3)

        # Create label with matching pyramid
        label_image = dataset.create_label(
            name=label_name,
            data=label_data,
            pyramid_levels=3,
        )

        # Verify pyramid structure matches image
        assert label_image.array_keys() == ["0", "1", "2"]
        assert dataset.array_keys() == ["0", "1", "2"]

        # Verify shape progression using same logic as Position.initialize_pyramid
        level0 = label_image["0"]
        level1 = label_image["1"]
        level2 = label_image["2"]

        assert level0.shape == label_shape
        # Use _scale_integers function like Position class does
        from iohub.ngff.nodes import _scale_integers

        expected_level1_shape = label_shape[:-3] + _scale_integers(label_shape[-3:], 2)
        expected_level2_shape = label_shape[:-3] + _scale_integers(label_shape[-3:], 4)

        assert level1.shape == expected_level1_shape
        assert level2.shape == expected_level2_shape

        # Verify downscaled levels are empty (same as images)
        assert np.all(level1.numpy() == 0)
        assert np.all(level2.numpy() == 0)


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    label_name=short_alpha_numeric,
    version=ngff_versions_st,
)
@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_labels_metadata_structure(channels_and_random_5d, label_name, version):
    """Test `iohub.ngff.Position.create_label()` NGFF metadata compliance"""
    channel_names, random_5d = channels_and_random_5d
    # Create TZYX label data with fixed pattern for testing
    label_shape = (
        random_5d.shape[0],
        random_5d.shape[2],
        random_5d.shape[3],
        random_5d.shape[4],
    )
    label_data = np.zeros(label_shape, dtype=np.uint16)
    if label_data.size > 4:
        label_data.flat[0] = 1
        label_data.flat[1] = 2

    colors = {1: [255, 0, 0, 255], 2: [0, 255, 0, 255]}
    properties = [
        {"label-value": 1, "type": "cell"},
        {"label-value": 2, "type": "nucleus"},
    ]

    with _temp_ome_zarr(random_5d, channel_names, "0", version=version) as dataset:
        label_image = dataset.create_label(
            name=label_name,
            data=label_data,
            colors=colors,
            properties=properties,
        )

        # Verify Position metadata structure (NGFF compliant)
        assert hasattr(dataset.metadata, "labels")
        assert dataset.metadata.labels.labels == [label_name]
        assert dataset.metadata.labels.image_label is None

        # Verify individual label image metadata
        assert hasattr(label_image.metadata, "multiscales")
        assert hasattr(label_image.metadata, "image_label")
        assert len(label_image.metadata.image_label.colors) == 2
        assert len(label_image.metadata.image_label.properties) == 2
        assert label_image.metadata.image_label.source["image"] == "../../"


@given(
    channels_and_random_5d=_channels_and_random_5d(),
    version=ngff_versions_st,
)
@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.data_too_large],
)
def test_labels_access_patterns(channels_and_random_5d, version):
    """Test `iohub.ngff.Position` label access methods and iterators"""
    channel_names, random_5d = channels_and_random_5d
    # Create TZYX label data
    label_shape = (
        random_5d.shape[0],
        random_5d.shape[2],
        random_5d.shape[3],
        random_5d.shape[4],
    )
    label1 = np.ones(label_shape, dtype=np.uint8)
    label2 = np.full(label_shape, 2, dtype=np.uint16)

    with _temp_ome_zarr(random_5d, channel_names, "0", version=version) as dataset:
        # Create multiple labels with different pyramid levels
        dataset.create_label("cells", label1, pyramid_levels=2)
        dataset.create_label("nuclei", label2, pyramid_levels=1)

        # Test has_labels property
        assert dataset.has_labels is True

        # Test get_label() returns PositionLabel
        cells = dataset.get_label("cells")
        nuclei = dataset.get_label("nuclei")

        from iohub.ngff.nodes import PositionLabel

        assert isinstance(cells, PositionLabel)
        assert isinstance(nuclei, PositionLabel)

        # Test level access patterns
        assert "0" in cells
        assert "1" in cells
        assert "0" in nuclei
        assert "1" not in nuclei

        # Test labels() generator follows pattern of wells(), positions()
        label_names = []
        for name, label_img in dataset.labels():
            label_names.append(name)
            assert isinstance(label_img, PositionLabel)

        assert sorted(label_names) == ["cells", "nuclei"]


def test_initialize_pyramid(tmp_path):
    """Test initialize_pyramid creates pyramid structure with correct shapes and metadata."""
    store_path = tmp_path / "test_init_pyramid.zarr"
    shape = (1, 2, 32, 64, 64)
    scale = (1.0, 1.0, 2.0, 0.5, 0.5)
    levels = 3

    with open_ome_zarr(
        store_path,
        layout="fov",
        mode="a",
        channel_names=["ch1", "ch2"],
    ) as pos:
        pos.create_zeros(
            "0",
            shape=shape,
            dtype=np.uint16,
            transform=[TransformationMeta(type="scale", scale=list(scale))],
        )

        pos.initialize_pyramid(levels=levels)

        # Verify all levels created
        assert len(pos.array_keys()) == levels
        for level in range(levels):
            assert str(level) in pos.array_keys()

        # Verify shapes are downsampled correctly (ZYX only)
        assert pos["0"].shape == shape
        assert pos["1"].shape == (1, 2, 16, 32, 32)  # 2x downscaled in ZYX
        assert pos["2"].shape == (1, 2, 8, 16, 16)  # 4x downscaled in ZYX

        # Verify metadata has all levels
        dataset_paths = pos.metadata.multiscales[0].get_dataset_paths()
        assert dataset_paths == ["0", "1", "2"]

        # Verify scale transforms are updated for each level
        for level in range(levels):
            level_scale = pos.metadata.multiscales[0].datasets[level].coordinate_transformations[0].scale
            expected_factor = 2**level
            assert level_scale[-3:] == [
                scale[-3] * expected_factor,
                scale[-2] * expected_factor,
                scale[-1] * expected_factor,
            ]


@pytest.mark.parametrize("implementation", ["zarr", "tensorstore"])
def test_compute_pyramid(tmp_path, implementation):
    """Test pyramid computation fills levels with downsampled data."""
    if implementation == "tensorstore":
        pytest.importorskip("tensorstore")

    store_path = tmp_path / "test_pyramid.zarr"

    rng = np.random.default_rng()
    data = rng.integers(0, 255, size=(1, 2, 32, 64, 64), dtype=np.uint16)

    with open_ome_zarr(
        store_path,
        layout="fov",
        mode="a",
        channel_names=["ch1", "ch2"],
        implementation=implementation,
    ) as pos:
        pos.create_image("0", data)
        pos.compute_pyramid(levels=3, method="mean")

        # Verify pyramid levels contain actual downsampled data
        assert pos["1"][:].mean() > 0
        assert pos["2"][:].mean() > 0

        # Recompute with levels=None auto-detects existing pyramid
        pos.compute_pyramid(method="median")
        assert pos["1"][:].mean() > 0

        # Changing levels without delete raises error
        with pytest.raises(ValueError, match="delete_pyramid"):
            pos.compute_pyramid(levels=2, method="mean")


@pytest.mark.parametrize("implementation", ["zarr", "tensorstore"])
def test_delete_pyramid(tmp_path, implementation):
    """Test delete_pyramid removes all pyramid levels except level 0."""
    if implementation == "tensorstore":
        pytest.importorskip("tensorstore")

    store_path = tmp_path / "test_delete_pyramid.zarr"

    rng = np.random.default_rng()
    data = rng.integers(0, 255, size=(1, 2, 16, 64, 64), dtype=np.uint16)

    with open_ome_zarr(
        store_path,
        layout="fov",
        mode="a",
        channel_names=["ch1", "ch2"],
        implementation=implementation,
    ) as pos:
        pos.create_image("0", data)
        pos.compute_pyramid(levels=3, method="mean")

        # Verify pyramid exists
        assert "0" in pos
        assert "1" in pos
        assert "2" in pos

        # Verify metadata has all levels
        dataset_paths = pos.metadata.multiscales[0].get_dataset_paths()
        assert dataset_paths == ["0", "1", "2"]

        # Delete pyramid
        pos.delete_pyramid()

        # Verify only level 0 remains in zarr arrays
        assert "0" in pos
        assert "1" not in pos
        assert "2" not in pos

        # Verify metadata is also updated to only have level 0
        dataset_paths = pos.metadata.multiscales[0].get_dataset_paths()
        assert dataset_paths == ["0"]

        # Verify level 0 data is preserved
        assert_array_equal(pos["0"][:], data)

    # Verify metadata persists after reopening
    with open_ome_zarr(store_path, mode="r") as pos:
        dataset_paths = pos.metadata.multiscales[0].get_dataset_paths()
        assert dataset_paths == ["0"]


@given(config=_pyramid_config())
@pytest.mark.parametrize("implementation", ["zarr", "tensorstore"])
@settings(max_examples=32)
def test_initialize_pyramid_shapes(implementation, config):
    if implementation == "tensorstore":
        pytest.importorskip("tensorstore")
    """Test initialize_pyramid produces correct cascade shapes for any dims subset."""
    import math

    shape, dims, levels = config
    channel_names = ["ch0"] * shape[1]

    with TemporaryDirectory() as tmp:
        store_path = Path(tmp) / "pyramid-test.zarr"
        with open_ome_zarr(
            store_path, layout="fov", mode="a", channel_names=channel_names, implementation=implementation
        ) as pos:
            pos.create_zeros("0", shape=shape, dtype=np.float32)
            pos.initialize_pyramid(levels=levels, dims=dims)

            assert len(pos.array_keys()) == levels

            axis_names = pos.axis_names
            prev_shape = shape
            for level in range(1, levels):
                current_shape = pos[str(level)].shape
                for i, name in enumerate(axis_names):
                    if name in dims:
                        assert current_shape[i] == math.ceil(prev_shape[i] / 2), (
                            f"Level {level} axis '{name}': expected ceil({prev_shape[i]}/2)="
                            f"{math.ceil(prev_shape[i] / 2)}, got {current_shape[i]}"
                        )
                    else:
                        assert current_shape[i] == prev_shape[i], (
                            f"Level {level} axis '{name}' should be unchanged: "
                            f"expected {prev_shape[i]}, got {current_shape[i]}"
                        )
                prev_shape = current_shape


@pytest.mark.parametrize("implementation", ["zarr", "tensorstore"])
@given(config=_pyramid_config())
@settings(max_examples=32)
def test_initialize_pyramid_scale_metadata(implementation, config):
    if implementation == "tensorstore":
        pytest.importorskip("tensorstore")
    """Test initialize_pyramid sets correct cumulative scale metadata per axis."""
    shape, dims, levels = config
    channel_names = ["ch0"] * shape[1]

    with TemporaryDirectory() as tmp:
        store_path = Path(tmp) / "pyramid-test.zarr"
        with open_ome_zarr(
            store_path, layout="fov", mode="a", channel_names=channel_names, implementation=implementation
        ) as pos:
            pos.create_zeros("0", shape=shape, dtype=np.float32)
            pos.initialize_pyramid(levels=levels, dims=dims)

            axis_names = pos.axis_names
            base_scale = pos.get_effective_scale("0")
            for level in range(1, levels):
                level_scale = pos.get_effective_scale(str(level))
                for i, name in enumerate(axis_names):
                    if name in dims:
                        assert level_scale[i] == pytest.approx(base_scale[i] * 2**level), (
                            f"Level {level} axis '{name}': scale should be "
                            f"{base_scale[i] * 2**level}, got {level_scale[i]}"
                        )
                    else:
                        assert level_scale[i] == pytest.approx(base_scale[i]), (
                            f"Level {level} axis '{name}': scale should be unchanged "
                            f"{base_scale[i]}, got {level_scale[i]}"
                        )


@given(config=_pyramid_config())
@pytest.mark.parametrize("implementation", ["zarr", "tensorstore"])
@settings(max_examples=16, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_compute_pyramid_shapes(implementation, config):
    """Test compute_pyramid fills correct shapes for any dims subset."""
    import math

    if implementation == "tensorstore":
        pytest.importorskip("tensorstore")
    shape, dims, levels = config
    channel_names = ["ch0"] * shape[1]

    with TemporaryDirectory() as tmp:
        store_path = Path(tmp) / "test.zarr"
        with open_ome_zarr(
            store_path, layout="fov", mode="a", channel_names=channel_names, implementation=implementation
        ) as pos:
            pos.create_zeros("0", shape=shape, dtype=np.float32)
            pos.compute_pyramid(levels=levels, dims=dims)

            axis_names = pos.axis_names
            prev_shape = shape
            for level in range(1, levels):
                current_shape = pos[str(level)].shape
                for i, name in enumerate(axis_names):
                    if name in dims:
                        assert current_shape[i] == math.ceil(prev_shape[i] / 2)
                    else:
                        assert current_shape[i] == prev_shape[i]
                prev_shape = current_shape


@pytest.mark.parametrize("implementation", ["zarr", "tensorstore"])
def test_initialize_pyramid_invalid_dims(implementation, tmp_path):
    """Test that unknown axis names in dims raise ValueError."""
    if implementation == "tensorstore":
        pytest.importorskip("tensorstore")
    store_path = tmp_path / "test.zarr"
    with open_ome_zarr(store_path, layout="fov", mode="a", channel_names=["ch0"], implementation=implementation) as pos:
        pos.create_zeros("0", shape=(1, 1, 2, 8, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="not in dataset axes"):
            pos.initialize_pyramid(levels=2, dims={"w"})


# ---------- v0.4 dedicated tests ----------


def test_write_ome_zarr_v04_fov_roundtrip(tmp_path):
    """Full round-trip: create v0.4 FOV store, write image, read back."""
    store_path = tmp_path / "v04.ome.zarr"
    data = np.random.default_rng(42).random((1, 2, 3, 64, 64)).astype(np.float32)
    with open_ome_zarr(
        store_path,
        layout="fov",
        mode="w-",
        channel_names=["A", "B"],
        version="0.4",
    ) as ds:
        ds.create_image("0", data)
        assert ds.version == "0.4"
        # Verify v2 file structure
        assert (store_path / ".zgroup").exists()
        assert (store_path / ".zattrs").exists()
        assert not (store_path / "zarr.json").exists()
    # Re-open read-only
    with open_ome_zarr(store_path, layout="fov", mode="r") as ds:
        assert ds.version == "0.4"
        assert_array_equal(ds["0"][:], data)
        assert ds.channel_names == ["A", "B"]


def test_write_ome_zarr_v04_hcs_roundtrip(tmp_path):
    """HCS plate creation with v0.4."""
    store_path = tmp_path / "v04_hcs.ome.zarr"
    data = np.zeros((1, 2, 3, 32, 32), dtype=np.uint16)
    with open_ome_zarr(
        store_path,
        layout="hcs",
        mode="w-",
        channel_names=["A", "B"],
        version="0.4",
    ) as plate:
        pos = plate.create_position("A", "1", "0")
        pos.create_image("0", data)
        # Flat metadata, no "ome" wrapper
        assert "plate" in plate.zattrs
        assert "ome" not in plate.zattrs
    with open_ome_zarr(store_path, layout="hcs", mode="r") as plate:
        assert plate.version == "0.4"
        assert_array_equal(plate["A/1/0"]["0"][:], data)


def test_sharding_raises_on_v04(tmp_path):
    """Sharding must raise ValueError for v0.4."""
    store_path = tmp_path / "v04_shard.zarr"
    with open_ome_zarr(
        store_path,
        layout="fov",
        mode="w-",
        channel_names=["A"],
        version="0.4",
    ) as ds:
        with pytest.raises(ValueError, match="Sharding is not supported"):
            ds.create_image(
                "0",
                np.zeros((1, 1, 1, 64, 64)),
                shards_ratio=(1, 1, 1, 2, 2),
            )
