import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import (
    apply_transform_to_czyx_and_save,
    create_empty_plate,
    process_single_position,
)

# Define strategies for the function parameters
position_keys_st = st.lists(
    st.tuples(
        st.text(min_size=1, max_size=3),
        st.text(min_size=1, max_size=3),
        st.text(min_size=1, max_size=3),
    ),
    min_size=1,
    max_size=10,
)
channel_names_st = st.lists(
    st.text(min_size=1, max_size=10), min_size=1, max_size=5
)
shape_st = st.tuples(
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=10),
)
chunks_st = st.one_of(
    st.none(),
    st.tuples(
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=10),
    ),
)
scale_st = st.tuples(
    st.floats(min_value=0.1, max_value=10),
    st.floats(min_value=0.1, max_value=10),
    st.floats(min_value=0.1, max_value=10),
    st.floats(min_value=0.1, max_value=10),
    st.floats(min_value=0.1, max_value=10),
)
dtype_st = st.sampled_from([np.float32, np.int16, np.uint8])


@given(
    store_path=st.just(Path("test.zarr")),
    position_keys=position_keys_st,
    channel_names=channel_names_st,
    shape=shape_st,
    chunks=chunks_st,
    scale=scale_st,
    dtype=dtype_st,
)
@settings(max_examples=5)
def test_create_empty_hcs_zarr(
    store_path, position_keys, channel_names, shape, chunks, scale, dtype
):
    with TemporaryDirectory() as temp_dir:
        output_zarr = Path(temp_dir) / "output.zarr"
        create_empty_plate(
            store_path,
            output_zarr / position_keys_st,
            channel_names,
            shape,
            chunks,
            scale,
            dtype,
        )
        assert os.path.isdir(output_zarr)
        with open_ome_zarr(store_path) as dataset:
            assert dataset.zattrs["channel_names"] == channel_names
            position = dataset[position_keys_st[0]]
            assert shape_st == position.data.shape
            assert position.chunks == chunks
            assert position.scale == scale
            assert position.dtype == dtype
            assert position.data.dtype == dtype


@given(
    channel_indices=st.lists(
        st.integers(min_value=0, max_value=1), min_size=1, max_size=2
    ),
    t_idx=st.integers(min_value=0, max_value=2),
    shape=shape_st,
    dtype=dtype_st,
)
@settings(max_examples=5)
def test_apply_transform_to_zyx_and_save(channel_indices, t_idx, shape, dtype):
    with TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "test.zarr"
        create_empty_plate(
            store_path,
            [("A", "1", "0")],
            ["Channel1", "Channel2"],
            shape,
            dtype=dtype,
        )

        def dummy_transform(data, **kwargs):
            return data * 2

        position = open_ome_zarr(store_path)["A/1/0"]
        apply_transform_to_czyx_and_save(
            dummy_transform,
            position,
            store_path,
            channel_indices,
            channel_indices,
            t_idx,
            t_idx,
        )
        # TODO: add assertions


@given(
    func=st.just(lambda x: x),
    input_data_path=st.just(Path("input.zarr")),
    output_path=st.just(Path("output.zarr")),
    time_indices_in=st.lists(
        st.integers(min_value=0, max_value=2), min_size=1, max_size=3
    ),
    time_indices_out=st.lists(
        st.integers(min_value=0, max_value=2), min_size=1, max_size=3
    ),
    channel_indices=st.lists(
        st.integers(min_value=0, max_value=1), min_size=1, max_size=2
    ),
    num_processes=st.integers(min_value=1, max_value=2),
    shape=shape_st,
    dtype=dtype_st,
)
@settings(max_examples=5)
def test_process_single_position(
    func,
    input_data_path,
    output_path,
    time_indices_in,
    time_indices_out,
    channel_indices,
    num_processes,
    shape,
    dtype,
):
    with TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "input.zarr"
        output_path = Path(temp_dir) / "output.zarr"
        create_empty_plate(
            input_path,
            [("A", "1", "0")],
            ["Channel1", "Channel2"],
            shape,
            dtype=dtype,
        )
        create_empty_plate(
            output_path,
            [("A", "1", "0")],
            ["Channel1", "Channel2"],
            shape,
            dtype=dtype,
        )

        process_single_position(
            func,
            input_path,
            output_path,
            time_indices_in,
            time_indices_out,
            channel_indices,
            channel_indices,
            num_processes,
        )
        # TODO: add assertions
