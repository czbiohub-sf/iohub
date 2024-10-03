import itertools
import string
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple

import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings
from numpy.typing import DTypeLike

from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import (
    apply_transform_to_czyx_and_save,
    create_empty_plate,
    process_single_position,
)


@contextmanager
def _temp_ome_zarr(
    store_name: str,
    position_keys: list[Tuple[str, str, str]],
    channel_names: list[str],
    shape: Tuple[int, ...],
    chunks: Optional[Tuple[int, ...]] = None,
    scale: Tuple[float, ...] = (1, 1, 1, 1, 1),
    dtype: DTypeLike = np.float32,
    base_dir: Optional[Path] = None,  # Added base_dir parameter
):
    """
    Helper context manager to generate a temporary OME-Zarr store.

    Parameters
    ----------
    store_name : str
        Name of the store, e.g., "input.zarr" or "output.zarr".
    position_keys : list[Tuple[str, str, str]]
        list of position keys, e.g., [("A", "1", "0")].
    channel_names : list[str]
        list of channel names.
    shape : Tuple[int, ...]
        TCZYX shape of the plate.
    chunks : Optional[Tuple[int, ...]], optional
        TCZYX chunk size, by default None.
    scale : Tuple[float, ...], optional
        TCZYX scale of the plate, by default (1, 1, 1, 1, 1).
    dtype : DTypeLike, optional
        Data type of the plate, by default np.float32.
    base_dir : Optional[Path], optional
        Base directory to create the store in.
        If None, a new TemporaryDirectory is created.

    Yields
    ------
    Path
        Path to the temporary OME-Zarr store.
    """
    if base_dir is None:
        # Create a new temporary directory if base_dir is not provided
        temp_dir = TemporaryDirectory()
        store_dir = Path(temp_dir.name)
        try:
            store_path = store_dir / store_name
            create_empty_plate(
                store_path=store_path,
                position_keys=position_keys,
                channel_names=channel_names,
                shape=shape,
                chunks=chunks,
                scale=scale,
                dtype=dtype,
            )
            yield store_path
        finally:
            temp_dir.cleanup()
    else:
        # Use the provided base_dir to create the store
        store_path = base_dir / store_name
        create_empty_plate(
            store_path=store_path,
            position_keys=position_keys,
            channel_names=channel_names,
            shape=shape,
            chunks=chunks,
            scale=scale,
            dtype=dtype,
        )
        yield store_path


@contextmanager
def _temp_ome_zarr_stores(
    position_keys: list[Tuple[str, str, str]],
    channel_names: list[str],
    shape: Tuple[int, ...],
    chunks: Optional[Tuple[int, ...]] = None,
    scale: Tuple[float, ...] = (1, 1, 1, 1, 1),
    dtype: DTypeLike = np.float32,
):
    """
    Helper context manager to generate temporary
    OME-Zarr input and output stores.

    Parameters
    ----------
    position_keys : list[Tuple[str, str, str]]
        list of position keys, e.g., [("A", "1", "0")].
    channel_names : list[str]
        list of channel names.
    shape : Tuple[int, ...]
        TCZYX shape of the plate.
    chunks : Optional[Tuple[int, ...]], optional
        TCZYX chunk size, by default None.
    scale : Tuple[float, ...], optional
        TCZYX scale of the plate, by default (1, 1, 1, 1, 1).
    dtype : DTypeLike, optional
        Data type of the plate, by default np.float32.

    Yields
    ------
    Tuple[Path, Path]
        Paths to the input and output OME-Zarr stores.
    """
    with TemporaryDirectory() as temp_dir:
        base_dir = Path(temp_dir)

        # Create input store
        with _temp_ome_zarr(
            store_name="input.zarr",
            position_keys=position_keys,
            channel_names=channel_names,
            shape=shape,
            chunks=chunks,
            scale=scale,
            dtype=dtype,
            base_dir=base_dir,  # Use the same base directory
        ) as input_store_path:
            # Create output store
            with _temp_ome_zarr(
                store_name="output.zarr",
                position_keys=position_keys,
                channel_names=channel_names,
                shape=shape,
                chunks=chunks,
                scale=scale,
                dtype=dtype,
                base_dir=base_dir,  # Use the same base directory
            ) as output_store_path:
                yield input_store_path, output_store_path


@st.composite
def plate_setup(draw):
    alphanum = string.ascii_letters + string.digits

    # Generate position keys
    position_keys = draw(
        st.lists(
            st.tuples(
                st.text(alphabet=alphanum, min_size=1, max_size=3),  # Plate
                st.text(alphabet=alphanum, min_size=1, max_size=3),  # Well
                st.text(
                    alphabet=alphanum, min_size=1, max_size=3
                ),  # Field of View
            ),
            min_size=1,
            max_size=3,
        )
    )

    # Generate number of channels
    num_channels = draw(st.integers(min_value=1, max_value=3))

    # Generate channel names based on the number of channels
    channel_names = [f"Channel_{i}" for i in range(num_channels)]

    # Generate shape ensuring that the
    # second dimension (C) matches num_channels
    T = draw(st.integers(min_value=1, max_value=3))  # Time
    Z = draw(st.integers(min_value=1, max_value=3))  # Z-slices
    Y = draw(st.integers(min_value=8, max_value=32))  # Y-dimension
    X = draw(st.integers(min_value=8, max_value=32))  # X-dimension
    shape = (T, num_channels, Z, Y, X)  # TCZYX

    # Generate chunks
    # Ensure that chunks are compatible with the shape dimensions
    chunks = draw(
        st.one_of(
            st.none(),
            st.tuples(
                st.integers(min_value=1, max_value=min(3, T)),  # T
                st.integers(min_value=1, max_value=min(3, num_channels)),  # C
                st.integers(min_value=1, max_value=min(3, Z)),  # Z
                st.integers(min_value=1, max_value=min(5, Y)),  # Y
                st.integers(min_value=1, max_value=min(5, X)),  # X
            ),
        )
    )

    # Generate scale
    scale = draw(
        st.lists(
            st.floats(
                min_value=0.1,
                max_value=2.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=5,
            max_size=5,
        )
    )

    # Generate dtype
    dtype = draw(st.sampled_from([np.float32, np.int16, np.uint8]))

    return position_keys, channel_names, shape, chunks, scale, dtype


@st.composite
def apply_transform_czyx_setup(draw):
    """
    Composite strategy to generate plate setup
    along with valid channel and time indices
    Returns
    -------
    Tuple containing:
        - position_keys
        - channel_names
        - shape
        - chunks
        - scale
        - dtype
        - channel_indices
        - time_indices
    """
    # Generate plate setup parameters
    position_keys, channel_names, shape, chunks, scale, dtype = draw(
        plate_setup()
    )
    T, C = shape[:2]

    # Define a helper strategy to generate channel indices based on C
    channel_indices_strategy = st.one_of(
        st.builds(
            slice,
            st.integers(min_value=0, max_value=0),
            st.integers(min_value=1, max_value=C),
            st.just(1),
        ),
        st.lists(
            st.integers(min_value=0, max_value=C - 1),
            min_size=1,
            max_size=min(3, C),
        ),
    )

    time_indices_strategy = st.one_of(
        st.lists(
            st.integers(min_value=0, max_value=T - 1),
            min_size=1,
            max_size=min(3, T),
        ),
    )

    # Generate input and output channel indices based on C
    channel_indices = draw(channel_indices_strategy)
    time_indices = draw(time_indices_strategy)

    return (
        position_keys,
        channel_names,
        shape,
        chunks,
        scale,
        dtype,
        channel_indices,
        time_indices,
    )


@st.composite
def process_single_position_setup(draw):
    """
    Composite strategy to generate plate setup
    along with valid channel and time indices

    Returns
    -------
    Tuple containing:
        - position_keys
        - channel_names
        - shape
        - chunks
        - scale
        - dtype
        - channel_indices
        - time_indices
    """
    # Generate plate setup parameters
    position_keys, channel_names, shape, chunks, scale, dtype = draw(
        plate_setup()
    )
    # NOTE: Chunking along T,C =1,1
    if chunks is not None:
        chunks = (1, 1) + chunks[2:]

    T, C = shape[:2]

    # Define a helper strategy to generate channel indices based on C
    channel_indices_strategy = st.one_of(
        st.none(),
        st.lists(
            st.builds(
                slice,
                st.integers(min_value=0, max_value=0),
                st.integers(min_value=1, max_value=C),
                st.just(1),
            ),
            min_size=1,
            max_size=min(3, C),
        ),
        st.lists(
            st.lists(
                st.integers(min_value=0, max_value=C - 1),
                min_size=1,
                max_size=C,
                # ensure each inner list has one element),
            ),
            min_size=1,
            max_size=min(3, C),
        ),
    )

    time_indices_strategy = st.one_of(
        st.none(),
        st.lists(
            st.integers(min_value=0, max_value=T - 1),
            min_size=1,
            max_size=min(3, T),
        ),
    )

    # Generate input and output channel indices based on C
    channel_indices = draw(channel_indices_strategy)
    time_indices = draw(time_indices_strategy)

    return (
        position_keys,
        channel_names,
        shape,
        chunks,
        scale,
        dtype,
        channel_indices,
        time_indices,
    )


# Define the transformation function
def dummy_transform(data, constant=2):
    return data * constant


# Populate the input store with random data
def populate_store(
    input_store_path: Path,
    position_keys: list[Tuple[str, str, str]],
    shape: Tuple[int, ...],
    dtype: DTypeLike,
):
    with open_ome_zarr(input_store_path, mode="r+") as input_dataset:
        for position_key_tuple in position_keys:
            position_path = "/".join(position_key_tuple)
            position = input_dataset[position_path]
            T, C, Z, Y, X = shape
            for t in range(T):
                for c in range(C):
                    # Generate random data based on dtype
                    if np.issubdtype(dtype, np.floating):
                        data = np.random.rand(Z, Y, X).astype(dtype)
                    else:
                        data = np.random.randint(
                            1, 20, size=(Z, Y, X), dtype=dtype
                        )
                    position.data.oindex[t, c] = data


# Verify the transformation
def verify_transformation(
    input_store_path: Path,
    output_store_path: Path,
    position_key_tuple: Tuple[str, str, str],
    shape: Tuple[int, ...],
    time_indices: list[int],
    channel_indices: list[int],
    transform_func,
    **kwargs,
):
    with open_ome_zarr(input_store_path) as input_dataset, open_ome_zarr(
        output_store_path
    ) as output_dataset:
        position_key_tuple = "/".join(position_key_tuple)
        input_position = input_dataset[position_key_tuple]
        output_position = output_dataset[position_key_tuple]

        # Extract extra metadata if provided
        extra_metadata = kwargs.pop("extra_metadata", None)

        # Check if extra_metadata is provided
        if extra_metadata is not None:
            assert output_position.zattrs["extra_metadata"] == extra_metadata

        # Check the transformation for each time point and channel
        input_data = input_position.data.oindex[time_indices, channel_indices]
        output_data = output_position.data.oindex[
            time_indices, channel_indices
        ]
        expected_data = transform_func(input_data, **kwargs)

        np.testing.assert_array_almost_equal(
            output_data,
            expected_data,
            err_msg=f"Mismatch in position {position_key_tuple}",
        )


@given(
    plate_setup=plate_setup(),
    extra_channels=st.lists(
        st.text(min_size=5, max_size=16), min_size=1, max_size=3
    ),
)
@settings(max_examples=5)
def test_create_empty_plate(plate_setup, extra_channels):
    position_keys, channel_names, shape, chunks, scale, dtype = plate_setup

    with TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "test.zarr"

        # Call the function under test
        create_empty_plate(
            store_path=store_path,
            position_keys=position_keys,
            channel_names=channel_names,
            shape=shape,
            chunks=chunks,
            scale=scale,
            dtype=dtype,
        )

        # Verify the store was created
        assert store_path.exists()

        # Open the store and verify its contents
        with open_ome_zarr(store_path) as dataset:
            # Verify channel names
            assert dataset.channel_names == channel_names

            # Verify positions
            for position_key_tuple in position_keys:
                position_path = "/".join(position_key_tuple)
                position = dataset[position_path]

                # Check shape
                assert position.data.shape == shape

                # Check chunks if provided
                if chunks is not None:
                    assert position.data.chunks == chunks
                else:
                    assert position.data.chunks == (1, 1) + tuple(shape[-3:])

                # Check dtype
                assert position.data.dtype == dtype
                assert position.scale == scale

        # Test when zarr store already exists
        create_empty_plate(
            store_path=store_path,
            position_keys=position_keys,
            channel_names=extra_channels,
            shape=shape,
            chunks=chunks,
            scale=scale,
            dtype=dtype,
        )

        with open_ome_zarr(store_path) as dataset:
            assert dataset.channel_names == (channel_names + extra_channels)
            shape = (shape[0], shape[1] + len(extra_channels)) + shape[2:]
            for position_key_tuple in position_keys:
                position_path = "/".join(position_key_tuple)
                position = dataset[position_path]
                assert position.data.shape == shape


@given(
    setup=apply_transform_czyx_setup(),
    constant=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=5, deadline=None)
def test_apply_transform_to_zyx_and_save(setup, constant):
    (
        position_keys,
        channel_names,
        shape,
        chunks,
        scale,
        dtype,
        channel_indices,
        time_indices,
    ) = setup

    # Use the enhanced context manager to get both input and output store paths
    with _temp_ome_zarr_stores(
        position_keys=position_keys,
        channel_names=channel_names,
        shape=shape,
        chunks=chunks,
        scale=scale,
        dtype=dtype,
    ) as (input_store_path, output_store_path):
        # Populate the input store with random data
        populate_store(input_store_path, position_keys, shape, dtype)

        kwargs = {"constant": constant}

        # Apply the transformation for each position and time point
        for position_key_tuple in position_keys:
            input_position_path = input_store_path / Path(*position_key_tuple)
            output_position_path = output_store_path / Path(
                *position_key_tuple
            )

            for t_in in time_indices:
                apply_transform_to_czyx_and_save(
                    func=dummy_transform,
                    input_position_path=Path(input_position_path),
                    output_position_path=Path(output_position_path),
                    input_channel_indices=channel_indices,
                    output_channel_indices=channel_indices,
                    input_time_index=t_in,
                    output_time_index=t_in,
                    **kwargs,
                )

            # Verify the transformation
            verify_transformation(
                input_store_path,
                output_store_path,
                position_key_tuple,
                shape,
                time_indices,
                channel_indices,
                dummy_transform,
                **kwargs,
            )


@given(
    setup=process_single_position_setup(),
    constant=st.integers(min_value=1, max_value=3),
    num_processes=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=3, deadline=None)
def test_process_single_position(setup, constant, num_processes):
    # def test_process_single_position(setup, constant, num_processes):
    (
        position_keys,
        channel_names,
        shape,
        chunks,
        scale,
        dtype,
        channel_indices,
        time_indices,
    ) = setup

    # Use the enhanced context manager to get both input and output store paths
    with _temp_ome_zarr_stores(
        position_keys=position_keys,
        channel_names=channel_names,
        shape=shape,
        chunks=chunks,
        scale=scale,
        dtype=dtype,
    ) as (input_store_path, output_store_path):
        # Populate Store with random data
        populate_store(input_store_path, position_keys, shape, dtype)

        # Choose a single position to process (e.g., the first one)
        for position_key_tuple in position_keys:
            input_position_path = input_store_path / Path(*position_key_tuple)
            output_position_path = output_store_path / Path(
                *position_key_tuple
            )
            kwargs = {"constant": constant, "extra_metadata": {"temp": 10}}

            # Apply the transformation using process_single_position
            process_single_position(
                func=dummy_transform,
                input_position_path=input_position_path,
                output_position_path=output_position_path,
                input_channel_indices=channel_indices,
                output_channel_indices=channel_indices,
                input_time_indices=time_indices,
                output_time_indices=time_indices,
                num_processes=num_processes,
                **kwargs,
            )

            # Handle None for process_single_position_setup
            if time_indices is None:
                time_indices = list(range(shape[0]))
            if channel_indices is None:
                channel_indices = [[c] for c in range(shape[1])]

            print("time_indices", time_indices)
            print("channel_indices", channel_indices)
            # Verify the transformation
            iterable = itertools.product(time_indices, channel_indices)
            for t_idx, chan_idx in iterable:
                verify_transformation(
                    input_store_path,
                    output_store_path,
                    position_key_tuple,
                    shape,
                    t_idx,
                    chan_idx,
                    dummy_transform,
                    **kwargs,
                )
