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
    num_channels = draw(st.integers(min_value=1, max_value=5))

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
                st.integers(min_value=1, max_value=min(5, T)),  # T
                st.integers(min_value=1, max_value=min(5, num_channels)),  # C
                st.integers(min_value=1, max_value=min(5, Z)),  # Z
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
        - input_channel_indices
        - output_channel_indices
        - input_time_indices
        - output_time_indices
    """
    # Generate plate setup parameters
    position_keys, channel_names, shape, chunks, scale, dtype = draw(
        plate_setup()
    )
    T, C = shape[:2]

    # Define a helper strategy to generate channel indices based on C
    channel_indices_strategy = st.one_of(
        st.none(),
        st.lists(st.slices(size=C), min_size=1, max_size=min(3, C)),
        st.lists(
            st.lists(st.integers(min_value=0, max_value=C - 1)),
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
    input_channel_indices = draw(channel_indices_strategy)
    output_channel_indices = draw(channel_indices_strategy)
    input_time_indices = draw(time_indices_strategy)
    output_time_indices = draw(time_indices_strategy)

    return (
        position_keys,
        channel_names,
        shape,
        chunks,
        scale,
        dtype,
        input_channel_indices,
        output_channel_indices,
        input_time_indices,
        output_time_indices,
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
    transform_func,
):
    with open_ome_zarr(input_store_path) as input_dataset, open_ome_zarr(
        output_store_path
    ) as output_dataset:
        position_key_tuple = "/".join(position_key_tuple)
        input_position = input_dataset[position_key_tuple]
        output_position = output_dataset[position_key_tuple]

        T, C, Z, Y, X = shape
        for t in range(T):
            for c in range(C):
                input_data = input_position.data.oindex[t, c][:]
                output_data = output_position.data.oindex[t, c][:]

                expected_data = transform_func(input_data)
                np.testing.assert_array_almost_equal(
                    output_data,
                    expected_data,
                    err_msg=f"Mismatch in position \
                        {position_key_tuple}, time {t}, channel {c}.",
                )


@given(
    setup=process_single_position_setup(),
    constant=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=3)
def test_apply_transform_to_zyx_and_save(setup, constant):
    (
        position_keys,
        channel_names,
        shape,
        chunks,
        scale,
        dtype,
        input_channel_indices,
        output_channel_indices,
        input_time_indices,
        output_time_indices,
    ) = setup

    input_channel_indices = input_channel_indices[0]
    output_channel_indices = output_channel_indices[0]
    input_time_indices = input_time_indices[0]
    output_time_indices = output_time_indices[0]

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

            for t_in, t_out in zip(input_time_indices, output_time_indices):
                apply_transform_to_czyx_and_save(
                    func=dummy_transform,
                    input_position_path=Path(input_position_path),
                    output_position_path=Path(output_position_path),
                    input_channel_indices=input_channel_indices,
                    output_channel_indices=output_channel_indices,
                    input_time_index=t_in,
                    output_time_index=t_out,
                    **kwargs,
                )

        # Verify the transformation
        verify_transformation(
            input_store_path,
            output_store_path,
            position_key_tuple,
            shape,
            dummy_transform,
        )


@settings(max_examples=3)
@given(
    setup=process_single_position_setup(),
    constant=st.integers(min_value=1, max_value=5),
    num_processes=st.integers(min_value=1, max_value=4),
)
def test_process_single_position(setup, constant, num_processes):
    (
        position_keys,
        channel_names,
        shape,
        chunks,
        scale,
        dtype,
        input_channel_indices,
        output_channel_indices,
        input_time_indices,
        output_time_indices,
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
        position_key_tuple = position_keys[0]
        input_position_path = Path(input_store_path) / Path(
            *position_key_tuple
        )
        output_position_path = Path(output_store_path) / Path(
            *position_key_tuple
        )

        kwargs = {"constant": constant, "extra_metadata": {"temp": 10}}

        # Apply the transformation using process_single_position
        process_single_position(
            func=dummy_transform,
            input_position_path=input_position_path,
            output_position_path=output_position_path,
            input_channel_indices=input_channel_indices,
            output_channel_indices=output_channel_indices,
            input_time_indices=input_time_indices,
            output_time_indices=output_time_indices,
            num_processes=num_processes,
            **kwargs,
        )

        # Verify the transformation
        verify_transformation(
            input_store_path,
            output_store_path,
            position_key_tuple,
            shape,
            dummy_transform,
        )
