import contextlib
import inspect
import io
import itertools
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Tuple, Union

import click
import numpy as np
from numpy.typing import DTypeLike

from iohub.ngff import Position, open_ome_zarr
from iohub.ngff_meta import TransformationMeta


def create_empty_plate(
    store_path: Path,
    position_keys: list[Tuple[str]],
    channel_names: list[str],
    shape: Tuple[int],
    chunks: Tuple[int] = None,
    scale: Tuple[float] = (1, 1, 1, 1, 1),
    dtype: DTypeLike = np.float32,
    max_chunk_size_bytes=500e6,
) -> None:
    """
    Create a new HCS Plate in OME-Zarr format if the plate does not exist.
    If the plate exists, append positions and channels
    if they are not already in the plate.

    Parameters
    ----------
    store_path : Path
        Path to the HCS plate.
    position_keys : list[Tuple[str]]
        Position keys to append if not present in the plate,
        e.g., [("A", "1", "0"), ("A", "1", "1")].
    channel_names : list[str]
        List of channel names. If the store exists,
        append if not present in metadata.
    shape : Tuple[int]
        TCZYX shape of the plate.
    chunks : Tuple[int], optional
        Chunk size for the plate TCZYX. If None,
        the chunk size is calculated based on the shape to be <500MB.
        Defaults to None.
    scale : Tuple[float], optional
        Scale of the plate TCZYX. Defaults to (1, 1, 1, 1, 1).
    dtype : DTypeLike, optional
        Data type of the plate. Defaults to np.float32.
    max_chunk_size_bytes : float, optional
        Maximum chunk size in bytes. Defaults to 500e6 (500MB).

    Examples
    --------
    Create a new plate with positions and channels:
    create_empty_plate(
        store_path=Path("/path/to/store"),
        position_keys=[("A", "1", "0"), ("A", "1", "1")],
        channel_names=["DAPI", "FITC"],
        shape=(1, 1, 256, 256, 256)
    )

    Create a plate with custom chunk size and scale:
    create_empty_plate(
        store_path=Path("/path/to/store"),
        position_keys=[("A", "1", "0")],
        channel_names=["DAPI"],
        shape=(1, 1, 256, 256, 256),
        chunks=(1, 1, 128, 128, 128),
        scale=(1, 1, 0.5, 0.5, 0.5)
    )

    Notes
    -----
    - If `chunks` is not provided, the function calculates an appropriate
    chunk size to keep the chunks under the specified `max_chunk_size_bytes`.
    - The function ensures that positions and channels are appended to an
    existing plate if they are not already present.
    """
    bytes_per_pixel = np.dtype(dtype).itemsize

    # Limiting the chunking to 500MB
    if chunks is None:
        chunk_zyx_shape = list(shape[-3:])
        # XY image is larger than MAX_CHUNK_SIZE
        while (
            chunk_zyx_shape[-3] > 1
            and np.prod(chunk_zyx_shape) * bytes_per_pixel
            > max_chunk_size_bytes
        ):
            chunk_zyx_shape[-3] = np.ceil(chunk_zyx_shape[-3] / 2).astype(int)
        chunk_zyx_shape = tuple(chunk_zyx_shape)

        chunks = 2 * (1,) + chunk_zyx_shape

    # Create plate
    output_plate = open_ome_zarr(
        str(store_path), layout="hcs", mode="a", channel_names=channel_names
    )

    # Create positions
    for position_key in position_keys:
        position_key_string = "/".join(position_key)
        # Check if position is already in the store, if not create it
        if position_key_string not in output_plate.zgroup:
            position = output_plate.create_position(*position_key)
            _ = position.create_zeros(
                name="0",
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                transform=[TransformationMeta(type="scale", scale=scale)],
            )
        else:
            position = output_plate[position_key_string]

    # Check if channel_names are already in the store, if not append them
    for channel_name in channel_names:
        metadata_channel_names = position.channel_names
        if channel_name not in metadata_channel_names:
            position.append_channel(channel_name, resize_arrays=True)


def apply_transform_to_zyx_and_save(
    func,
    position: Position,
    output_store_path: Path,
    channel_indices_in: Union[list[slice], list[list[int]]],
    channel_indices_out: Union[list[slice], list[list[int]]],
    time_indices_in: int,
    time_indices_out: int,
    **kwargs,
) -> None:
    """
    Load a CZYX array from a Position object,
    apply a transformation, and save the result.

    Parameters
    ----------
    func : Callable
        The function to be applied to the data.
        Should take a CZYX array and return a transformed CZYX array.
    position : Position
        The position object to read from.
    output_store_path : Path
        The path to output OME-Zarr Store.
    channel_indices_in : Union[list[slice], list[list[int]]]
        The channel indices to process. Acceptable values:
        - A list of slices: [slice(0, 2), slice(2, 4), ...].
        - A list of lists of integers: [[0], [1], [2, 3, 4], ...].
        If empty list, process all channels.
    channel_indices_out : Union[list[slice], list[list[int]]]
        The channel indices to write to. Acceptable values:
        - A list of slices: [slice(0, 2), slice(2, 4), ...].
        - A list of lists of integers: [[0], [1], [2, 3, 4], ...].
        If empty list, write to all channels.
    time_indices_in : int
        The time index to process.
    time_indices_out : int
        The time index to write to.
    kwargs : dict, optional
        Additional arguments to pass to the function.
        A dictionary with key "extra_metadata" can be passed to
        be stored at a FOV level, e.g.,
        kwargs={"extra_metadata": {"Temperature": 37.5, "CO2_level": 0.5}}.

    Examples
    --------
    Using slices for channel_indices_in:
    apply_transform_to_zyx_and_save(
        func=some_function,
        position=some_position,
        output_store_path=Path("/path/to/output"),
        channel_indices_in=[slice(0, 2), slice(2, 4)],
        channel_indices_out=[[0], [1]],
        time_indices_in=0,
        time_indices_out=0,
    )

    Using list of lists for channel_indices_in:
    apply_transform_to_zyx_and_save(
        func=some_function,
        position=some_position,
        output_store_path=Path("/path/to/output"),
        channel_indices_in=[[0, 1], [2, 3]],
        channel_indices_out=[[0], [1]],
        time_indices_in=0,
        time_indices_out=0,
    )

    Notes
    -----
    - If channel_indices_in or channel_indices_out
    contain nested lists, the indices should be integers.
    - Ensure that the lengths of channel_indices_in and
    channel_indices_out match if they are provided.
    """

    # TODO: temporary fix to slumkit issue
    if _is_nested(channel_indices_in):
        channel_indices_in = [
            int(x) for x in channel_indices_in if x.isdigit()
        ]
    if _is_nested(channel_indices_out):
        channel_indices_out = [
            int(x) for x in channel_indices_out if x.isdigit()
        ]

    # Check if time_indices_in should be added to the func kwargs
    # This is needed when a different processing is needed for each time point,
    # for example during stabilization
    all_func_params = inspect.signature(func).parameters.keys()
    if "time_indices_in" in all_func_params:
        kwargs["time_indices_in"] = time_indices_in

    # Process CZYX given with the given indeces
    # if channel_indices_in is not None and len(channel_indices_in) > 0:
    click.echo(
        f"Processing t={time_indices_in} and channels {channel_indices_in}"
    )
    czyx_data = position.data.oindex[time_indices_in, channel_indices_in]
    if not _check_nan_n_zeros(czyx_data):
        transformed_czyx = func(czyx_data, **kwargs)
        # Write to file
        with open_ome_zarr(output_store_path, mode="r+") as output_dataset:
            output_dataset[0].oindex[
                time_indices_out, channel_indices_out
            ] = transformed_czyx
        click.echo(
            f"Finished Writing.. t={time_indices_in} and \
            channel output={channel_indices_out}"
        )
    else:
        click.echo(f"Skipping t={time_indices_in} due to all zeros or nans")


# TODO: modify how we get the time and channesl like recOrder
# (isinstance(input, list) or instance(input,int) or all)
def process_single_position(
    func,
    input_position_path: Path,
    output_store_path: Path,
    channel_indices_in: Union[list[slice], list[list[int]]] = [],
    channel_indices_out: Union[list[slice], list[list[int]]] = [],
    time_indices_in: Union[list[Union[int, slice]], int] = "all",
    time_indices_out: list[int] = [],
    num_processes: int = 1,
    **kwargs,
) -> None:
    """
    Apply function to data in an `iohub` `Position`,
    parallelizing over time and channel indices,
    and save result in an output Zarr store.

    Parameters
    ----------
    func :CZYX -> CZYX Callable
        The function to be applied to the data.
        Should take a CZYX array and return a transformed CZYX array.
    input_position_path : Path
        The path to the input Position (e.g., input_position_path.zarr/0/0/0).
    output_store_path : Path
        The path to the output OME-Zarr store (e.g., output_store_path.zarr).
    time_indices_in : Union[list[Union[int, slice]], int], optional
        The time indices to process. Acceptable values:
        - "all": Process all time points.
        - A list of slices: [slice(0, 2), slice(2, 4), ...].
        - A single integer: 0, 1, ...
        If "all", time_indices_out should also be "all". Defaults to "all".
    time_indices_out : list[int], optional
        The time indices to write to. Must match time_indices_in if not empty.
        Typically used for stabilization, which needs per timepoint processing.
        Defaults to an empty list.
    channel_indices_in : Union[list[slice], list[list[int]]], optional
        The channel indices to process. Acceptable values:
        - A list of slices: [slice(0, 2), slice(2, 4), ...].
        - A list of lists of integers: [[0], [1], [2, 3, 4], ...].
        If empty, process all channels.
        Must match channel_indices_out if not empty.
        Defaults to an empty list.
    channel_indices_out : Union[list[slice], list[list[int]]], optional
        The channel indices to write to. Acceptable values:
        - A list of slices: [slice(0, 2), slice(2, 4), ...].
        - A list of lists of integers: [[0], [1], [2, 3, 4], ...].
        If empty, write to all channels.
        Must match channel_indices_in if not empty.
        Defaults to an empty list.
    num_processes : int, optional
        Number of simultaneous processes per position. Defaults to 1.
    kwargs : dict, optional
        Additional arguments to pass to the function.
        A dictionary with key "extra_metadata"
        can be passed to be stored at a FOV level,
        e.g.,
        kwargs={"extra_metadata": {"Temperature": 37.5, "CO2_level": 0.5}}.

    Examples
    --------
    Using slices for channel_indices_in:
    process_single_position(
        func=some_function,
        input_position_path=Path("/path/to/input"),
        output_store_path=Path("/path/to/output"),
        time_indices_in=[slice(1, 2), slice(2, 3)],
        channel_indices_in=[slice(0, 2), slice(2, 4)],
        channel_indices_out=[[0], [1]],
    )

    Using list of lists for channel_indices_in:
    process_single_position(
        func=some_function,
        input_position_path=Path("/path/to/input"),
        output_store_path=Path("/path/to/output"),
        time_indices_in=[slice(1, 2), slice(2, 3)],
        channel_indices_in=[[0, 1], [2, 3]],
        channel_indices_out=[[0], [1]],
    )

    Notes
    -----
    - Multiprocessing over T and C:
    channel_indices_in and channel_indices_out should be empty.
    - Multiprocessing over T only:
    channel_indices_in and channel_indices_out should be provided.
    """
    # Function to be applied
    click.echo(f"Function to be applied: \t{func}")

    # Get the reader and writer
    click.echo(f"Input data path:\t{input_position_path}")
    click.echo(f"Output data path:\t{str(output_store_path)}")
    input_dataset = open_ome_zarr(str(input_position_path))
    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer):
        input_dataset.print_tree()
    click.echo(f" Input data tree: {stdout_buffer.getvalue()}")

    # Find time indices
    if time_indices_in == "all":
        time_indices_in = range(input_dataset.data.shape[0])
        time_indices_out = time_indices_in
    elif isinstance(time_indices_in, list):
        # Check for invalid times
        time_ubound = input_dataset.data.shape[0] - 1
        if np.max(time_indices_in) > time_ubound:
            raise ValueError(
                f"time_indices_in = {time_indices_in} includes \
                a time index beyond the maximum index of \
                the dataset = {time_ubound}"
            )
        # Handle the case when time_indices out is not provided.
        # It defaults to the t_indices_in
        if len(time_indices_out) == 0:
            time_indices_out = range(len(time_indices_in))

    # Check the arguments for the function
    all_func_params = inspect.signature(func).parameters.keys()
    # Extract the relevant kwargs for the function 'func'
    func_args = {}
    non_func_args = {}

    for k, v in kwargs.items():
        if k in all_func_params:
            func_args[k] = v
        else:
            non_func_args[k] = v

    # Write the settings into the metadata if existing
    if "extra_metadata" in non_func_args:
        # For each dictionary in the nest
        with open_ome_zarr(output_store_path, mode="r+") as output_dataset:
            for params_metadata_keys in kwargs["extra_metadata"].keys():
                output_dataset.zattrs["extra_metadata"] = non_func_args[
                    "extra_metadata"
                ]

    # Loop through (T, C), deskewing and writing as we go
    click.echo(f"\nStarting multiprocess pool with {num_processes} processes")

    if channel_indices_in is None or len(channel_indices_in) == 0:
        # If C is not empty, use itertools.product with both ranges
        _, C, _, _, _ = input_dataset.data.shape
        iterable = [
            ([c], [c], time_idx, time_idx_out)
            for (time_idx, time_idx_out), c in itertools.product(
                zip(time_indices_in, time_indices_out), range(C)
            )
        ]
        partial_apply_transform_to_zyx_and_save = partial(
            apply_transform_to_zyx_and_save,
            func,
            input_dataset,
            output_store_path / Path(*input_position_path.parts[-3:]),
            **func_args,
        )
    else:
        # If C is empty, use only the range for time_indices_in
        iterable = list(zip(time_indices_in, time_indices_out))
        partial_apply_transform_to_zyx_and_save = partial(
            apply_transform_to_zyx_and_save,
            func,
            input_dataset,
            output_store_path / Path(*input_position_path.parts[-3:]),
            channel_indices_in,
            channel_indices_out,
            **func_args,
        )

    click.echo(f"\nStarting multiprocess pool with {num_processes} processes")
    with mp.Pool(num_processes) as p:
        p.starmap(
            partial_apply_transform_to_zyx_and_save,
            iterable,
        )


def _is_nested(lst):
    """
    Check if the list is nested or not.

    NOTE: this function was created for a bug in slumkit that nested
    channel_indices_in into a list of lists
    TODO: check if this is still an issue in slumkit
    """
    return any(isinstance(i, list) for i in lst) or any(
        isinstance(i, str) for i in lst
    )


def _check_nan_n_zeros(input_array):
    """
    Checks if any of the channels are all zeros or nans and returns true
    """
    if len(input_array.shape) == 3:
        # Check if all the values are zeros or nans
        if np.all(input_array == 0) or np.all(np.isnan(input_array)):
            # Return true
            return True
    elif len(input_array.shape) == 4:
        # Get the number of channels
        num_channels = input_array.shape[0]
        # Loop through the channels
        for c in range(num_channels):
            # Get the channel
            zyx_array = input_array[c, :, :, :]

            # Check if all the values are zeros or nans
            if np.all(zyx_array == 0) or np.all(np.isnan(zyx_array)):
                # Return true
                return True
    else:
        raise ValueError("Input array must be 3D or 4D")

    # Return false
    return False
