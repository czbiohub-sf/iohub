import contextlib
import inspect
import io
import itertools
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Tuple

import click
import numpy as np
from numpy.typing import DTypeLike

from iohub.ngff import Position, open_ome_zarr
from iohub.ngff_meta import TransformationMeta


def create_empty_hcs_zarr(
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
    This function creates a new HCS Plate in OME-Zarr format if the plate does not exist.

    If the plate exists, appends positions and channels if they are not
    already in the plate.

    Parameters
    ----------
    store_path : Path
        hcs plate path
    position_keys : list[Tuple[str]]
        Position keys, will append if not present in the plate.
        e.g. [("A", "1", "0"), ("A", "1", "1")]
    shape : Tuple[int]
    chunks : Tuple[int]
    scale : Tuple[float]
    channel_names : list[str]
        Channel names, will append if not present in metadata.
    dtype : DTypeLike
    """
    MAX_CHUNK_SIZE = max_chunk_size_bytes  # in bytes
    bytes_per_pixel = np.dtype(dtype).itemsize

    # Limiting the chunking to 500MB
    if chunks is None:
        chunk_zyx_shape = list(shape[-3:])
        # XY image is larger than MAX_CHUNK_SIZE
        while (
            chunk_zyx_shape[-3] > 1
            and np.prod(chunk_zyx_shape) * bytes_per_pixel > MAX_CHUNK_SIZE
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
        # Read channel names directly from metadata to avoid race conditions
        metadata_channel_names = [
            channel.label for channel in position.metadata.omero.channels
        ]
        if channel_name not in metadata_channel_names:
            position.append_channel(channel_name, resize_arrays=True)


def apply_transform_to_zyx_and_save(
    func,
    position: Position,
    output_path: Path,
    input_channel_indices: list[int],
    output_channel_indices: list[int],
    t_idx_in: int,
    t_idx_out: int,
    **kwargs,
) -> None:
    """
    Load a CZYX array from a Position object, apply a transformation to CZYX.

    Parameters
    ----------
    func : CZYX -> CZYX function
        The function to be applied to the data
    position : Position
        The position object to read from
    output_path : Path
        The path to output OME-Zarr Store
    input_channel_indices : list
        The channel indices to process.
        If empty list, process all channels.
        Must match output_channel_indices if not empty
    output_channel_indices : list
        The channel indices to write to.
        If empty list, write to all channels.
        Must match input_channel_indices if not empty
    t_idx_in : int
        The time index to process
    t_idx_out : int
        The time index to write to
    kwargs : dict
        Additional arguments to pass to the CZYX function
    """

    # TODO: temporary fix to slumkit issue
    if _is_nested(input_channel_indices):
        input_channel_indices = [
            int(x) for x in input_channel_indices if x.isdigit()
        ]
    if _is_nested(output_channel_indices):
        output_channel_indices = [
            int(x) for x in output_channel_indices if x.isdigit()
        ]

    # Check if t_idx_in should be added to the func kwargs
    # This is needed when a different processing is needed for each time point, for example during stabilization
    all_func_params = inspect.signature(func).parameters.keys()
    if "t_idx_in" in all_func_params:
        kwargs["t_idx_in"] = t_idx_in

    # Process CZYX given with the given indeces
    # if input_channel_indices is not None and len(input_channel_indices) > 0:
    click.echo(f"Processing t={t_idx_in} and channels {input_channel_indices}")
    czyx_data = position.data.oindex[t_idx_in, input_channel_indices]
    if not _check_nan_n_zeros(czyx_data):
        transformed_czyx = func(czyx_data, **kwargs)
        # Write to file
        with open_ome_zarr(output_path, mode="r+") as output_dataset:
            output_dataset[0].oindex[
                t_idx_out, output_channel_indices
            ] = transformed_czyx
        click.echo(
            f"Finished Writing.. t={t_idx_in} and channel output={output_channel_indices}"
        )
    else:
        click.echo(f"Skipping t={t_idx_in} due to all zeros or nans")


# TODO: modify how we get the time and channesl like recOrder (isinstance(input, list) or instance(input,int) or all)
def process_single_position(
    func,
    input_data_path: Path,
    output_path: Path,
    time_indices_in: list = "all",
    time_indices_out: list = [],
    input_channel_idx: list = [],
    output_channel_idx: list = [],
    num_processes: int = mp.cpu_count(),
    **kwargs,
) -> None:
    """
    Register a single position with multiprocessing parallelization over T and C

    Parameters
    ----------
    func : CZYX -> CZYX function
        The function to be applied to the data
    input_data_path : Path
        The path to input position
    output_path : Path
        The path to output OME-Zarr Store
    time_indices_in : list
        The time indices to process.
        Must match time_indices_out if not "all"
    time_indices_out : list
        The time indices to write to.
        Must match time_indices_in if not empty.
        Typically used for stabilization, which needs a per timepoint processing
    input_channel_idx : list
        The channel indices to process.
        If empty list, process all channels.
        Must match output_channel_idx if not empty
    output_channel_idx : list
        The channel indices to write to.
        If empty list, write to all channels.
        Must match input_channel_idx if not empty
    num_processes : int
        Number of simulatenous processes per position
    kwargs : dict
        Additional arguments to pass to the CZYX function

    Usage:
    ------
    Multiprocessing over T and C:
    - input_channel_idx and output_channel_idx should be empty.
    Multiprocessing over T only:
    - input_channel_idx and output_channel_idx should be provided.
    """
    # Function to be applied
    click.echo(f"Function to be applied: \t{func}")

    # Get the reader and writer
    click.echo(f"Input data path:\t{input_data_path}")
    click.echo(f"Output data path:\t{str(output_path)}")
    input_dataset = open_ome_zarr(str(input_data_path))
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
                f"time_indices_in = {time_indices_in} includes a time index beyond the maximum index of the dataset = {time_ubound}"
            )
        # Handle the case when time_indices out is not provided. It defaults to the t_indices_in
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
        with open_ome_zarr(output_path, mode="r+") as output_dataset:
            for params_metadata_keys in kwargs["extra_metadata"].keys():
                output_dataset.zattrs["extra_metadata"] = non_func_args[
                    "extra_metadata"
                ]

    # Loop through (T, C), deskewing and writing as we go
    click.echo(f"\nStarting multiprocess pool with {num_processes} processes")

    if input_channel_idx is None or len(input_channel_idx) == 0:
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
            output_path / Path(*input_data_path.parts[-3:]),
            **func_args,
        )
    else:
        # If C is empty, use only the range for time_indices_in
        iterable = list(zip(time_indices_in, time_indices_out))
        partial_apply_transform_to_zyx_and_save = partial(
            apply_transform_to_zyx_and_save,
            func,
            input_dataset,
            output_path / Path(*input_data_path.parts[-3:]),
            input_channel_idx,
            output_channel_idx,
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

    NOTE: this function was created for a bug in slumkit that nested input_channel_indices into a list of lists
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
