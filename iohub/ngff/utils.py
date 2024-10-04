import inspect
import itertools
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Any, Callable, Tuple, Union

import click
import numpy as np
from numpy.typing import DTypeLike, NDArray

from iohub.ngff import open_ome_zarr
from iohub.ngff.nodes import TransformationMeta


def create_empty_plate(
    store_path: Path,
    position_keys: list[Tuple[str]],
    channel_names: list[str],
    shape: Tuple[int],
    chunks: Tuple[int] = None,
    scale: Tuple[float] = (1, 1, 1, 1, 1),
    dtype: DTypeLike = np.float32,
    max_chunk_size_bytes: float = 500e6,
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
        TCZYX chunk size of the plate. If None, the chunk size is calculated
        based on the shape to be less than max_chunk_size_bytes.
        Defaults to None.
    scale : Tuple[float], optional
        TCZYX scale of the plate. Defaults to (1, 1, 1, 1, 1).
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

    # Limiting the chunking to 500MB
    if chunks is None:
        chunk_zyx_shape = _calculate_zyx_chunk_size(
            shape,
            np.dtype(dtype).itemsize,
            max_chunk_size_bytes,
        )

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


def apply_transform_to_czyx_and_save(
    func: Callable[[NDArray, Any], NDArray],
    input_position_path: Path,
    output_position_path: Path,
    input_channel_indices: Union[list[int], slice],
    output_channel_indices: Union[list[int], slice],
    input_time_index: int,
    output_time_index: int,
    **kwargs,
) -> None:
    """
    Load a CZYX array from a position store,
    apply a transformation, and save the result.

    Parameters
    ----------
    func : Callable[[NDArray, Any], NDArray]
        The function to be applied to the data.
        func must take the CZYX NDArray as the first argument and return
        a CZXY NDArray. Additional arguments are passed through **kwargs.
    input_position_path : Path
        The path to input OME-Zarr position store
        (e.g., input_store_path.zarr/A/1/0).
    output_position_path : Path
        The path to output OME-Zarr position store
        (e.g., output_store_path.zarr/A/1/0).
    input_channel_indices : Union[list[int], slice]
        The channel indices to process. Acceptable values:
        - Slices: slice(0, 2).
        - A list of integers: [0, 1, 2, 3, 4].
    output_channel_indices : Union[list[int], slice]
        The channel indices to write to. Acceptable values:
        - Slices: slice(0, 2).
        - A list of integers: [0, 1, 2, 3, 4].
    input_time_index : int
        The time index to process.
    output_time_index : int
        The time index to write to.
    kwargs : dict, optional
        Additional arguments to pass to the function.

    Examples
    --------
    Using slices for input_channel_indices:
    apply_transform_to_zyx_and_save(
        func=some_function,
        input_position_path=Path("/path/to/input.zarr/A/1/0"),
        output_position_path=Path("/path/to/output.zarr/A/1/0"),
        input_channel_indices=slice(0, 2),
        output_channel_indices=[0],
        input_time_index=0,
        output_time_index=0,
    )

    Using list for input_channel_indices:
    apply_transform_to_zyx_and_save(
        func=some_function,
        input_position_path=Path("/path/to/input.zarr/A/1/0"),
        output_store_path=Path("/path/to/output.zarr/A/1/0"),
        input_channel_indices=[0, 1, 2, 3, 4],
        output_channel_indices=[0, 1, 2],
        input_time_index=0,
        output_time_index=0,
    )

    """
    # Check if input_time_indices should be added to the func kwargs
    # This is needed when a different processing is needed for each time point,
    # for example during stabilization
    all_func_params = inspect.signature(func).parameters.keys()
    if "input_time_index" in all_func_params:
        kwargs["input_time_index"] = input_time_index

    # Process CZYX given with the given indices
    # if input_channel_indices is not None and len(input_channel_indices) > 0:
    click.echo(f"Processing t={input_time_index}, c={input_channel_indices}")
    input_dataset = open_ome_zarr(input_position_path)
    czyx_data = input_dataset.data.oindex[
        input_time_index, input_channel_indices
    ]
    if not _check_nan_n_zeros(czyx_data):
        transformed_czyx = func(czyx_data, **kwargs)
        # Write to file
        with open_ome_zarr(output_position_path, mode="r+") as output_dataset:
            output_dataset[0].oindex[
                output_time_index, output_channel_indices
            ] = transformed_czyx
        click.echo(
            f"Finished t={input_time_index}, c={output_channel_indices}"
        )
    else:
        click.echo(
            f"Skipping t={input_time_index}, c={output_channel_indices}"
            "due to all zeros or nans"
        )


def process_single_position(
    func: Callable[[NDArray, Any], NDArray],
    input_position_path: Path,
    output_position_path: Path,
    input_channel_indices: Union[list[slice], list[list[int]]] = None,
    output_channel_indices: Union[list[slice], list[list[int]]] = None,
    input_time_indices: list[int] = None,
    output_time_indices: list[int] = None,
    num_processes: int = 1,
    **kwargs,
) -> None:
    """
    Apply function to data in an `iohub` position store,
    parallelizing over time and channel indices,
    and save result in an output position store.

    Parameters
    ----------
    func : Callable[[NDArray, Any], NDArray]
        The function to be applied to the data.
        func must take the CZYX NDArray as the first argument and return
        a CZXY NDArray. Additional arguments are passed through **kwargs.
    input_position_path : Path
        The path to the input OME-Zarr position store
        (e.g., input_store_path.zarr/A/1/0).
    output_position_path : Path
        The path to the output OME-Zarr position store
        (e.g., output_store_path.zarr/A/1/0).
    input_time_indices : list[int], optional
        If not provided, all timepoints will be processed.
    output_time_indices : list[int], optional
        The time indices to write to. Must match length of input_time_indices.
        Typically used for stabilization, which needs per timepoint processing.
        If not provided, input_time_indices will be used.
    input_channel_indices : Union[list[slice], list[list[int]]], optional
        The channel indices to process. Acceptable values:
        - A list of slices: [slice(0, 2), slice(2, 4), ...].
        - A list of lists of integers: [[0, 1, 2, 3, 4]].
        If empty, process all channels.
        Must match output_channel_indices if not empty.
        Defaults to an empty list.
    output_channel_indices : Union[list[slice], list[list[int]]], optional
        The channel indices to write to. Acceptable values:
        - A list of slices: [slice(0, 2), slice(2, 4), ...].
        - A list of lists of integers: [[0, 1, 2, 3, 4]].
        If empty, write to all channels.
        Must match input_channel_indices if not empty.
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
    Using slices for input_channel_indices:
    process_single_position(
        func=some_function,
        input_position_path=Path("/path/to/input"),
        output_store_path=Path("/path/to/output"),
        input_time_indices=[0],
        input_channel_indices=[slice(0, 2), slice(2, 4)],
        output_channel_indices=[[0], [1]],
    )

    Using list of lists for input_channel_indices:
    process_single_position(
        func=some_function,
        input_position_path=Path("/path/to/input"),
        output_store_path=Path("/path/to/output"),
        input_time_indices=[1, 2],
        input_channel_indices=[[0, 1], [2, 3]],
        output_channel_indices=[[0], [1]],
    )

    Notes
    -----
    - Multiprocessing over T and C:
    input_channel_indices and output_channel_indices should be empty.
    - Multiprocessing over T only:
    input_channel_indices and output_channel_indices should be provided.
    """
    # Function to be applied
    click.echo(f"Function to be applied: \t{func}")
    click.echo(f"Input data path:\t{input_position_path}")
    click.echo(f"Output data path:\t{output_position_path}")

    # Get the reader
    with open_ome_zarr(input_position_path) as input_dataset:
        input_data_shape = input_dataset.data.shape

    # Process time indices
    if input_time_indices is None:
        input_time_indices = list(range(input_data_shape[0]))
        output_time_indices = input_time_indices
    assert (
        type(input_time_indices) is list
    ), "input_time_indices must be a list"
    if output_time_indices is None:
        output_time_indices = input_time_indices

    # Process channel indices
    if input_channel_indices is None:
        input_channel_indices = [[c] for c in range(input_data_shape[1])]
        output_channel_indices = input_channel_indices
    assert (
        type(input_channel_indices) is list
    ), "input_channel_indices must be a list"
    if output_channel_indices is None:
        output_channel_indices = input_channel_indices

    # Check for invalid times
    time_ubound = input_data_shape[0] - 1
    if np.max(input_time_indices) > time_ubound:
        raise ValueError(
            f"""input_time_indices = {input_time_indices} includes
            a time index beyond the maximum index of
            the dataset = {time_ubound}"""
        )

    # Write extra metadata to the output store
    extra_metadata = kwargs.pop("extra_metadata", None)
    with open_ome_zarr(output_position_path, mode="r+") as output_dataset:
        output_dataset.zattrs["extra_metadata"] = extra_metadata

    # Loop through (T, C), applying transform and writing as we go
    iterable = itertools.product(
        zip(input_channel_indices, output_channel_indices),
        zip(input_time_indices, output_time_indices),
    )
    flat_iterable = ((*c, *t) for c, t in iterable)

    partial_apply_transform_to_czyx_and_save = partial(
        apply_transform_to_czyx_and_save,
        func,
        input_position_path,
        output_position_path,
        **kwargs,
    )

    click.echo(
        f"\nStarting multiprocess pool with\
    {num_processes} processes"
    )
    with mp.Pool(num_processes) as p:
        p.starmap(
            partial_apply_transform_to_czyx_and_save,
            flat_iterable,
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


def _calculate_zyx_chunk_size(shape, bytes_per_pixel, max_chunk_size_bytes):
    """
    Calculate the chunk size for ZYX dimensions based on the shape,
    bytes per pixel of data, and desired max chunk size.
    """

    chunk_zyx_shape = list(shape[-3:])

    # while XY image is larger than MAX_CHUNK_SIZE
    while (
        chunk_zyx_shape[-3] > 1
        and np.prod(chunk_zyx_shape) * bytes_per_pixel > max_chunk_size_bytes
    ):
        chunk_zyx_shape[-3] = np.ceil(chunk_zyx_shape[-3] / 2).astype(int)
    chunk_zyx_shape = tuple(chunk_zyx_shape)
    return chunk_zyx_shape
