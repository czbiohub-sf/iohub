import itertools
import string
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given, settings
from numpy.typing import DTypeLike

from iohub.core.compat import V04_MAX_CHUNK_SIZE_BYTES
from iohub.ngff import open_ome_zarr
from iohub.ngff.utils import (
    _indices_to_shard_aligned_batches,
    _match_indices_to_batches,
    _V05_DEFAULT_ZYX_CHUNKS,
    apply_transform_to_tczyx_and_save,
    create_empty_plate,
    process_single_position,
)


@contextmanager
def _temp_ome_zarr(
    store_name: str,
    position_keys: list[tuple[str, str, str]],
    channel_names: list[str],
    shape: tuple[int, ...],
    chunks: tuple[int, ...] | None = None,
    scale: tuple[float, ...] = (1, 1, 1, 1, 1),
    dtype: DTypeLike = np.float32,
    base_dir: Path | None = None,  # Added base_dir parameter
    version: Literal["0.4", "0.5"] = "0.5",
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
    version : Literal["0.4", "0.5"], optional
        OME-Zarr version, by default "0.4".

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
                version=version,
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
            version=version,
        )
        yield store_path


@contextmanager
def _temp_ome_zarr_stores(
    position_keys: list[tuple[str, str, str]],
    channel_names: list[str],
    shape: tuple[int, ...],
    chunks: tuple[int, ...] | None = None,
    shards_ratio: tuple[int, ...] | None = None,
    scale: tuple[float, ...] = (1, 1, 1, 1, 1),
    dtype: DTypeLike = np.float32,
    version: Literal["0.4", "0.5"] = "0.5",
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
    shape : tuple[int, ...]
        TCZYX shape of the plate.
    chunks : tuple[int, ...], optional
        TCZYX chunk size, by default None.
    shards_ratio : tuple[int, ...], optional
        Sharding ratio, by default None.
    scale : tuple[float, ...], optional
        TCZYX scale of the plate, by default (1, 1, 1, 1, 1).
    dtype : DTypeLike, optional
        Data type of the plate, by default np.float32.
    version : Literal["0.4", "0.5"], optional
        OME-Zarr version, by default "0.4".

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
            version=version,
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
                version=version,
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
                st.text(alphabet=alphanum, min_size=1, max_size=3),  # Field of View
            ),
            min_size=1,
            max_size=3,
        )
    )

    # Generate number of channels
    num_channels = draw(st.integers(min_value=1, max_value=3))

    # Generate channel names based on the number of channels
    channel_names = [f"Channel_{i}" for i in range(num_channels)]

    version = draw(st.just("0.5"))

    # Generate shape ensuring that the
    # second dimension (C) matches num_channels
    T = draw(st.integers(min_value=1, max_value=3))  # Time
    Z = draw(st.integers(min_value=1, max_value=3))  # Z-slices
    Y = draw(st.integers(min_value=8, max_value=32))  # Y-dimension
    X = draw(st.integers(min_value=8, max_value=32))  # X-dimension
    shape = (T, num_channels, Z, Y, X)  # TCZYX

    if version == "0.5":
        shards_ratio = draw(st.one_of(st.just((2, 1, 1, 2, 2)), st.just(None)))
    else:
        shards_ratio = None

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

    return (
        position_keys,
        channel_names,
        shape,
        chunks,
        shards_ratio,
        scale,
        dtype,
        version,
    )


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
    (
        position_keys,
        channel_names,
        shape,
        chunks,
        shards_ratio,
        scale,
        dtype,
        version,
    ) = draw(plate_setup())
    T, C = shape[:2]

    # Define a helper strategy to generate channel indices based on C.
    # Integer lists are drawn ``unique=True`` and sorted: zarrs only accelerates
    # monotonically-increasing unique oindex selectors and falls back to the
    # buggy BatchedCodecPipeline (zarr-python#2834 / iohub#404) for duplicate
    # or unsorted inputs.
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
            unique=True,
        ).map(sorted),
    )

    time_indices_strategy = st.one_of(
        st.lists(
            st.integers(min_value=0, max_value=T - 1),
            min_size=1,
            max_size=min(3, T),
            unique=True,
        ).map(sorted),
    )

    # Generate input and output channel indices based on C
    channel_indices = draw(channel_indices_strategy)
    time_indices = draw(time_indices_strategy)

    return (
        position_keys,
        channel_names,
        shape,
        chunks,
        shards_ratio,
        scale,
        dtype,
        channel_indices,
        time_indices,
        version,
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
    (
        position_keys,
        channel_names,
        shape,
        chunks,
        shards_ratio,
        scale,
        dtype,
        version,
    ) = draw(plate_setup())
    # NOTE: Chunking along T,C =1,1
    if chunks is not None:
        chunks = (1, 1, *chunks[2:])

    T, C = shape[:2]

    # Define a helper strategy to generate channel indices based on C.
    # Integer lists are drawn ``unique=True`` and sorted: zarrs only accelerates
    # monotonically-increasing unique oindex selectors and falls back to the
    # buggy BatchedCodecPipeline (zarr-python#2834 / iohub#404) for duplicate
    # or unsorted inputs.
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
                unique=True,
            ).map(sorted),
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
            unique=True,
        ).map(sorted),
    )

    # Generate input and output channel indices based on C
    channel_indices = draw(channel_indices_strategy)
    time_indices = draw(time_indices_strategy)

    return (
        position_keys,
        channel_names,
        shape,
        chunks,
        shards_ratio,
        scale,
        dtype,
        channel_indices,
        time_indices,
        version,
    )


# Define the transformation function
def dummy_transform(data, constant=2):
    return data * constant


# Populate the input store with random data
def populate_store(
    input_store_path: Path,
    position_keys: list[tuple[str, str, str]],
    shape: tuple[int, ...],
    dtype: DTypeLike,
):
    with open_ome_zarr(input_store_path, mode="r+") as input_dataset:
        for position_key_tuple in position_keys:
            position_path = "/".join(position_key_tuple)
            position = input_dataset[position_path]
            _T, _C, _Z, _Y, _X = shape
            # Generate random data based on dtype
            if np.issubdtype(dtype, np.floating):
                data = np.random.default_rng().random(shape).astype(dtype)
            else:
                data = np.random.default_rng().integers(1, 20, size=shape, dtype=dtype)
            position.data[:] = data


# Verify the transformation
def verify_transformation(
    input_store_path: Path,
    output_store_path: Path,
    position_key_tuple: tuple[str, str, str],
    shape: tuple[int, ...],
    time_indices: list[int],
    channel_indices: list[int],
    transform_func,
    **kwargs,
):
    with (
        open_ome_zarr(input_store_path) as input_dataset,
        open_ome_zarr(output_store_path) as output_dataset,
    ):
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
        output_data = output_position.data.oindex[time_indices, channel_indices]
        expected_data = transform_func(input_data, **kwargs)

        np.testing.assert_array_almost_equal(
            output_data,
            expected_data,
            err_msg=f"Mismatch in position {position_key_tuple}",
        )


@given(
    plate_setup=plate_setup(),
    extra_channels=st.lists(st.text(min_size=5, max_size=16), min_size=1, max_size=3),
)
@settings(max_examples=5)
def test_create_empty_plate(plate_setup, extra_channels):
    (
        position_keys,
        channel_names,
        shape,
        chunks,
        shards_ratio,
        scale,
        dtype,
        version,
    ) = plate_setup
    assume(len(set(extra_channels)) == len(extra_channels))
    assume(not any(c in channel_names for c in extra_channels))

    with TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "test.zarr"

        # Call the function under test
        create_empty_plate(
            store_path=store_path,
            position_keys=position_keys,
            channel_names=channel_names,
            shape=shape,
            chunks=chunks,
            shards_ratio=shards_ratio,
            scale=scale,
            dtype=dtype,
            version=version,
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
                    assert position.data.chunks == (1, 1, *tuple(shape[-3:]))

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
            shards_ratio=shards_ratio,
            scale=scale,
            dtype=dtype,
            version=version,
        )

        with open_ome_zarr(store_path) as dataset:
            assert dataset.channel_names == (channel_names + extra_channels)
            shape = (shape[0], shape[1] + len(extra_channels), *shape[2:])
            for position_key_tuple in position_keys:
                position_path = "/".join(position_key_tuple)
                position = dataset[position_path]
                assert position.data.shape == shape


@given(
    setup=apply_transform_czyx_setup(),
    constant=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=5, deadline=None)
def test_apply_transform_to_czyx_and_save(setup, constant):
    (
        position_keys,
        channel_names,
        shape,
        chunks,
        shards_ratio,
        scale,
        dtype,
        channel_indices,
        time_indices,
        version,
    ) = setup
    assume(shards_ratio is None)

    # Use the enhanced context manager to get both input and output store paths
    with _temp_ome_zarr_stores(
        position_keys=position_keys,
        channel_names=channel_names,
        shape=shape,
        chunks=chunks,
        shards_ratio=shards_ratio,
        scale=scale,
        dtype=dtype,
        version=version,
    ) as (input_store_path, output_store_path):
        # Populate the input store with random data
        populate_store(input_store_path, position_keys, shape, dtype)

        kwargs = {"constant": constant}

        # Apply the transformation for each position and time point
        for position_key_tuple in position_keys:
            input_position_path = input_store_path / Path(*position_key_tuple)
            output_position_path = output_store_path / Path(*position_key_tuple)

            for t_in in time_indices:
                apply_transform_to_tczyx_and_save(
                    func=dummy_transform,
                    input_position_path=Path(input_position_path),
                    output_position_path=Path(output_position_path),
                    input_channel_indices=channel_indices,
                    output_channel_indices=channel_indices,
                    input_time_indices=[t_in],
                    output_time_indices=[t_in],
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
    setup=apply_transform_czyx_setup(),
    constant=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=5, deadline=None)
def test_apply_transform_to_tczyx_and_save(setup, constant):
    (
        position_keys,
        channel_names,
        shape,
        chunks,
        shards_ratio,
        scale,
        dtype,
        channel_indices,
        time_indices,
        version,
    ) = setup

    # Use the enhanced context manager to get both input and output store paths
    with _temp_ome_zarr_stores(
        position_keys=position_keys,
        channel_names=channel_names,
        shape=shape,
        chunks=chunks,
        shards_ratio=shards_ratio,
        scale=scale,
        dtype=dtype,
        version=version,
    ) as (input_store_path, output_store_path):
        # Populate the input store with random data
        populate_store(input_store_path, position_keys, shape, dtype)

        kwargs = {"constant": constant}

        # Apply the transformation for each position and time point
        for position_key_tuple in position_keys:
            input_position_path = input_store_path / Path(*position_key_tuple)
            output_position_path = output_store_path / Path(*position_key_tuple)

            apply_transform_to_tczyx_and_save(
                func=dummy_transform,
                input_position_path=Path(input_position_path),
                output_position_path=Path(output_position_path),
                input_channel_indices=channel_indices,
                output_channel_indices=channel_indices,
                input_time_indices=time_indices,
                output_time_indices=time_indices,
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
    indices=st.lists(st.integers(min_value=0), min_size=1, unique=True),
    shard_size=st.integers(min_value=1),
)
def test_indices_to_shard_aligned_batches(indices, shard_size):
    """Test ``_indices_to_shard_aligned_batches``"""
    batches = _indices_to_shard_aligned_batches(indices, shard_size)
    assert isinstance(batches, list)
    elements = []
    for batch in batches:
        assert batch
        assert isinstance(batch, list)
        elements.extend(batch)
        first_element = batch[0]
        shard_index = first_element // shard_size
        lower_bound = shard_index * shard_size
        upper_bound = lower_bound + shard_size
        for item in batch:
            assert isinstance(item, int)
            assert lower_bound <= item < upper_bound, batches
    assert elements == sorted(indices)


@given(
    indices=st.lists(st.integers(min_value=0), min_size=1, unique=True),
    shard_size=st.integers(min_value=1),
)
def test_match_indices_to_batches(indices, shard_size):
    """Test ``_match_indices_to_batches``"""
    batched_reference = _indices_to_shard_aligned_batches(indices, shard_size)
    matched_batches = _match_indices_to_batches(
        flat_indices=indices,
        original_reference=indices,
        batched_reference=batched_reference,
    )
    assert matched_batches == batched_reference


@given(
    setup=process_single_position_setup(),
    constant=st.integers(min_value=1, max_value=3),
    num_workers=st.sampled_from([1, 2]),
    use_threads=st.booleans(),
)
@settings(max_examples=3, deadline=None)
def test_process_single_position(setup, constant, num_workers, use_threads):
    (
        position_keys,
        channel_names,
        shape,
        chunks,
        shards_ratio,
        scale,
        dtype,
        channel_indices,
        time_indices,
        version,
    ) = setup

    with _temp_ome_zarr_stores(
        position_keys=position_keys,
        channel_names=channel_names,
        shape=shape,
        chunks=chunks,
        shards_ratio=shards_ratio,
        scale=scale,
        dtype=dtype,
        version=version,
    ) as (input_store_path, output_store_path):
        populate_store(input_store_path, position_keys, shape, dtype)

        for position_key_tuple in position_keys:
            input_position_path = input_store_path / Path(*position_key_tuple)
            output_position_path = output_store_path / Path(*position_key_tuple)
            kwargs = {"constant": constant, "extra_metadata": {"temp": 10}}

            process_single_position(
                func=dummy_transform,
                input_position_path=input_position_path,
                output_position_path=output_position_path,
                input_channel_indices=channel_indices,
                output_channel_indices=channel_indices,
                input_time_indices=time_indices,
                output_time_indices=time_indices,
                num_workers=num_workers,
                use_threads=use_threads,
                **kwargs,
            )

            if time_indices is None:
                time_indices = list(range(shape[0]))
            if channel_indices is None:
                channel_indices = [[c] for c in range(shape[1])]

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


# -- Explicit tests for version-specific chunk/shard defaults -----------------
#
# The hypothesis-based test_create_empty_plate exercises many parameter
# combinations but does not assert the exact defaults the issue #401 spec
# prescribes. These tests pin those defaults down so CI fails deterministically
# if they regress, rather than relying on a favorable hypothesis draw.


def _open_array(store_path: Path, position_key: tuple[str, str, str]):
    return open_ome_zarr(store_path)["/".join(position_key)].data


@pytest.mark.parametrize(
    ("shape", "expected_chunks"),
    [
        # Large shape: chunks clamped to DCA spec (16, 256, 256).
        ((2, 2, 64, 1024, 1024), (1, 1, 16, 256, 256)),
        # Small Z: clamped to Z.
        ((2, 2, 8, 1024, 1024), (1, 1, 8, 256, 256)),
        # Small YX: clamped to YX.
        ((2, 2, 64, 128, 200), (1, 1, 16, 128, 200)),
        # Fully smaller than defaults.
        ((1, 1, 4, 32, 32), (1, 1, 4, 32, 32)),
    ],
)
def test_v05_default_chunks(tmp_path, shape, expected_chunks):
    """v0.5 default chunks are DCA-aligned (16, 256, 256), clamped to shape."""
    store = tmp_path / "test.zarr"
    create_empty_plate(
        store_path=store,
        position_keys=[("A", "1", "0")],
        channel_names=[f"c{i}" for i in range(shape[1])],
        shape=shape,
        version="0.5",
    )
    arr = _open_array(store, ("A", "1", "0"))
    assert arr.chunks == expected_chunks


def test_v05_default_shards_cover_zyx(tmp_path):
    """v0.5 default shards have shape (1, 1, Z, Y, X) — one shard per (T, C)."""
    shape = (3, 2, 64, 1024, 1024)
    store = tmp_path / "test.zarr"
    create_empty_plate(
        store_path=store,
        position_keys=[("A", "1", "0")],
        channel_names=[f"c{i}" for i in range(shape[1])],
        shape=shape,
        version="0.5",
    )
    arr = _open_array(store, ("A", "1", "0"))
    # Shard spans one full (Z, Y, X) volume per (T, C) slot.
    assert arr.shards == (1, 1, shape[2], shape[3], shape[4])
    # And chunks stay DCA-aligned.
    assert arr.chunks == (1, 1, *_V05_DEFAULT_ZYX_CHUNKS)


def test_v05_default_shards_with_non_divisible_zyx(tmp_path):
    """Shards still cover the full (Z, Y, X) even when dims are not multiples of chunks."""
    # Z=20, Y=300, X=300 — none divide evenly into (16, 256, 256).
    shape = (1, 1, 20, 300, 300)
    store = tmp_path / "test.zarr"
    create_empty_plate(
        store_path=store,
        position_keys=[("A", "1", "0")],
        channel_names=["c0"],
        shape=shape,
        version="0.5",
    )
    arr = _open_array(store, ("A", "1", "0"))
    # Shard = chunk * ceil(dim/chunk) — must be >= dim along each axis.
    assert arr.chunks == (1, 1, 16, 256, 256)
    assert arr.shards[0] == 1
    assert arr.shards[1] == 1
    assert arr.shards[2] >= 20
    assert arr.shards[3] >= 300
    assert arr.shards[4] >= 300


def test_v05_explicit_shards_ratio_is_honored(tmp_path):
    """An explicit shards_ratio overrides the default."""
    shape = (4, 2, 16, 256, 256)
    store = tmp_path / "test.zarr"
    create_empty_plate(
        store_path=store,
        position_keys=[("A", "1", "0")],
        channel_names=[f"c{i}" for i in range(shape[1])],
        shape=shape,
        chunks=(1, 1, 16, 256, 256),
        shards_ratio=(2, 2, 1, 1, 1),
        version="0.5",
    )
    arr = _open_array(store, ("A", "1", "0"))
    assert arr.chunks == (1, 1, 16, 256, 256)
    assert arr.shards == (2, 2, 16, 256, 256)


def test_v04_default_chunks_cover_full_zyx(tmp_path):
    """v0.4 default chunks are (1, 1, Z, Y, X) when under the byte cap."""
    shape = (2, 2, 4, 64, 64)
    store = tmp_path / "test.zarr"
    create_empty_plate(
        store_path=store,
        position_keys=[("A", "1", "0")],
        channel_names=[f"c{i}" for i in range(shape[1])],
        shape=shape,
        version="0.4",
    )
    arr = _open_array(store, ("A", "1", "0"))
    assert arr.chunks == (1, 1, shape[2], shape[3], shape[4])


def test_v04_default_chunks_capped_by_byte_limit(tmp_path):
    """v0.4 chunks halve Z until the chunk fits under V04_MAX_CHUNK_SIZE_BYTES."""
    # Pick a shape whose single (Z, Y, X) volume in float32 exceeds the cap.
    # float32 is 4 bytes; cap is 500 MB → a (256, 1024, 1024) volume is
    # 1 GiB, so the default must halve Z at least once.
    shape = (1, 1, 256, 1024, 1024)
    dtype = np.float32
    store = tmp_path / "test.zarr"
    create_empty_plate(
        store_path=store,
        position_keys=[("A", "1", "0")],
        channel_names=["c0"],
        shape=shape,
        dtype=dtype,
        version="0.4",
    )
    arr = _open_array(store, ("A", "1", "0"))
    t_chunk, c_chunk, z_chunk, y_chunk, x_chunk = arr.chunks
    assert (t_chunk, c_chunk) == (1, 1)
    assert (y_chunk, x_chunk) == (shape[3], shape[4])
    assert z_chunk < shape[2], "Z should have been halved to respect byte cap"
    bytes_per_chunk = z_chunk * y_chunk * x_chunk * np.dtype(dtype).itemsize
    assert bytes_per_chunk <= V04_MAX_CHUNK_SIZE_BYTES


def test_v04_default_has_no_sharding(tmp_path):
    """v0.4 (Zarr v2) never has a sharding codec, regardless of the new defaults."""
    store = tmp_path / "test.zarr"
    create_empty_plate(
        store_path=store,
        position_keys=[("A", "1", "0")],
        channel_names=["c0"],
        shape=(2, 1, 8, 64, 64),
        version="0.4",
    )
    arr = _open_array(store, ("A", "1", "0"))
    assert arr.shards is None


def test_v04_rejects_explicit_shards_ratio(tmp_path):
    """Passing shards_ratio on a v0.4 store raises (Zarr v2 has no sharding)."""
    store = tmp_path / "test.zarr"
    with pytest.raises(ValueError, match="Sharding is not supported in Zarr v2"):
        create_empty_plate(
            store_path=store,
            position_keys=[("A", "1", "0")],
            channel_names=["c0"],
            shape=(2, 1, 8, 64, 64),
            shards_ratio=(1, 1, 1, 1, 1),
            version="0.4",
        )


# -- Write path on sharded v0.5 stores ---------------------------------------
#
# Round-trip writes into v0.5 stores with non-trivial shard layouts,
# exercising both the #401 default (shard per (T, C) slot) and a
# channel-spanning shard that groups multiple channels into one shard.


def test_process_single_position_on_sharded_v05_store(tmp_path):
    """process_single_position writes to a default-sharded v0.5 store correctly."""
    shape = (2, 1, 4, 16, 16)
    position_key = ("A", "1", "0")
    input_store = tmp_path / "input.zarr"
    output_store = tmp_path / "output.zarr"
    for store in (input_store, output_store):
        create_empty_plate(
            store_path=store,
            position_keys=[position_key],
            channel_names=["c0"],
            shape=shape,
            version="0.5",
        )
    populate_store(input_store, [position_key], shape, np.float32)

    process_single_position(
        func=dummy_transform,
        input_position_path=input_store / Path(*position_key),
        output_position_path=output_store / Path(*position_key),
        input_channel_indices=[[0]],
        output_channel_indices=[[0]],
        input_time_indices=[0, 1],
        output_time_indices=[0, 1],
        constant=2,
    )

    out_arr = _open_array(output_store, position_key)
    assert out_arr.shards == (1, 1, shape[2], shape[3], shape[4])

    with open_ome_zarr(input_store) as in_ds, open_ome_zarr(output_store) as out_ds:
        in_data = in_ds["/".join(position_key)].data[:]
        out_data = out_ds["/".join(position_key)].data[:]
    np.testing.assert_array_almost_equal(out_data, dummy_transform(in_data, constant=2))


def test_apply_transform_to_tczyx_on_multi_channel_shard(tmp_path):
    """Multi-channel oindex write into a shard that spans multiple C slots.

    Grouping channels within a single shard is a common layout for
    stores produced downstream (e.g. multi-channel stitched outputs);
    ``apply_transform_to_tczyx_and_save`` should round-trip correctly
    when the write addresses both channels of a single shard in one call.
    """
    shape = (1, 4, 4, 16, 16)
    shards_ratio = (1, 2, 1, 1, 1)  # shard_c = 2 -> one write spans two C slots
    position_key = ("A", "1", "0")
    input_store = tmp_path / "input.zarr"
    output_store = tmp_path / "output.zarr"
    for store in (input_store, output_store):
        create_empty_plate(
            store_path=store,
            position_keys=[position_key],
            channel_names=[f"c{i}" for i in range(shape[1])],
            shape=shape,
            chunks=(1, 1, 4, 16, 16),
            shards_ratio=shards_ratio,
            version="0.5",
        )
    populate_store(input_store, [position_key], shape, np.float32)

    apply_transform_to_tczyx_and_save(
        func=dummy_transform,
        input_position_path=input_store / Path(*position_key),
        output_position_path=output_store / Path(*position_key),
        input_channel_indices=[0, 1],
        output_channel_indices=[0, 1],
        input_time_indices=[0],
        output_time_indices=[0],
        constant=2,
    )

    with open_ome_zarr(input_store) as in_ds, open_ome_zarr(output_store) as out_ds:
        in_slice = in_ds["/".join(position_key)].data[:1, :2]
        out_slice = out_ds["/".join(position_key)].data[:1, :2]
    np.testing.assert_array_almost_equal(out_slice, dummy_transform(in_slice, constant=2))
