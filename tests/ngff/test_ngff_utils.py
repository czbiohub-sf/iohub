import itertools
import json
import math
import os
import shutil
import string
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import hypothesis.strategies as st
import numpy as np
import pytest
import xarray as xr
from hypothesis import assume, given, settings
from numpy.typing import DTypeLike

from iohub.core.compat import V04_MAX_CHUNK_SIZE_BYTES
from iohub.ngff import open_ome_zarr
from iohub.ngff._write_units import plan_write_unit, progress_dir_for
from iohub.ngff.models import LabelsMeta
from iohub.ngff.utils import (
    _V05_DEFAULT_ZYX_CHUNKS,
    _available_cpus,
    _contiguous_runs,
    _indices_to_shard_aligned_batches,
    _match_indices_to_batches,
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

        # Each extra_metadata entry is written as a top-level zattrs key,
        # not nested under an "extra_metadata" key.
        if extra_metadata is not None:
            for key, value in extra_metadata.items():
                assert output_position.zattrs[key] == value
            # The legacy wrapper key must not be written.
            assert "extra_metadata" not in dict(output_position.zattrs)

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


def test_create_empty_plate_copy_metadata_from():
    """Test that metadata_sources copies custom zattrs but not labels."""
    position_keys = [("A", "1", "0"), ("A", "1", "1")]
    channel_names = ["DAPI", "GFP"]
    shape = (1, 2, 32, 64, 64)
    scale = (1, 1, 0.5, 0.108, 0.108)
    custom_zattrs = {"extra_metadata": {"temperature": 37.0, "protocol": "v2"}}

    with TemporaryDirectory() as temp_dir:
        src_path = Path(temp_dir) / "source.zarr"
        dst_path = Path(temp_dir) / "dest.zarr"

        # Create source plate with custom metadata and label references
        create_empty_plate(
            store_path=src_path,
            position_keys=position_keys,
            channel_names=channel_names,
            shape=shape,
            scale=scale,
        )

        with open_ome_zarr(str(src_path), mode="r+") as plate:
            for _name, pos in plate.positions():
                for k, v in custom_zattrs.items():
                    pos.zattrs[k] = v
                # Write a label reference into the source OME metadata
                pos.metadata.labels = LabelsMeta(labels=["nuclei"])
                pos.dump_meta()

        # Create dest plate with different channel names but metadata_sources
        dst_channels = ["Phase", "Fluorescence"]
        dst_shape = (1, 2, 16, 64, 64)
        dst_scale = (1, 1, 1.0, 0.108, 0.108)
        create_empty_plate(
            store_path=dst_path,
            position_keys=position_keys,
            channel_names=dst_channels,
            shape=dst_shape,
            scale=dst_scale,
            metadata_sources=src_path,
        )

        # Verify metadata was copied
        with open_ome_zarr(str(dst_path), mode="r") as dst_plate:
            for _name, dst_pos in dst_plate.positions():
                # Custom zattrs should be copied
                assert dst_pos.zattrs["extra_metadata"] == custom_zattrs["extra_metadata"]

                # Dataset layout (shape/chunks) should be from the dest, not source
                assert dst_pos.data.shape == dst_shape

                # Scale should be the dest's scale, not the source's
                assert tuple(dst_pos.scale) == pytest.approx(dst_scale)

                # Omero channel info should be preserved (dest's channels)
                assert dst_pos.channel_names == dst_channels

                # Label references should NOT be copied (no backing arrays)
                dst_labels = getattr(dst_pos.metadata, "labels", None)
                assert dst_labels is None or dst_labels == []


def test_create_empty_plate_copy_metadata_subset_positions():
    """Test metadata_sources when dest has a subset of source positions."""
    position_keys_src = [("A", "1", "0"), ("A", "1", "1"), ("B", "1", "0")]
    position_keys_dst = [("A", "1", "0"), ("B", "1", "0")]
    channel_names = ["DAPI"]
    shape = (1, 1, 16, 32, 32)

    with TemporaryDirectory() as temp_dir:
        src_path = Path(temp_dir) / "source.zarr"
        dst_path = Path(temp_dir) / "dest.zarr"

        create_empty_plate(
            store_path=src_path,
            position_keys=position_keys_src,
            channel_names=channel_names,
            shape=shape,
        )

        # Tag source positions
        with open_ome_zarr(str(src_path), mode="r+") as plate:
            for name, pos in plate.positions():
                pos.zattrs["source_position"] = name

        create_empty_plate(
            store_path=dst_path,
            position_keys=position_keys_dst,
            channel_names=channel_names,
            shape=shape,
            metadata_sources=src_path,
        )

        with open_ome_zarr(str(dst_path), mode="r") as dst_plate:
            dst_names = {name for name, _ in dst_plate.positions()}
            assert dst_names == {"A/1/0", "B/1/0"}
            for name, pos in dst_plate.positions():
                assert pos.zattrs["source_position"] == name


def test_create_empty_plate_copy_metadata_skips_existing_positions():
    """metadata_sources only writes to newly created positions.

    Positions that already exist in the output plate must be left
    unchanged, even when the source plate has metadata for them.
    """
    position_keys = [("A", "1", "0"), ("A", "1", "1")]
    channel_names = ["DAPI"]
    shape = (1, 1, 16, 32, 32)

    with TemporaryDirectory() as temp_dir:
        src_path = Path(temp_dir) / "source.zarr"
        dst_path = Path(temp_dir) / "dest.zarr"

        # Source plate tags every position.
        create_empty_plate(
            store_path=src_path,
            position_keys=position_keys,
            channel_names=channel_names,
            shape=shape,
        )
        with open_ome_zarr(str(src_path), mode="r+") as plate:
            for _name, pos in plate.positions():
                pos.zattrs["extra_metadata"] = {"origin": "source"}

        # Pre-create A/1/0 in the destination with its own metadata.
        create_empty_plate(
            store_path=dst_path,
            position_keys=[("A", "1", "0")],
            channel_names=channel_names,
            shape=shape,
        )
        with open_ome_zarr(str(dst_path), mode="r+") as plate:
            plate["A/1/0"].zattrs["extra_metadata"] = {"origin": "dest"}

        # Append both positions with metadata_sources.
        create_empty_plate(
            store_path=dst_path,
            position_keys=position_keys,
            channel_names=channel_names,
            shape=shape,
            metadata_sources=src_path,
        )

        with open_ome_zarr(str(dst_path), mode="r") as dst_plate:
            # Pre-existing position is left untouched.
            assert dst_plate["A/1/0"].zattrs["extra_metadata"] == {"origin": "dest"}
            # Newly created position receives the source metadata.
            assert dst_plate["A/1/1"].zattrs["extra_metadata"] == {"origin": "source"}


def test_create_empty_plate_copy_metadata_missing_source_root():
    """A non-existent metadata_sources source raises FileNotFoundError."""
    with TemporaryDirectory() as temp_dir:
        dst_path = Path(temp_dir) / "dest.zarr"
        missing_src = Path(temp_dir) / "does_not_exist.zarr"

        with pytest.raises(FileNotFoundError):
            create_empty_plate(
                store_path=dst_path,
                position_keys=[("A", "1", "0")],
                channel_names=["DAPI"],
                shape=(1, 1, 16, 32, 32),
                metadata_sources=missing_src,
            )


def test_create_empty_plate_copy_metadata_position_absent_in_source():
    """Positions missing from the source are created without metadata."""
    channel_names = ["DAPI"]
    shape = (1, 1, 16, 32, 32)

    with TemporaryDirectory() as temp_dir:
        src_path = Path(temp_dir) / "source.zarr"
        dst_path = Path(temp_dir) / "dest.zarr"

        # Source only has A/1/0.
        create_empty_plate(
            store_path=src_path,
            position_keys=[("A", "1", "0")],
            channel_names=channel_names,
            shape=shape,
        )
        with open_ome_zarr(str(src_path), mode="r+") as plate:
            plate["A/1/0"].zattrs["extra_metadata"] = {"origin": "source"}

        # Dest requests A/1/0 (present) and A/1/1 (absent in source).
        create_empty_plate(
            store_path=dst_path,
            position_keys=[("A", "1", "0"), ("A", "1", "1")],
            channel_names=channel_names,
            shape=shape,
            metadata_sources=src_path,
        )

        with open_ome_zarr(str(dst_path), mode="r") as dst_plate:
            # Both positions exist with their data arrays and channels.
            dst_names = {name for name, _ in dst_plate.positions()}
            assert dst_names == {"A/1/0", "A/1/1"}
            assert dst_plate["A/1/1"].data.shape == shape
            assert dst_plate["A/1/1"].channel_names == channel_names

            # Present source position got its metadata copied.
            assert dst_plate["A/1/0"].zattrs["extra_metadata"] == {"origin": "source"}
            # Absent source position has no copied custom metadata.
            assert "extra_metadata" not in dict(dst_plate["A/1/1"].zattrs)


def test_create_empty_plate_copy_metadata_multiple_sources():
    """metadata_sources accepts a list of plates and merges their zattrs."""
    position_keys = [("A", "1", "0")]
    channel_names = ["DAPI"]
    shape = (1, 1, 16, 32, 32)

    with TemporaryDirectory() as temp_dir:
        src_a_path = Path(temp_dir) / "source_a.zarr"
        src_b_path = Path(temp_dir) / "source_b.zarr"
        dst_path = Path(temp_dir) / "dest.zarr"

        # First source contributes "from_a".
        create_empty_plate(
            store_path=src_a_path,
            position_keys=position_keys,
            channel_names=channel_names,
            shape=shape,
        )
        with open_ome_zarr(str(src_a_path), mode="r+") as plate:
            plate["A/1/0"].zattrs["from_a"] = {"origin": "a"}

        # Second source contributes a disjoint key "from_b".
        create_empty_plate(
            store_path=src_b_path,
            position_keys=position_keys,
            channel_names=channel_names,
            shape=shape,
        )
        with open_ome_zarr(str(src_b_path), mode="r+") as plate:
            plate["A/1/0"].zattrs["from_b"] = {"origin": "b"}

        create_empty_plate(
            store_path=dst_path,
            position_keys=position_keys,
            channel_names=channel_names,
            shape=shape,
            metadata_sources=[src_a_path, src_b_path],
        )

        with open_ome_zarr(str(dst_path), mode="r") as dst_plate:
            dst_zattrs = dict(dst_plate["A/1/0"].zattrs)
            # Disjoint keys from both sources are merged.
            assert dst_zattrs["from_a"] == {"origin": "a"}
            assert dst_zattrs["from_b"] == {"origin": "b"}


def test_create_empty_plate_copy_metadata_earlier_source_wins():
    """When sources share a key, the earlier source in the list wins."""
    position_keys = [("A", "1", "0")]
    channel_names = ["DAPI"]
    shape = (1, 1, 16, 32, 32)

    with TemporaryDirectory() as temp_dir:
        src_a_path = Path(temp_dir) / "source_a.zarr"
        src_b_path = Path(temp_dir) / "source_b.zarr"
        dst_path = Path(temp_dir) / "dest.zarr"

        # Both sources define "extra_metadata" with conflicting values.
        for src_path, origin in ((src_a_path, "a"), (src_b_path, "b")):
            create_empty_plate(
                store_path=src_path,
                position_keys=position_keys,
                channel_names=channel_names,
                shape=shape,
            )
            with open_ome_zarr(str(src_path), mode="r+") as plate:
                plate["A/1/0"].zattrs["extra_metadata"] = {"origin": origin}

        # source_a precedes source_b, so its value must take precedence.
        create_empty_plate(
            store_path=dst_path,
            position_keys=position_keys,
            channel_names=channel_names,
            shape=shape,
            metadata_sources=[src_a_path, src_b_path],
        )

        with open_ome_zarr(str(dst_path), mode="r") as dst_plate:
            assert dst_plate["A/1/0"].zattrs["extra_metadata"] == {"origin": "a"}


def _plate_with_zattrs(store_path, position_keys, channel_names, shape, zattrs):
    """Create a plate and stamp ``zattrs`` onto every position."""
    create_empty_plate(
        store_path=store_path,
        position_keys=position_keys,
        channel_names=channel_names,
        shape=shape,
    )
    with open_ome_zarr(str(store_path), mode="r+") as plate:
        for _name, pos in plate.positions():
            for k, v in zattrs.items():
                pos.zattrs[k] = v


def test_create_empty_plate_metadata_keys_filters_by_pattern():
    """metadata_keys selects which source zattrs are copied, via fnmatch."""
    position_keys = [("A", "1", "0")]
    channel_names = ["DAPI"]
    shape = (1, 1, 16, 32, 32)
    source_zattrs = {
        "provenance-deskew": {"angle": 30},
        "provenance-stitch": {"overlap": 0.1},
        "acquisition": {"scope": "mantis"},
        "frame_log": list(range(100)),
    }

    with TemporaryDirectory() as temp_dir:
        src_path = Path(temp_dir) / "source.zarr"
        dst_path = Path(temp_dir) / "dest.zarr"
        _plate_with_zattrs(src_path, position_keys, channel_names, shape, source_zattrs)

        create_empty_plate(
            store_path=dst_path,
            position_keys=position_keys,
            channel_names=channel_names,
            shape=shape,
            metadata_sources=src_path,
            metadata_keys={"provenance-*", "acquisition"},
        )

        with open_ome_zarr(str(dst_path), mode="r") as dst_plate:
            dst_zattrs = dict(dst_plate["A/1/0"].zattrs)
        assert dst_zattrs["provenance-deskew"] == {"angle": 30}
        assert dst_zattrs["provenance-stitch"] == {"overlap": 0.1}
        assert dst_zattrs["acquisition"] == {"scope": "mantis"}
        # Unmatched key is left behind.
        assert "frame_log" not in dst_zattrs


def test_create_empty_plate_metadata_keys_none_copies_everything():
    """The default (None) keeps the previous copy-every-non-OME-key behaviour."""
    position_keys = [("A", "1", "0")]
    channel_names = ["DAPI"]
    shape = (1, 1, 16, 32, 32)
    source_zattrs = {"alpha": 1, "beta": 2}

    with TemporaryDirectory() as temp_dir:
        src_path = Path(temp_dir) / "source.zarr"
        dst_path = Path(temp_dir) / "dest.zarr"
        _plate_with_zattrs(src_path, position_keys, channel_names, shape, source_zattrs)

        create_empty_plate(
            store_path=dst_path,
            position_keys=position_keys,
            channel_names=channel_names,
            shape=shape,
            metadata_sources=src_path,
        )

        with open_ome_zarr(str(dst_path), mode="r") as dst_plate:
            dst_zattrs = dict(dst_plate["A/1/0"].zattrs)
        assert dst_zattrs["alpha"] == 1
        assert dst_zattrs["beta"] == 2


def test_create_empty_plate_metadata_keys_without_sources_raises():
    """metadata_keys filters nothing on its own, so it is rejected alone."""
    with TemporaryDirectory() as temp_dir:
        dst_path = Path(temp_dir) / "dest.zarr"

        with pytest.raises(ValueError, match="metadata_keys"):
            create_empty_plate(
                store_path=dst_path,
                position_keys=[("A", "1", "0")],
                channel_names=["DAPI"],
                shape=(1, 1, 16, 32, 32),
                metadata_keys={"provenance-*"},
            )

        # The guard runs before anything is created.
        assert not dst_path.exists()


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


def test_process_single_position_rejects_reserved_ome_keys():
    """extra_metadata keys colliding with reserved OME-Zarr keys raise."""
    shape = (1, 1, 2, 4, 4)
    with TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "test.zarr"
        create_empty_plate(store_path, [("A", "1", "0")], ["c"], shape)
        position_path = store_path / "A" / "1" / "0"
        with pytest.raises(ValueError, match="reserved OME-Zarr"):
            process_single_position(
                func=lambda x: x,
                input_position_path=position_path,
                output_position_path=position_path,
                extra_metadata={"multiscales": {"oops": 1}},
            )


def test_process_single_position_rejects_invalid_extra_metadata():
    """Non-mapping or non-string-keyed extra_metadata raises TypeError."""
    shape = (1, 1, 2, 4, 4)
    with TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "test.zarr"
        create_empty_plate(store_path, [("A", "1", "0")], ["c"], shape)
        position_path = store_path / "A" / "1" / "0"
        with pytest.raises(TypeError, match="must be a mapping"):
            process_single_position(
                func=lambda x: x,
                input_position_path=position_path,
                output_position_path=position_path,
                extra_metadata=["not", "a", "mapping"],
            )
        with pytest.raises(TypeError, match="must be strings"):
            process_single_position(
                func=lambda x: x,
                input_position_path=position_path,
                output_position_path=position_path,
                extra_metadata={1: {"oops": 1}},
            )


def test_process_single_position_warns_on_overwrite_of_existing_zattrs_key():
    shape = (1, 1, 1, 2, 2)
    with TemporaryDirectory() as temp_dir:
        store_path = Path(temp_dir) / "test.zarr"
        create_empty_plate(store_path, [("A", "1", "0")], ["c"], shape)
        position_path = store_path / "A" / "1" / "0"

        with open_ome_zarr(position_path, layout="fov", mode="r+") as pos:
            pos.zattrs["biahub-test"] = {"v": 1}

        with pytest.warns(UserWarning, match="will be overwritten"):
            process_single_position(
                func=lambda x: x,
                input_position_path=position_path,
                output_position_path=position_path,
                extra_metadata={"biahub-test": {"v": 2}},
                num_workers=1,
            )

        with open_ome_zarr(position_path, layout="fov", mode="r") as pos:
            assert pos.zattrs["biahub-test"] == {"v": 2}


@pytest.mark.parametrize(
    ("env", "expected_min", "expected_max"),
    [
        ("4", 4, 4),  # honour SLURM_CPUS_PER_TASK exactly
        (None, 1, None),  # fall back to os.cpu_count() when unset
        ("", 1, None),  # fall back when empty
        ("abc", 1, None),  # fall back when non-numeric
    ],
)
def test_available_cpus_honours_slurm_env(monkeypatch, env, expected_min, expected_max):
    if env is None:
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    else:
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", env)
    n = _available_cpus()
    assert n >= expected_min
    if expected_max is not None:
        assert n == expected_max


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
    with pytest.deprecated_call(match="shards_ratio is deprecated"):
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


def test_v05_explicit_shard_shape_is_honored(tmp_path):
    """An explicit TCZYX shard shape overrides the default."""
    store = tmp_path / "test.zarr"
    create_empty_plate(
        store_path=store,
        position_keys=[("A", "1", "0")],
        channel_names=["c0", "c1"],
        shape=(4, 2, 16, 256, 256),
        chunks=(1, 1, 16, 256, 256),
        shards=(2, 1, 16, 256, 256),
        version="0.5",
    )
    arr = _open_array(store, ("A", "1", "0"))
    assert arr.shards == (2, 1, 16, 256, 256)


def test_v05_shard_size_target_fills_space_before_time(tmp_path):
    """A byte target is met without the caller knowing the chunk shape.

    The mantis-v2 case from issue #458: one (Z, Y, X) volume is ~440 MB, so a
    2 GB target buys the volume plus three more timepoints.
    """
    store = tmp_path / "test.zarr"
    create_empty_plate(
        store_path=store,
        position_keys=[("A", "1", "0")],
        channel_names=[f"c{i}" for i in range(6)],
        shape=(5, 6, 86, 1664, 1193),
        shards="2GB",
        dtype=np.uint16,
        version="0.5",
    )
    arr = _open_array(store, ("A", "1", "0"))
    assert arr.chunks == (1, 1, *_V05_DEFAULT_ZYX_CHUNKS)
    assert arr.shards == (4, 1, 96, 1792, 1280)
    assert math.prod(arr.shards) * 2 <= 2_000_000_000


def test_v05_shard_extent_keyword_matches_the_default(tmp_path):
    """``shards="XYZ"`` states the default explicitly."""
    shape = (4, 2, 20, 300, 300)
    stores = {}
    for name, shards in (("default", None), ("explicit", "XYZ")):
        store = tmp_path / f"{name}.zarr"
        create_empty_plate(
            store_path=store,
            position_keys=[("A", "1", "0")],
            channel_names=["c0", "c1"],
            shape=shape,
            shards=shards,
            version="0.5",
        )
        stores[name] = _open_array(store, ("A", "1", "0")).shards
    assert stores["default"] == stores["explicit"] == (1, 1, 32, 512, 512)


def test_v05_shards_and_shards_ratio_conflict(tmp_path):
    """Asking for both geometries is an error rather than a silent winner."""
    with pytest.raises(ValueError, match="not both"):
        create_empty_plate(
            store_path=tmp_path / "test.zarr",
            position_keys=[("A", "1", "0")],
            channel_names=["c0"],
            shape=(4, 1, 16, 256, 256),
            shards="2GB",
            shards_ratio=(2, 1, 1, 1, 1),
            version="0.5",
        )


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
    with pytest.deprecated_call(), pytest.raises(ValueError, match="Sharding is not supported in Zarr v2"):
        create_empty_plate(
            store_path=store,
            position_keys=[("A", "1", "0")],
            channel_names=["c0"],
            shape=(2, 1, 8, 64, 64),
            shards_ratio=(1, 1, 1, 1, 1),
            version="0.4",
        )


@pytest.mark.parametrize("shards", ["2GB", "XYZ", (1, 1, 8, 64, 64)])
def test_v04_rejects_shards(tmp_path, shards):
    """Every shards form is rejected on a v0.4 store, not silently ignored."""
    store = tmp_path / "test.zarr"
    with pytest.raises(ValueError, match="Sharding is not supported in Zarr v2"):
        create_empty_plate(
            store_path=store,
            position_keys=[("A", "1", "0")],
            channel_names=["c0"],
            shape=(2, 1, 8, 64, 64),
            shards=shards,
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


# -- Gapped time selections on T-sharded stores -------------------------------
#
# Sharding along T makes ``process_single_position`` batch a shard's worth of
# timepoints into a single write, and that write can end up addressing
# non-consecutive timepoints: an all-zero or all-NaN input in the middle of the
# batch is dropped, or the caller asks for non-consecutive indices outright.
# A gapped selection is not expressible as a slice, which a sharded write
# requires, so the write has to be split into runs of consecutive timepoints.


@pytest.mark.parametrize(
    ("indices", "expected"),
    [
        ([], []),
        ([3], [slice(0, 1)]),
        ([0, 1, 2, 3, 4], [slice(0, 5)]),
        ([0, 3, 4], [slice(0, 1), slice(1, 3)]),  # the t=1,2-skipped case
        ([0, 2, 4], [slice(0, 1), slice(1, 2), slice(2, 3)]),
        ([5, 6, 9], [slice(0, 2), slice(2, 3)]),
    ],
)
def test_contiguous_runs(indices, expected):
    """``_contiguous_runs`` slices an index list at every gap."""
    assert _contiguous_runs(indices) == expected


def _t_sharded_stores(tmp_path, shape, position_key, shard_t):
    """Input and output stores whose shards hold ``shard_t`` timepoints each."""
    input_store = tmp_path / "input.zarr"
    output_store = tmp_path / "output.zarr"
    for store in (input_store, output_store):
        create_empty_plate(
            store_path=store,
            position_keys=[position_key],
            channel_names=[f"c{i}" for i in range(shape[1])],
            shape=shape,
            chunks=(1, 1, shape[2], shape[3] // 2, shape[4] // 2),
            shards_ratio=(shard_t, 1, 1, 2, 2),
            version="0.5",
        )
    return input_store, output_store


def test_process_single_position_with_zero_timepoints_inside_a_t_shard(tmp_path):
    """A dropped timepoint mid-batch must not break the shard-aligned write.

    ``t=1`` and ``t=2`` are all zeros, so they are skipped and the surviving
    output indices of the ``t=0..4`` batch are ``[0, 3, 4]`` — a gapped
    selection. Regression test: this used to reach a zarr-python code path that
    indexed the value buffer with the shard's own coordinates, raising
    ``IndexError`` or attempting an absurd allocation.
    """
    shape = (5, 2, 4, 16, 16)
    position_key = ("A", "1", "0")
    input_store, output_store = _t_sharded_stores(tmp_path, shape, position_key, shard_t=5)
    populate_store(input_store, [position_key], shape, np.float32)
    with open_ome_zarr(input_store, mode="r+") as in_ds:
        in_ds["/".join(position_key)].data[1:3] = 0

    assert _open_array(output_store, position_key).shards[0] == 5

    process_single_position(
        func=dummy_transform,
        input_position_path=input_store / Path(*position_key),
        output_position_path=output_store / Path(*position_key),
        input_channel_indices=[[0], [1]],
        output_channel_indices=[[0], [1]],
        input_time_indices=list(range(shape[0])),
        output_time_indices=list(range(shape[0])),
        constant=2,
    )

    with open_ome_zarr(input_store) as in_ds, open_ome_zarr(output_store) as out_ds:
        in_data = in_ds["/".join(position_key)].data[:]
        out_data = out_ds["/".join(position_key)].data[:]
    # Skipped timepoints are left at the fill value; the rest are transformed.
    expected = dummy_transform(in_data, constant=2)
    np.testing.assert_array_almost_equal(out_data, expected)
    assert not out_data[1:3].any()


def test_apply_transform_to_tczyx_with_gapped_output_time_indices(tmp_path):
    """A caller-requested gap within one T shard is written correctly."""
    shape = (5, 1, 4, 16, 16)
    position_key = ("A", "1", "0")
    input_store, output_store = _t_sharded_stores(tmp_path, shape, position_key, shard_t=5)
    populate_store(input_store, [position_key], shape, np.float32)

    apply_transform_to_tczyx_and_save(
        func=dummy_transform,
        input_position_path=input_store / Path(*position_key),
        output_position_path=output_store / Path(*position_key),
        input_channel_indices=0,
        output_channel_indices=0,
        input_time_indices=[0, 2, 4],
        output_time_indices=[0, 2, 4],
        constant=2,
    )

    with open_ome_zarr(input_store) as in_ds, open_ome_zarr(output_store) as out_ds:
        in_data = in_ds["/".join(position_key)].data[:]
        out_data = out_ds["/".join(position_key)].data[:]
    written = [0, 2, 4]
    np.testing.assert_array_almost_equal(out_data[written], dummy_transform(in_data[written], constant=2))
    assert not out_data[[1, 3]].any()


# -- Interrupted-write repair and resume -----------------------------------


def counting_transform(data, constant=2, call_log_dir=None):
    """Multiply like dummy_transform, recording one file per invocation.

    A file per call rather than a counter so the tally survives being made
    from worker processes.
    """
    if call_log_dir is not None:
        log = Path(call_log_dir)
        log.mkdir(parents=True, exist_ok=True)
        (log / os.urandom(8).hex()).touch()
    return data * constant


def _call_count(call_log_dir: Path) -> int:
    return len(list(call_log_dir.iterdir())) if call_log_dir.exists() else 0


def _shard_files(store_path: Path, position_key: tuple[str, str, str]) -> list[Path]:
    chunk_root = store_path / Path(*position_key) / "0" / "c"
    return sorted(p for p in chunk_root.rglob("*") if p.is_file())


def _make_stores(tmp_path: Path, shape, position_key, **plate_kwargs):
    input_store = tmp_path / "input.zarr"
    output_store = tmp_path / "output.zarr"
    for store in (input_store, output_store):
        create_empty_plate(
            store_path=store,
            position_keys=[position_key],
            channel_names=[f"c{c}" for c in range(shape[1])],
            shape=shape,
            version="0.5",
            **plate_kwargs,
        )
    populate_store(input_store, [position_key], shape, np.float32)
    return input_store, output_store


def _tear(path: Path) -> None:
    """Truncate a shard, as a job killed part-way through a write would."""
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])


#: A geometry matching what reconstruction pipelines produce: inner chunks
#: that do not divide the data extent, so the shard grid rounds up past the
#: array bound and no write can cover a whole shard. Every write is then a
#: read-modify-write, which is what makes a torn shard fatal on retry rather
#: than merely wasteful. The shard also spans ten timepoints, so one write
#: unit is a batch of ten.
_RMW_SHAPE = (23, 1, 17, 20, 20)
_RMW_PLATE = {"chunks": (1, 1, 16, 16, 16), "shards_ratio": (10, 1, 2, 2, 2)}
#: Timepoints in the first shard-aligned batch of ``_RMW_SHAPE``.
_RMW_UNIT_SIZE = 10


def test_plan_write_unit_only_claims_fully_covered_shards(tmp_path):
    """A unit owns a shard only if it writes every in-bounds element of it."""
    shape = (23, 2, 4, 8, 8)
    position_key = ("A", "1", "0")
    _, output_store = _make_stores(
        tmp_path,
        shape,
        position_key,
        chunks=(1, 1, 4, 8, 8),
        shards_ratio=(10, 1, 1, 1, 1),
    )
    position_path = output_store / Path(*position_key)

    with open_ome_zarr(position_path, layout="fov", mode="r") as dataset:
        array = dataset.data
        assert array.shards == (10, 1, 4, 8, 8)

        # A whole shard row of timepoints is owned, and maps to one file.
        unit = plan_write_unit(array, list(range(10)), [0])
        assert unit is not None
        assert [path for path, _ in unit.shards] == [position_path / "0" / "c" / "0" / "0" / "0" / "0" / "0"]

        # The ragged final row is owned too: t=20..22 is all of it that exists.
        assert plan_write_unit(array, [20, 21, 22], [0]) is not None

        # A subset of a shard row is shared with another write, so untracked.
        assert plan_write_unit(array, [0, 1], [0]) is None
        assert plan_write_unit(array, [*range(10), 10], [0]) is None


def test_write_over_torn_shard_recovers(tmp_path):
    """A shard left half-written by a killed job is replaced, not read back.

    Without clearing the file first, the write is a read-modify-write which
    re-reads the damaged shard and fails on its checksum, so every retry of
    the position fails the same way.
    """
    shape = _RMW_SHAPE
    position_key = ("A", "1", "0")
    input_store, output_store = _make_stores(tmp_path, shape, position_key, **_RMW_PLATE)
    run = partial(
        process_single_position,
        func=dummy_transform,
        input_position_path=input_store / Path(*position_key),
        output_position_path=output_store / Path(*position_key),
        constant=2,
    )

    run()
    shards = _shard_files(output_store, position_key)
    assert shards
    _tear(shards[0])

    run()

    with open_ome_zarr(input_store) as in_ds, open_ome_zarr(output_store) as out_ds:
        expected = dummy_transform(in_ds["/".join(position_key)].data[:], constant=2)
        np.testing.assert_array_almost_equal(out_ds["/".join(position_key)].data[:], expected)


@pytest.mark.parametrize("num_workers", [1, 2])
@pytest.mark.parametrize("use_threads", [False, True])
def test_resume_skips_units_already_written(tmp_path, num_workers, use_threads):
    shape = (4, 1, 4, 8, 8)
    position_key = ("A", "1", "0")
    input_store, output_store = _make_stores(tmp_path, shape, position_key)
    call_log = tmp_path / "calls"
    run = partial(
        process_single_position,
        func=counting_transform,
        input_position_path=input_store / Path(*position_key),
        output_position_path=output_store / Path(*position_key),
        num_workers=num_workers,
        use_threads=use_threads,
        constant=2,
        call_log_dir=str(call_log),
    )

    run(resume=True)
    first_pass = _call_count(call_log)
    assert first_pass == shape[0]

    run(resume=True)
    assert _call_count(call_log) == first_pass, "resume recomputed units that were already written"

    with open_ome_zarr(input_store) as in_ds, open_ome_zarr(output_store) as out_ds:
        expected = counting_transform(in_ds["/".join(position_key)].data[:], constant=2)
        np.testing.assert_array_almost_equal(out_ds["/".join(position_key)].data[:], expected)


def test_resume_without_markers_recomputes_everything(tmp_path):
    """Resuming a store written by an earlier version recomputes it."""
    shape = (2, 1, 4, 8, 8)
    position_key = ("A", "1", "0")
    input_store, output_store = _make_stores(tmp_path, shape, position_key)
    call_log = tmp_path / "calls"
    run = partial(
        process_single_position,
        func=counting_transform,
        input_position_path=input_store / Path(*position_key),
        output_position_path=output_store / Path(*position_key),
        constant=2,
        call_log_dir=str(call_log),
    )

    run()
    marker_dir = progress_dir_for(output_store / Path(*position_key))
    assert list(marker_dir.iterdir())
    shutil.rmtree(marker_dir)

    run(resume=True)
    assert _call_count(call_log) == 2 * shape[0]


def test_resume_recomputes_a_unit_whose_shard_is_torn(tmp_path):
    """A completion marker is not trusted over an unreadable shard."""
    shape = _RMW_SHAPE
    position_key = ("A", "1", "0")
    input_store, output_store = _make_stores(tmp_path, shape, position_key, **_RMW_PLATE)
    call_log = tmp_path / "calls"
    run = partial(
        process_single_position,
        func=counting_transform,
        input_position_path=input_store / Path(*position_key),
        output_position_path=output_store / Path(*position_key),
        constant=2,
        call_log_dir=str(call_log),
    )

    run(resume=True)
    first_pass = _call_count(call_log)
    assert first_pass == shape[0]
    _tear(_shard_files(output_store, position_key)[0])

    run(resume=True)
    # Only the torn unit is recomputed: its whole batch of timepoints, and
    # nothing from the units that are still intact.
    expected = first_pass + _RMW_UNIT_SIZE
    assert _call_count(call_log) == expected, "torn shard was skipped instead of rewritten"

    with open_ome_zarr(input_store) as in_ds, open_ome_zarr(output_store) as out_ds:
        expected = counting_transform(in_ds["/".join(position_key)].data[:], constant=2)
        np.testing.assert_array_almost_equal(out_ds["/".join(position_key)].data[:], expected)


@pytest.mark.parametrize(
    ("time_indices", "channel_indices"),
    [
        ([0], [0]),  # list of indices, as process_single_position usually yields
        ([0], 0),  # scalar channel: a caller passed a flat channel list
        (0, 0),  # scalar in both dimensions
        ([0], slice(0, 1)),  # slice, as a caller grouping channels may pass
    ],
    ids=["lists", "scalar-channel", "scalar-both", "slice-channel"],
)
def test_plan_write_unit_accepts_every_selection_form(tmp_path, time_indices, channel_indices):
    """Scalar and slice selections are legal and must be planned, not rejected.

    ``process_single_position`` hands a bare int to
    ``apply_transform_to_tczyx_and_save`` whenever a caller passes a flat list
    of channel indices instead of a list of channel groups, which is what
    ``biahub concatenate`` does. Treating that as a sequence raised
    ``TypeError: 'int' object is not iterable`` and aborted the step.
    """
    shape = (4, 3, 4, 8, 8)
    position_key = ("A", "1", "0")
    _, output_store = _make_stores(tmp_path, shape, position_key)

    with open_ome_zarr(output_store / Path(*position_key), layout="fov", mode="r") as dataset:
        unit = plan_write_unit(dataset.data, time_indices, channel_indices)

    assert unit is not None
    assert unit.time_indices == (0,)
    assert unit.channel_indices == (0,)
    assert len(unit.shards) == 1


def test_process_single_position_with_flat_channel_indices(tmp_path):
    """End-to-end with concatenate's calling convention (flat channel list)."""
    shape = (2, 3, 4, 8, 8)
    position_key = ("A", "1", "0")
    input_store, output_store = _make_stores(tmp_path, shape, position_key)

    process_single_position(
        func=dummy_transform,
        input_position_path=input_store / Path(*position_key),
        output_position_path=output_store / Path(*position_key),
        input_channel_indices=[0, 1, 2],
        output_channel_indices=[0, 1, 2],
        constant=2,
        resume=True,
    )

    with open_ome_zarr(input_store) as in_ds, open_ome_zarr(output_store) as out_ds:
        expected = dummy_transform(in_ds["/".join(position_key)].data[:], constant=2)
        np.testing.assert_array_almost_equal(out_ds["/".join(position_key)].data[:], expected)


# -- Progress records live beside the store, not inside it ------------------


def test_progress_records_are_written_beside_the_store(tmp_path):
    """Nothing iohub owns is written inside the output store.

    Progress used to live in the array directory, which meant a ``cp -r`` of a
    finished store carried it along and a later resume against the copy skipped
    everything.
    """
    shape = (2, 1, 4, 8, 8)
    position_key = ("A", "1", "0")
    input_store, output_store = _make_stores(tmp_path, shape, position_key)

    process_single_position(
        func=dummy_transform,
        input_position_path=input_store / Path(*position_key),
        output_position_path=output_store / Path(*position_key),
        constant=2,
        resume=True,
    )

    stray = [p for p in output_store.rglob("*") if "iohub" in p.name or "progress" in p.name]
    assert stray == [], f"iohub wrote inside the store: {stray}"

    expected = tmp_path / ".iohub-progress" / output_store.name / Path(*position_key)
    assert expected == progress_dir_for(output_store / Path(*position_key))
    assert sorted(p.name for p in expected.glob("*.done")) != []


def test_progress_record_names_the_shards_it_guarantees(tmp_path):
    """Each record lists the shards that must still exist for it to count."""
    shape = (2, 1, 4, 8, 8)
    position_key = ("A", "1", "0")
    input_store, output_store = _make_stores(tmp_path, shape, position_key)
    # One all-zero timepoint, which is skipped rather than written.
    with open_ome_zarr(input_store, mode="r+") as dataset:
        dataset["/".join(position_key)].data[0] = 0

    process_single_position(
        func=dummy_transform,
        input_position_path=input_store / Path(*position_key),
        output_position_path=output_store / Path(*position_key),
        constant=2,
        resume=True,
    )

    records = {
        p.name: json.loads(p.read_text()) for p in progress_dir_for(output_store / Path(*position_key)).glob("*.done")
    }
    assert len(records) == shape[0]
    by_shard_count = sorted(len(r["shards"]) for r in records.values())
    # The all-zero unit claims nothing; the written one names its shard.
    assert by_shard_count == [0, 1]


def test_resume_recomputes_when_the_store_data_was_deleted(tmp_path):
    """Records outside the store must not survive deletion of the store's data.

    This is the hazard created by moving the records out: ``rm -rf`` on the
    store no longer removes them, so a resume that trusted them blindly would
    skip every unit and leave an empty store.
    """
    shape = (2, 1, 4, 8, 8)
    position_key = ("A", "1", "0")
    input_store, output_store = _make_stores(tmp_path, shape, position_key)
    call_log = tmp_path / "calls"
    run = partial(
        process_single_position,
        func=counting_transform,
        input_position_path=input_store / Path(*position_key),
        output_position_path=output_store / Path(*position_key),
        constant=2,
        call_log_dir=str(call_log),
        resume=True,
    )

    run()
    first_pass = _call_count(call_log)
    assert first_pass == shape[0]

    shutil.rmtree(output_store / Path(*position_key) / "0" / "c")
    run()
    assert _call_count(call_log) == 2 * first_pass, "resume skipped units whose data was gone"

    with open_ome_zarr(input_store) as in_ds, open_ome_zarr(output_store) as out_ds:
        expected = counting_transform(in_ds["/".join(position_key)].data[:], constant=2)
        np.testing.assert_array_almost_equal(out_ds["/".join(position_key)].data[:], expected)


def test_write_xarray_repairs_without_writing_records(tmp_path):
    """write_xarray gets repair only: no progress records, inside or outside."""
    shape = (2, 1, 4, 8, 8)
    position_key = ("A", "1", "0")
    _, output_store = _make_stores(tmp_path, shape, position_key)
    czyx = np.ones(shape[1:], dtype=np.float32)
    coords = {
        "t": [0.0],
        "c": ["c0"],
        "z": np.arange(shape[2], dtype=float),
        "y": np.arange(shape[3], dtype=float),
        "x": np.arange(shape[4], dtype=float),
    }
    array = xr.DataArray(czyx[None], dims=("t", "c", "z", "y", "x"), coords=coords)

    with open_ome_zarr(output_store / Path(*position_key), layout="fov", mode="r+") as position:
        position.write_xarray(array)

    assert [p for p in output_store.rglob("*") if "iohub" in p.name] == []
    assert progress_dir_for(output_store / Path(*position_key)).exists() is False


def test_rerun_clears_shards_when_every_timepoint_is_skipped(tmp_path):
    """A unit whose input became all-zero must not keep the old output.

    The writing branch clears the unit's shards unconditionally, so a timepoint
    skipped *alongside* one that wrote is cleared. Without clearing here too,
    the same input condition would give a different store depending only on
    whether a sibling timepoint shared the shard.
    """
    shape = (2, 1, 4, 8, 8)
    position_key = ("A", "1", "0")
    input_store, output_store = _make_stores(tmp_path, shape, position_key)
    run = partial(
        process_single_position,
        func=dummy_transform,
        input_position_path=input_store / Path(*position_key),
        output_position_path=output_store / Path(*position_key),
        constant=2,
    )

    run()
    with open_ome_zarr(output_store) as out_ds:
        assert np.any(np.asarray(out_ds["/".join(position_key)].data[0])), "nothing written to t=0"

    # The input for t=0 becomes all zeros, so that unit is skipped entirely.
    with open_ome_zarr(input_store, mode="r+") as in_ds:
        in_ds["/".join(position_key)].data[0] = 0
    run()

    with open_ome_zarr(output_store) as out_ds:
        data = np.asarray(out_ds["/".join(position_key)].data[:])
    assert not np.any(data[0]), "stale output kept for a timepoint that is now all-zero"
    assert np.any(data[1]), "the untouched timepoint was lost"

    records = {
        p.name: json.loads(p.read_text()) for p in progress_dir_for(output_store / Path(*position_key)).glob("*.done")
    }
    empty = [name for name, record in records.items() if record["shards"] == []]
    assert len(empty) == 1
    assert empty[0].startswith("t0-0")


def test_resume_recomputes_when_an_unclaimed_shard_is_present(tmp_path):
    """A record must describe the store exactly, including what it omits.

    A record claiming no shards next to a leftover file from an earlier run
    describes a store that is not what a fresh run would produce, so the unit
    has to be recomputed rather than skipped.
    """
    shape = (2, 1, 4, 8, 8)
    position_key = ("A", "1", "0")
    input_store, output_store = _make_stores(tmp_path, shape, position_key)
    call_log = tmp_path / "calls"
    run = partial(
        process_single_position,
        func=counting_transform,
        input_position_path=input_store / Path(*position_key),
        output_position_path=output_store / Path(*position_key),
        constant=2,
        call_log_dir=str(call_log),
        resume=True,
    )

    run()
    first_pass = _call_count(call_log)
    assert first_pass == shape[0]

    # Rewrite one record to claim nothing while its shard stays on disk — the
    # state a store is left in by a version that did not clear on skip.
    records = sorted(progress_dir_for(output_store / Path(*position_key)).glob("t0-0_*.done"))
    assert len(records) == 1
    records[0].write_text(json.dumps({"shards": []}))

    run()
    assert _call_count(call_log) == first_pass + 1, "resume skipped a unit whose store no longer matches its record"
