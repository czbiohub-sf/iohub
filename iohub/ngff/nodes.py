"""
Node object and convenience functions for the OME-NGFF (OME-Zarr) Hierarchy.
"""

# TODO: remove this in the future (PEP deferred for 3.11, now 3.12?)
from __future__ import annotations

import logging
import math
import os
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Generator, Literal, Sequence, Type, TypeAlias, overload

import numpy as np
import zarr.codecs
from numpy.typing import ArrayLike, DTypeLike, NDArray
from pydantic import ValidationError
from zarr.core.chunk_key_encodings import ChunkKeyEncodingParams
from zarr.storage._utils import normalize_path

from iohub.ngff.display import channel_display_settings
from iohub.ngff.models import (
    TO_DICT_SETTINGS,
    AcquisitionMeta,
    AxisMeta,
    ChannelAxisMeta,
    DatasetMeta,
    ImageMeta,
    ImagesMeta,
    MultiScaleMeta,
    OMEROMeta,
    PlateAxisMeta,
    PlateMeta,
    RDefsMeta,
    SpaceAxisMeta,
    TimeAxisMeta,
    TransformationMeta,
    WellGroupMeta,
    WellIndexMeta,
    WindowDict,
)

_logger = logging.getLogger(__name__)

# Type alias for position specification tuples in create_positions
PositionSpec: TypeAlias = (
    tuple[str, str, str]
    | tuple[str, str, str, int | None]
    | tuple[str, str, str, int | None, int | None]
    | tuple[str, str, str, int | None, int | None, int]
)


def _pad_shape(shape: tuple[int, ...], target: int = 5):
    """Pad shape tuple to a target length."""
    pad = target - len(shape)
    return (1,) * pad + shape


def _open_store(
    store_path: str | Path,
    mode: Literal["r", "r+", "a", "w", "w-"],
    version: Literal["0.4", "0.5"],
):
    store_path = Path(store_path).resolve()
    if not store_path.exists() and mode in ("r", "r+"):
        raise FileNotFoundError(
            f"Dataset directory not found at {str(store_path)}."
        )
    if version not in ("0.4", "0.5"):
        _logger.warning(
            "IOHub is only tested against OME-NGFF v0.4 and v0.5. "
            f"Requested version {version} may not work properly."
        )
    try:
        zarr_format = None
        if mode in ("w", "w-") or (mode == "a" and not store_path.exists()):
            zarr_format = 3 if version == "0.5" else 2
        root = zarr.open_group(store_path, mode=mode, zarr_format=zarr_format)
    except Exception as e:
        raise RuntimeError(
            f"Cannot open Zarr root group at {str(store_path)}"
        ) from e
    return root


def _scale_integers(values: Sequence[int], factor: int) -> tuple[int, ...]:
    """Computes the ceiling of the input sequence divided by the factor."""
    return tuple(int(math.ceil(v / factor)) for v in values)


def _case_insensitive_local_fs() -> bool:
    """Check if the local filesystem is case-insensitive."""
    return Path(__file__.lower()).exists() and Path(__file__.upper()).exists()


class NGFFNode:
    """A node (group level in Zarr) in an NGFF dataset."""

    _MEMBER_TYPE: Type[NGFFNode | zarr.Array]
    _DEFAULT_AXES = [
        TimeAxisMeta(name="T", unit="second"),
        ChannelAxisMeta(name="C"),
        *[SpaceAxisMeta(name=i, unit="micrometer") for i in ("Z", "Y", "X")],
    ]

    def __init__(
        self,
        group: zarr.Group,
        parse_meta: bool = True,
        channel_names: list[str] | None = None,
        axes: list[AxisMeta] | None = None,
        version: Literal["0.4", "0.5"] = "0.4",
        overwriting_creation: bool = False,
    ):
        if channel_names:
            self._channel_names = channel_names
        elif not parse_meta:
            raise ValueError(
                "Channel names need to be provided or in metadata."
            )
        if axes:
            self.axes = axes
        self._group = group
        self._overwrite = overwriting_creation
        self._version: Literal["0.4", "0.5"] = version
        if parse_meta:
            self._parse_meta()
        if not hasattr(self, "axes"):
            self.axes = self._DEFAULT_AXES
        # TODO: properly check the underlying storage type
        # This works for now as only the local filesystem is supported
        self._case_insensitive_fs = _case_insensitive_local_fs()

    @property
    def zgroup(self):
        """Corresponding Zarr group of the node."""
        return self._group

    @property
    def zattrs(self):
        """Zarr attributes of the node.
        Assignments will modify the metadata file."""
        return self._group.attrs

    @property
    def maybe_wrapped_ome_attrs(self):
        """Container of OME metadata attributes."""
        return self.zattrs.get("ome") or self.zattrs

    @property
    def version(self) -> Literal["0.4", "0.5"]:
        """NGFF version"""
        return self._version

    @property
    def channel_names(self):
        return self._channel_names

    @property
    def _parent_path(self):
        """The parent Zarr group path of the node.
        None for the root node."""
        if self._group.name == "/":
            return None
        else:
            return os.path.dirname(self._group.name)

    @property
    def _member_names(self):
        """Group keys (default) or array keys (overridden)."""
        return self.group_keys()

    @property
    def _child_attrs(self):
        """Attributes to pass on when constructing child type instances"""
        return dict(
            version=self._version,
            axes=self.axes,
            channel_names=self._channel_names,
            overwriting_creation=self._overwrite,
        )

    def __len__(self):
        return len(self._member_names)

    def __getitem__(self, key):
        key = normalize_path(str(key))
        znode = self.zgroup.get(key)
        if not znode:
            raise KeyError(key)
        levels = len(key.split("/")) - 1
        item_type = self._MEMBER_TYPE
        for _ in range(levels):
            item_type = item_type._MEMBER_TYPE
        if issubclass(item_type, ImageArray):
            return item_type.from_zarr_array(znode)
        else:
            return item_type(group=znode, parse_meta=True, **self._child_attrs)

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __delitem__(self, key):
        """.. Warning: this does NOT clean up metadata!"""
        key = normalize_path(str(key))
        if key in self._member_names:
            del self[key]

    def __contains__(self, key):
        key = normalize_path(str(key))
        if not self._case_insensitive_fs:
            return key in self._member_names
        for name in self._member_names:
            if key.lower() != name.lower():
                continue
            if key != name:
                _logger.warning(
                    f"Key '{key}' matched member '{name}'. "
                    "This may not work on case-sensitive filesystems."
                )
            return True
        return False

    def __iter__(self):
        yield from self._member_names

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def group_keys(self):
        """Sorted list of keys to all the child zgroups (if any).

        Returns
        -------
        list[str]
        """
        return sorted(list(self._group.group_keys()))

    def array_keys(self):
        """Sorted list of keys to all the child zarrays (if any).

        Returns
        -------
        list[str]
        """
        return sorted(list(self._group.array_keys()))

    def is_root(self):
        """Whether this node is the root node

        Returns
        -------
        bool
        """
        return self._group.name == "/"

    def is_leaf(self):
        """Wheter this node is a leaf node,
        meaning that no child Zarr group is present.
        Usually a position/fov node for NGFF-HCS if True.

        Returns
        -------
        bool
        """
        return not self.group_keys()

    def print_tree(self, level: int | None = None):
        """Print hierarchy of the node to stdout.

        Parameters
        ----------
        level : int, optional
            Maximum depth to show, by default None
        """
        print(self.zgroup.tree(level=level))

    def iteritems(self):
        for key in self._member_names:
            try:
                yield key, self[key]
            except Exception:
                _logger.warning(
                    "Skipped item at {}: invalid {}.".format(
                        key, type(self._MEMBER_TYPE)
                    )
                )

    def get_channel_index(self, name: str):
        """Get the index of a given channel in the channel list.

        Parameters
        ----------
        name : str
            Name of the channel.

        Returns
        -------
        int
            Index of the channel.
        """
        if not hasattr(self, "_channel_names"):
            raise AttributeError(
                "Channel names are not set for this NGFF node. "
                f"Cannot get the index for channel name '{name}'"
            )
        if name not in self._channel_names:
            raise ValueError(
                f"Channel {name} is not in "
                f"the existing channels: {self._channel_names}"
            )
        return self._channel_names.index(name)

    def _warn_invalid_meta(self):
        msg = "Zarr group at {} does not have valid metadata for {}".format(
            self._group.path, type(self)
        )
        _logger.warning(msg)

    def _parse_meta(self):
        """Parse and set NGFF metadata from `.zattrs`."""
        raise NotImplementedError

    def _dump_ome(self, ome: dict):
        """Dump OME metadata to the `.zattrs` file."""
        if self.version == "0.4":
            self.zattrs.update(ome)
        elif self.version == "0.5":
            if "version" not in ome:
                ome["version"] = "0.5"
            self.zattrs["ome"] = ome

    def dump_meta(self):
        """Dumps metadata JSON to the `.zattrs` file."""
        raise NotImplementedError

    def close(self):
        """Close Zarr store."""
        self._group.store.close()


class ImageArray(zarr.Array):
    """Container object for image stored as a zarr array (up to 5D)"""

    @classmethod
    def from_zarr_array(cls, zarray: zarr.Array):
        return cls(zarray._async_array)

    @property
    def frames(self):
        return self._get_dim(0)

    @property
    def channels(self):
        return self._get_dim(1)

    @property
    def slices(self):
        return self._get_dim(2)

    @property
    def height(self):
        return self._get_dim(3)

    @property
    def width(self):
        return self._get_dim(4)

    def _get_dim(self, idx):
        return _pad_shape(self.shape, target=5)[idx]

    def numpy(self):
        """Return the whole image as an in-RAM NumPy array.
        `self.numpy()` is equivalent to `self[:]`."""
        return self[:]

    def dask_array(self):
        """Return as a dask array"""
        import dask.array as da

        # Note: Designed to work with zarr DirectoryStore
        return da.from_zarr(self.store.root, component=self.path)

    def downscale(self):
        raise NotImplementedError

    def tensorstore(self, **kwargs):
        """
        Open the zarr array as a TensorStore object.
        Needs the optional dependency ``tensorstore``.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments to pass to ``tensorstore.open()``,
            by default None

        Returns
        -------
        TensorStore
            Handle to the Zarr array.
        """
        import tensorstore as ts

        ts_spec = {
            "driver": "zarr2" if self.metadata.zarr_format == 2 else "zarr3",
            "kvstore": {
                "driver": "file",
                "path": str(Path(self.store.root) / self.path.strip("/")),
            },
        }
        if "read" in kwargs or "write" in kwargs:
            raise ValueError("Cannot override file mode for the Zarr store.")
        zarr_dataset = ts.open(
            ts_spec, read=True, write=not self.read_only, **kwargs
        ).result()
        return zarr_dataset


class TiledImageArray(ImageArray):
    """Container object for tiled image stored as a zarr array (up to 5D)."""

    @property
    def rows(self):
        """Number of rows in the tiles."""
        return int(self.shape[-2] / self.chunks[-2])

    @property
    def columns(self):
        """Number of columns in the tiles."""
        return int(self.shape[-1] / self.chunks[-1])

    @property
    def tiles(self):
        """A tuple of the tiled grid size (rows, columns)."""
        return (self.rows, self.columns)

    @property
    def tile_shape(self):
        """shape of a tile, the same as chunk size of the underlying array."""
        return self.chunks

    def get_tile(
        self,
        row: int,
        column: int,
        pre_dims: tuple[int | slice, ...] | None = None,
    ) -> NDArray:
        """Get a tile as an up-to-5D in-RAM NumPy array.

        Parameters
        ----------
        row : int
            Row index.
        column : int
            Column index.
        pre_dims : tuple[int | slice, ...], optional
            Indices or slices for previous dimensions than rows and columns
            with matching shape, e.g. (t, c, z) for 5D arrays,
            by default None (select all).

        Returns
        -------
        NDArray
        """
        self._check_rc(row, column)
        return self[self.get_tile_slice(row, column, pre_dims=pre_dims)]

    def write_tile(
        self,
        data: ArrayLike,
        row: int,
        column: int,
        pre_dims: tuple[int | slice, ...] | None = None,
    ) -> None:
        """Write a tile in the Zarr store.

        Parameters
        ----------
        data : ArrayLike
            Value to store.
        row : int
            Row index.
        column : int
            Column index.
        pre_dims : tuple[int | slice, ...], optional
            Indices or slices for previous dimensions than rows and columns
            with matching shape, e.g. (t, c, z) for 5D arrays,
            by default None (select all).
        """
        self._check_rc(row, column)
        self[self.get_tile_slice(row, column, pre_dims=pre_dims)] = data

    def get_tile_slice(
        self,
        row: int,
        column: int,
        pre_dims: tuple[int | slice, ...] | None = None,
    ) -> tuple[slice, ...]:
        """Get the slices for a tile in the underlying array.

        Parameters
        ----------
        row : int
            Row index.
        column : int
            Column index.
        pre_dims :  tuple[int | slice, ...], optional
            Indices or slices for previous dimensions than rows and columns
            with matching shape, e.g. (t, c, z) for 5D arrays,
            by default None (select all).

        Returns
        -------
        tuple[slice, ...]
            Tuple of slices for all the dimensions of the array.
        """
        self._check_rc(row, column)
        y, x = self.chunks[-2:]
        r_slice = slice(row * y, (row + 1) * y)
        c_slice = slice(column * x, (column + 1) * x)
        pad = [slice(None)] * (len(self.shape) - 2)
        if pre_dims is not None:
            try:
                if len(pre_dims) != len(pad):
                    raise IndexError(
                        f"Length of `pre_dims` should be {len(pad)}, "
                        f"got {len(pre_dims)}."
                    )
            except TypeError:
                raise TypeError(
                    "Argument `pre_dims` should be a sequence, "
                    f"got type {type(pre_dims)}."
                )
            for i, sel in enumerate(pre_dims):
                if isinstance(sel, int):
                    sel = slice(sel)
                if sel is not None:
                    pad[i] = sel
        return tuple((pad + [r_slice, c_slice]))

    @staticmethod
    def _check_rc(row: int, column: int):
        if not (isinstance(row, int) and isinstance(column, int)):
            raise TypeError("Row and column indices must be integers.")


class Position(NGFFNode):
    """The Zarr group level directly containing multiscale image arrays.

    Parameters
    ----------
    group : zarr.Group
        Zarr heirarchy group object
    parse_meta : bool, optional
        Whether to parse NGFF metadata in `.zattrs`, by default True
    channel_names : list[str], optional
        List of channel names, by default None
    axes : list[AxisMeta], optional
        List of axes (`ngff_meta.AxisMeta`, up to 5D), by default None
    overwriting_creation : bool, optional
        Whether to overwrite or error upon creating an existing child item,
        by default False

    Attributes
    ----------
    version : Literal["0.4", "0.5"]
        OME-NGFF specification version
    zgroup : Group
        Root Zarr group holding arrays
    zattr : Attributes
        Zarr attributes of the group
    channel_names : list[str]
        Name of the channels
    axes : list[AxisMeta]
        Axes metadata
    """

    _MEMBER_TYPE = ImageArray

    def __init__(
        self,
        group: zarr.Group,
        parse_meta: bool = True,
        channel_names: list[str] | None = None,
        axes: list[AxisMeta] | None = None,
        version: Literal["0.4", "0.5"] = "0.4",
        overwriting_creation: bool = False,
    ):
        super().__init__(
            group=group,
            parse_meta=parse_meta,
            channel_names=channel_names,
            axes=axes,
            version=version,
            overwriting_creation=overwriting_creation,
        )

    def _set_meta(self):
        self.axes = self.metadata.multiscales[0].axes
        if self.metadata.omero is not None:
            self._channel_names = [
                c.label for c in self.metadata.omero.channels
            ]
        else:
            _logger.warning(
                "OMERO metadata not found. "
                "Using channel indices as channel names."
            )
            example_image: ImageArray = self[
                self.metadata.multiscales[0].datasets[0].path
            ]
            self._channel_names = list(range(example_image.channels))

    def _parse_meta(self):
        try:
            self.metadata = ImagesMeta.model_validate(
                self.maybe_wrapped_ome_attrs
            )
            self._set_meta()
        except ValidationError as e:
            _logger.warning(str(e))
            self._warn_invalid_meta()

    def dump_meta(self):
        """Dumps metadata JSON to the `.zattrs` file."""
        ome = self.metadata.model_dump(**TO_DICT_SETTINGS)
        self._dump_ome(ome)

    @property
    def _zarr_format(self):
        return 3 if self.version == "0.5" else 2

    @property
    def _member_names(self):
        return self.array_keys()

    @property
    def data(self):
        """.. warning::
            This property does *NOT* aim to retrieve all the arrays.
            And it may also fail to retrive any data if arrays exist but
            are not named conventionally.

        Alias for an array named '0' in the position,
        which is usually the raw data (or the finest resolution in a pyramid).

        Returns
        -------
        ImageArray

        Raises
        ------
        KeyError
            If no array is named '0'.

        Notes
        -----
        Do not depend on this in non-interactive code!
        The name is hard-coded and is not guaranteed
        by the OME-NGFF specification.
        """
        try:
            return self["0"]
        except KeyError:
            raise KeyError(
                "There is no array named '0' "
                f"in the group of: {self.array_keys()}"
            )

    def __getitem__(self, key: int | str) -> ImageArray:
        """Get an image array member of the position.
        E.g. Raw-coordinates image, a multi-scale level, or labels

        Parameters
        ----------
        key : int| str
            Name or path to the image array.
            Integer key is converted to string (name).

        Returns
        -------
        ImageArray
            Container object for image stored as a zarr array (up to 5D)
        """
        return super().__getitem__(key)

    def __setitem__(self, key, value: NDArray):
        """Write an up-to-5D image with default settings."""
        key = normalize_path(str(key))
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"Value must be a NumPy array. Got type {type(value)}."
            )
        self.create_image(key, value)

    def images(self) -> Generator[tuple[str, ImageArray]]:
        """Returns a generator that iterate over the name and value
        of all the image arrays in the group.

        Yields
        ------
        tuple[str, ImageArray]
            Name and image array object.
        """
        yield from self.iteritems()

    def create_image(
        self,
        name: str,
        data: NDArray,
        chunks: tuple[int, ...] | None = None,
        shards_ratio: tuple[int, ...] | None = None,
        transform: list[TransformationMeta] | None = None,
        check_shape: bool = True,
    ):
        """Create a new image array in the position.

        Parameters
        ----------
        name : str
            Name key of the new image.
        data : NDArray
            Image data.
        chunks : tuple[int, ...], optional
            Chunk size, by default None.
            ZYX stack size will be used if not specified.
        shards_ratio : tuple[int, ...], optional
            Sharding ratio for each dimension, by default None.
            Each shard contains the product of the ratios number of chunks.
            No sharding will be used if not specified.
        transform : list[TransformationMeta], optional
            List of coordinate transformations, by default None.
            Should be specified for a non-native resolution level.
        check_shape : bool, optional
            Whether to check if image shape matches dataset axes,
            by default True

        Returns
        -------
        ImageArray
            Container object for image stored as a zarr array (up to 5D)
        """
        img_arr = self.create_zeros(
            name=name,
            shape=data.shape,
            dtype=data.dtype,
            chunks=chunks,
            shards_ratio=shards_ratio,
            transform=transform,
            check_shape=check_shape,
        )
        img_arr[...] = data
        return img_arr

    def create_zeros(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: DTypeLike,
        chunks: tuple[int, ...] | None = None,
        shards_ratio: tuple[int, ...] | None = None,
        transform: list[TransformationMeta] | None = None,
        check_shape: bool = True,
    ):
        """Create a new zero-filled image array in the position.
        Under default zarr-python settings of lazy writing,
        this will not write the array values,
        but only create a ``.zarray`` file.
        This is useful for writing larger-than-RAM images
        and/or writing from multiprocesses in chunks.

        Parameters
        ----------
        name : str
            Name key of the new image.
        shape : tuple[int, ...]
            Image shape.
        dtype : DTypeLike
            Data type.
        chunks : tuple[int, ...], optional
            Chunk size, by default None.
            ZYX stack size will be used if not specified.
        shards_ratio : tuple[int, ...], optional
            Sharding ratio for each dimension, by default None.
            Each shard contains the product of the ratios number of chunks.
            No sharding will be used if not specified.
        transform : list[TransformationMeta], optional
            List of coordinate transformations, by default None.
            Should be specified for a non-native resolution level.
        check_shape : bool, optional
            Whether to check if image shape matches dataset axes,
            by default True

        Returns
        -------
        ImageArray
            Container object for a zero-filled image as a lazy zarr array
        """
        if not chunks:
            chunks = self._default_chunks(shape, 3)
        if check_shape:
            self._check_shape(shape)
        if shards_ratio:
            if len(shards_ratio) != len(shape):
                raise ValueError(
                    f"Sharding ratio length {len(shards_ratio)} "
                    f"does not match shape length {len(shape)}."
                )
            shards = tuple(c * s for c, s in zip(chunks, shards_ratio))
        else:
            shards = None
        img_arr = ImageArray.from_zarr_array(
            self._group.create_array(
                name=name,
                shape=shape,
                dtype=dtype,
                chunks=chunks,
                shards=shards,
                overwrite=self._overwrite,
                fill_value=0,
                dimension_names=(
                    [ax.name for ax in self.axes]
                    if self._zarr_format == 3
                    else None
                ),
                chunk_key_encoding=ChunkKeyEncodingParams(
                    name="default" if self._zarr_format == 3 else "v2",
                    separator="/",
                ),
                **self._create_compressor_options(),
            )
        )
        self._create_image_meta(img_arr.basename, transform=transform)
        return img_arr

    @staticmethod
    def _default_chunks(shape, last_data_dims: int):
        chunks = shape[-min(last_data_dims, len(shape)) :]
        return _pad_shape(chunks, target=len(shape))

    def _check_shape(self, data_shape: tuple[int, ...]) -> None:
        if len(data_shape) != len(self.axes):
            raise ValueError(
                f"Image has {len(data_shape)} dimensions, "
                f"while the dataset has {len(self.axes)}."
            )
        num_ch = len(self.channel_names)
        if ch_axis := self._find_axis("channel"):
            msg = (
                f"Image has {data_shape[ch_axis]} channels, "
                f"while the dataset  has {num_ch}."
            )
            if data_shape[ch_axis] > num_ch:
                raise ValueError(msg)
            elif data_shape[ch_axis] < num_ch:
                _logger.warning(msg)
        else:
            _logger.info(
                "Dataset channel axis is not set. "
                "Skipping channel shape check."
            )

    def _create_compressor_options(self):
        shuffle = zarr.codecs.BloscShuffle.bitshuffle
        if self._zarr_format == 3:
            return {
                "compressors": zarr.codecs.BloscCodec(
                    cname="zstd",
                    clevel=1,
                    shuffle=shuffle,
                )
            }
        else:
            from numcodecs import Blosc

            return {
                "compressor": Blosc(
                    cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE
                )
            }

    def _create_image_meta(
        self,
        name: str,
        transform: list[TransformationMeta] | None = None,
        extra_meta: dict | None = None,
    ):
        if not transform:
            transform = [
                TransformationMeta(type="scale", scale=[1.0] * len(self.axes))
            ]
        dataset_meta = DatasetMeta(
            path=name, coordinate_transformations=transform
        )
        if not hasattr(self, "metadata"):
            self.metadata = ImagesMeta(
                multiscales=[
                    MultiScaleMeta(
                        version=self.version,
                        axes=self.axes,
                        datasets=[dataset_meta],
                        name=name,
                        coordinate_transformations=None,
                        metadata=extra_meta,
                    )
                ],
                omero=self._omero_meta(id=0, name=self._group.basename),
                version="0.5" if self.version == "0.5" else None,
            )
        elif (
            dataset_meta.path
            not in self.metadata.multiscales[0].get_dataset_paths()
        ):
            self.metadata.multiscales[0].datasets.append(dataset_meta)
        self.dump_meta()

    def _omero_meta(
        self,
        id: int,
        name: str,
        clims: list[tuple[float, float, float, float]] | None = None,
    ):
        if not clims:
            clims = [None] * len(self.channel_names)
        channels = []
        for i, (channel_name, clim) in enumerate(
            zip(self.channel_names, clims)
        ):
            if i == 0:
                first_chan = True
            channels.append(
                channel_display_settings(
                    channel_name, clim=clim, first_chan=first_chan
                )
            )
        omero_meta = OMEROMeta(
            version=self.version,
            id=id,
            name=name,
            channels=channels,
            rdefs=RDefsMeta(default_t=0, default_z=0),
        )
        return omero_meta

    def _find_axis(self, axis_type):
        for i, axis in enumerate(self.axes):
            if axis.type == axis_type:
                return i
        return None

    def _get_channel_axis(self):
        if (ch_ax := self._find_axis("channel")) is None:
            raise KeyError(
                "Axis 'channel' does not exist. "
                "Please update `self.axes` first."
            )
        else:
            return ch_ax

    def append_channel(self, chan_name: str, resize_arrays: bool = True):
        """Append a channel to the end of the channel list.

        Parameters
        ----------
        chan_name : str
            Name of the new channel
        resize_arrays : bool, optional
            Whether to resize all the image arrays for the new channel,
            by default True
        """
        if chan_name in self._channel_names:
            raise ValueError(f"Channel name '{chan_name}' already exists.")
        self._channel_names.append(chan_name)
        if resize_arrays:
            for _, img in self.images():
                ch_ax = self._get_channel_axis()
                shape = list(img.shape)
                if ch_ax < len(shape):
                    shape[ch_ax] += 1
                # prepend axis
                elif ch_ax == len(shape):
                    shape = _pad_shape(tuple(shape), target=len(shape) + 1)
                else:
                    raise IndexError(
                        f"Cannot infer channel axis for shape {shape}."
                    )
                img.resize(shape)
        if "omero" in self.metadata.model_dump().keys():
            self.metadata.omero.channels.append(
                channel_display_settings(chan_name)
            )
            self.dump_meta()

    def rename_channel(self, old: str, new: str):
        """Rename a channel in the channel list.

        Parameters
        ----------
        old : str
            Current name of the channel
        new : str
            New name of the channel
        """
        ch_idx = self.get_channel_index(old)
        self._channel_names[ch_idx] = new
        if hasattr(self.metadata, "omero"):
            self.metadata.omero.channels[ch_idx].label = new
        self.dump_meta()

    def update_channel(self, chan_name: str, target: str, data: ArrayLike):
        """Update a channel slice of the target image array with new data.

        The incoming data shape needs to be the same as the target array
        except for the non-existent channel dimension.
        For example a TCZYX array of shape (2, 3, 4, 1024, 2048) can be updated
        with data of shape (2, 4, 1024, 2048)

        Parameters
        ----------
        chan_name : str
            Channel name
        target: str
            Name of the image array to update
        data : ArrayLike
            New data array to write

        Notes
        -----
        This method is a syntactical variation
        of assigning values to the slice with the `=` operator,
        and users are encouraged to use array indexing directly.
        """
        img = self[target]
        ch_idx = self.get_channel_index(chan_name)
        ch_ax = self._get_channel_axis()
        ortho_sel = [slice(None)] * len(img.shape)
        ortho_sel[ch_ax] = ch_idx
        img.set_orthogonal_selection(tuple(ortho_sel), data)

    def initialize_pyramid(self, levels: int) -> None:
        """
        Initializes the pyramid arrays with a down scaling of 2 per level.
        Decimals shapes are rounded up to ceiling.
        Scales metadata are also updated.

        Parameters
        ----------
        levels : int
            Number of down scaling levels, if levels is 1 nothing happens.
        """
        array = self.data
        for level in range(1, levels):
            factor = 2**level

            shape = array.shape[:-3] + _scale_integers(
                array.shape[-3:], factor
            )

            chunks = _pad_shape(
                _scale_integers(array.chunks, factor), len(shape)
            )

            transforms = deepcopy(
                self.metadata.multiscales[0]
                .datasets[0]
                .coordinate_transformations
            )
            for tr in transforms:
                if tr.type == "scale":
                    for i in range(len(tr.scale))[-3:]:
                        tr.scale[i] *= factor

            self.create_zeros(
                name=str(level),
                shape=shape,
                dtype=array.dtype,
                chunks=chunks,
                transform=transforms,
            )

    @property
    def scale(self) -> list[float]:
        """
        Helper function for scale transform metadata of
        highest resolution scale.
        """
        return self.get_effective_scale(
            self.metadata.multiscales[0].datasets[0].path
        )

    @property
    def axis_names(self) -> list[str]:
        """
        Helper function for axis names of the highest resolution scale.

        Returns lowercase axis names.
        """
        return [
            axis.name.lower() for axis in self.metadata.multiscales[0].axes
        ]

    def get_axis_index(self, axis_name: str) -> int:
        """
        Get the index of a given axis.

        Parameters
        ----------
        name : str
            Name of the axis. Case insensitive.

        Returns
        -------
        int
            Index of the axis.
        """
        return self.axis_names.index(axis_name.lower())

    def _get_all_transforms(
        self, image: str | Literal["*"]
    ) -> list[TransformationMeta]:
        """Get all transforms metadata
        for one image array or the whole FOV.

        Parameters
        ----------
        image : str | Literal["*"]
            Name of one image array (e.g. "0") to query,
            or "*" for the whole FOV

        Returns
        -------
        list[TransformationMeta]
            All transforms applicable to this image or FOV.
        """
        transforms: list[TransformationMeta] = (
            [
                t
                for t in self.metadata.multiscales[
                    0
                ].coordinate_transformations
            ]
            if self.metadata.multiscales[0].coordinate_transformations
            is not None
            else []
        )
        if image != "*" and image in self:
            for i, dataset_meta in enumerate(
                self.metadata.multiscales[0].datasets
            ):
                if dataset_meta.path == image:
                    transforms.extend(
                        self.metadata.multiscales[0]
                        .datasets[i]
                        .coordinate_transformations
                    )
        elif image != "*":
            raise ValueError(f"Key {image} not recognized.")
        return transforms

    def get_effective_scale(
        self,
        image: str | Literal["*"],
    ) -> list[float]:
        """Get the effective coordinate scale metadata
        for one image array or the whole FOV.

        Parameters
        ----------
        image : str | Literal["*"]
            Name of one image array (e.g. "0") to query,
            or "*" for the whole FOV

        Returns
        -------
        list[float]
            A list of floats representing the total scale
            for the image or FOV for each axis.
        """
        transforms = self._get_all_transforms(image)

        full_scale = np.ones(len(self.axes), dtype=float)
        for transform in transforms:
            if transform.type == "scale":
                full_scale *= np.array(transform.scale)

        return [float(x) for x in full_scale]

    def get_effective_translation(
        self,
        image: str | Literal["*"],
    ) -> TransformationMeta:
        """Get the effective coordinate translation metadata
        for one image array or the whole FOV.

        Parameters
        ----------
        image : str | Literal["*"]
            Name of one image array (e.g. "0") to query,
            or "*" for the whole FOV

        Returns
        -------
        list[float]
            A list of floats representing the total translation
            for the image or FOV for each axis.
        """
        transforms = self._get_all_transforms(image)
        full_translation = np.zeros(len(self.axes), dtype=float)
        for transform in transforms:
            if transform.type == "translation":
                full_translation += np.array(transform.translation)

        return [float(x) for x in full_translation]

    def set_transform(
        self,
        image: str | Literal["*"],
        transform: list[TransformationMeta],
    ):
        """Set the coordinate transformations metadata
        for one image array or the whole FOV.

        Parameters
        ----------
        image : str | Literal["*"]
            Name of one image array (e.g. "0") to transform,
            or "*" for the whole FOV
        transform : list[TransformationMeta]
            List of transformations to apply
            (:py:class:`iohub.ngff_meta.TransformationMeta`)
        """
        if image == "*":
            self.metadata.multiscales[0].coordinate_transformations = transform
        elif image in self:
            for i, dataset_meta in enumerate(
                self.metadata.multiscales[0].datasets
            ):
                if dataset_meta.path == image:
                    self.metadata.multiscales[0].datasets[i] = DatasetMeta(
                        path=image, coordinate_transformations=transform
                    )
        else:
            raise ValueError(f"Key {image} not recognized.")
        self.dump_meta()

    def set_scale(
        self, image: str | Literal["*"], axis_name: str, new_scale: float
    ):
        """Set the scale for a named axis.
        Either one image array or the whole FOV.

        Parameters
        ----------
        image : str | Literal['*']
            Name of one image array (e.g. "0") to transform,
            or "*" for the whole FOV
        axis_name : str
            Name of the axis to set.
        new_scale : float
            Value of the new scale.
        """
        if new_scale <= 0:
            raise ValueError(
                f"New scale {axis_name}: {new_scale} is not positive!"
            )
        if image not in self and image != "*":
            raise KeyError(f"Image {image} not found.")
        axis_index = self.get_axis_index(axis_name)
        # Update scale while preserving existing transforms
        if image == "*":
            transforms = (
                self.metadata.multiscales[0].coordinate_transformations or []
            )
        else:
            for dataset_meta in self.metadata.multiscales[0].datasets:
                if dataset_meta.path == image:
                    transforms = dataset_meta.coordinate_transformations
                    break
        # Append old scale to metadata
        iohub_dict = {}
        if "iohub" in self.zattrs:
            iohub_dict = self.zattrs["iohub"]
        if "previous_transforms" not in iohub_dict:
            iohub_dict["previous_transforms"] = []
        iohub_dict["previous_transforms"].append(
            {
                "image": image,
                "transforms": [
                    t.model_dump(**TO_DICT_SETTINGS) for t in transforms
                ],
                "modified": datetime.now().isoformat(),
            }
        )
        self.zattrs["iohub"] = iohub_dict
        # Replace default identity transform with scale
        if transforms == [TransformationMeta(type="identity")]:
            transforms = [TransformationMeta(type="scale", scale=[1.0] * 5)]
        # Add scale transform if not present
        if not any([transform.type == "scale" for transform in transforms]):
            transforms.append(
                TransformationMeta(type="scale", scale=[1.0] * 5)
            )
        new_transforms = []
        for transform in transforms:
            if transform.type == "scale":
                old_scale = transform.scale[axis_index]
                transform.scale[axis_index] = new_scale
            new_transforms.append(transform)
        _logger.info(
            f"Updating scale for axis {axis_name} "
            f"from {old_scale} to {new_scale}."
        )
        self.set_transform(image, new_transforms)

    def set_contrast_limits(self, channel_name: str, window: WindowDict):
        """Set the contrast limits for a channel.

        Parameters
        ----------
        channel_name : str
            Name of the channel to set
        window : WindowDict
            Contrast limit (min, max, start, end)
        """
        channel_index = self.get_channel_index(channel_name)
        self.metadata.omero.channels[channel_index].window = window
        self.dump_meta()


class TiledPosition(Position):
    """Variant of the NGFF position node
    with convenience methods to create and access tiled arrays.
    Other parameters and attributes are the same as
    :py:class:`iohub.ngff.Position`.
    """

    _MEMBER_TYPE = TiledImageArray

    def make_tiles(
        self,
        name: str,
        grid_shape: tuple[int, int],
        tile_shape: tuple[int, ...],
        dtype: DTypeLike,
        transform: list[TransformationMeta] | None = None,
        chunk_dims: int = 2,
    ):
        """Make a tiled image array filled with zeros.
        Chunk size is inferred from tile shape.

        Parameters
        ----------
        name : str
            Name of the array.
        grid_shape : tuple[int, int]
            2-tuple of the tiling grid shape (rows, columns).
        tile_shape : tuple[int, ...]
            Shape of each tile (up to 5D).
        dtype : DTypeLike
            Data type in NumPy convention
        transform : list[TransformationMeta], optional
            List of coordinate transformations, by default None.
            Should be specified for a non-native resolution level.
        chunk_dims : int, optional
            Non-singleton dimensions of the chunksize,
            by default 2 (chunk by 2D (y, x) tile size).

        Returns
        -------
        TiledImageArray
        """
        xy_shape = tuple(
            int(i) for i in np.array(grid_shape) * np.array(tile_shape[-2:])
        )
        chunks = self._default_chunks(
            shape=tile_shape, last_data_dims=chunk_dims
        )
        return TiledImageArray.from_zarr_array(
            self.create_zeros(
                name=name,
                shape=tile_shape[:-2] + xy_shape,
                dtype=dtype,
                chunks=chunks,
                transform=transform,
            )
        )


class Well(NGFFNode):
    """The Zarr group level containing position groups.

    Parameters
    ----------
    group : zarr.Group
        Zarr heirarchy group object
    parse_meta : bool, optional
        Whether to parse NGFF metadata in `.zattrs`, by default True
    version : Literal["0.4", "0.5"]
        OME-NGFF specification version
    overwriting_creation : bool, optional
        Whether to overwrite or error upon creating an existing child item,
        by default False

    Attributes
    ----------
    version : Literal["0.4", "0.5"]
        OME-NGFF specification version
    zgroup : Group
        Root Zarr group holding arrays
    zattr : Attributes
        Zarr attributes of the group
    """

    _MEMBER_TYPE = Position

    def __init__(
        self,
        group: zarr.Group,
        parse_meta: bool = True,
        channel_names: list[str] | None = None,
        axes: list[AxisMeta] | None = None,
        version: Literal["0.4", "0.5"] = "0.4",
        overwriting_creation: bool = False,
    ):
        super().__init__(
            group=group,
            parse_meta=parse_meta,
            channel_names=channel_names,
            axes=axes,
            version=version,
            overwriting_creation=overwriting_creation,
        )

    def _parse_meta(self):
        if well_group_meta := self.maybe_wrapped_ome_attrs.get("well"):
            if "version" not in well_group_meta:
                well_group_meta["version"] = self.version
            self.metadata = WellGroupMeta(**well_group_meta)
        else:
            self._warn_invalid_meta()

    def dump_meta(self):
        """Dumps metadata JSON to the `.zattrs` file."""
        ome = {"well": self.metadata.model_dump(**TO_DICT_SETTINGS)}
        self._dump_ome(ome)

    def __getitem__(self, key: str):
        """Get a position member of the well.

        Parameters
        ----------
        key : str
            Name or path to the position.

        Returns
        -------
        Position
            Container object for the position group
        """
        return super().__getitem__(key)

    def _create_position_nosync(self, name: str, acquisition: int = 0):
        "create_position, but doesn't write the metadata yet."
        pos_grp = self._group.create_group(name, overwrite=self._overwrite)
        # build metadata
        image_meta = ImageMeta(acquisition=acquisition, path=pos_grp.basename)
        if not hasattr(self, "metadata"):
            self.metadata = WellGroupMeta(
                images=[image_meta], version=self.version
            )
        else:
            self.metadata.images.append(image_meta)
        return Position(group=pos_grp, parse_meta=False, **self._child_attrs)

    def create_position(self, name: str, acquisition: int = 0):
        """Creates a new position group in the well group.

        Parameters
        ----------
        name : str
            Name key of the new position
        acquisition : int, optional
            The index of the acquisition, by default 0
        """
        pos = self._create_position_nosync(name, acquisition=acquisition)
        self.dump_meta()
        return pos

    def positions(self):
        """Returns a generator that iterate over the name and value
        of all the positions in the well.

        Yields
        ------
        tuple[str, Position]
            Name and position object.
        """
        yield from self.iteritems()


class Row(NGFFNode):
    """The Zarr group level containing wells.

    Parameters
    ----------
    group : zarr.Group
        Zarr heirarchy group object
    parse_meta : bool, optional
        Whether to parse NGFF metadata in `.zattrs`, by default True
    version : Literal["0.4", "0.5"]
        OME-NGFF specification version
    overwriting_creation : bool, optional
        Whether to overwrite or error upon creating an existing child item,
        by default False

    Attributes
    ----------
    version : Literal["0.4", "0.5"]
        OME-NGFF specification version
    zgroup : Group
        Root Zarr group holding arrays
    zattr : Attributes
        Zarr attributes of the group
    """

    _MEMBER_TYPE = Well

    def __init__(
        self,
        group: zarr.Group,
        parse_meta: bool = True,
        channel_names: list[str] | None = None,
        axes: list[AxisMeta] | None = None,
        version: Literal["0.4", "0.5"] = "0.4",
        overwriting_creation: bool = False,
    ):
        super().__init__(
            group=group,
            parse_meta=parse_meta,
            channel_names=channel_names,
            axes=axes,
            version=version,
            overwriting_creation=overwriting_creation,
        )

    def __getitem__(self, key: str):
        """Get a well member of the row.

        Parameters
        ----------
        key : str
            Name or path to the well.

        Returns
        -------
        Well
            Container object for the well group
        """
        return super().__getitem__(key)

    def wells(self):
        """Returns a generator that iterate over the name and value
        of all the wells in the row.

        Yields
        ------
        tuple[str, Well]
            Name and well object.
        """
        yield from self.iteritems()

    def _parse_meta(self):
        # this node does not have NGFF metadata
        return


class Plate(NGFFNode):
    _MEMBER_TYPE = Row

    @classmethod
    def from_positions(
        cls,
        store_path: str | Path,
        positions: dict[str, Position],
    ) -> Plate:
        """Create a new HCS store from existing OME-Zarr stores
        by copying images and metadata from a dictionary of positions.

        .. warning: This assumes same channel names and axes across the FOVs
            and does not check for consistent shape and chunk size.

        Parameters
        ----------
        store_path : str | Path
            Path of the new store
        positions : dict[str, Position]
            Dictionary where keys are destination path names ('row/column/fov')
            and values are :py:class:`iohub.ngff.Position` objects.

        Returns
        -------
        Plate
            New plate with copied data

        Examples
        --------
        Combine an HCS-layout store and an FOV-layout store:

        >>> from iohub.ngff import open_ome_zarr, Plate

        >>> with open_ome_zarr("hcs.zarr") as old_plate:
        >>>     fovs = dict(old_plate.positions())

        >>> with open_ome_zarr("fov.zarr") as old_position:
        >>>     fovs["Z/1/0"] = old_position

        >>> new_plate = Plate.from_positions("combined.zarr", fovs)
        """
        # TODO: remove when zarr-python adds back `copy_store`
        raise NotImplementedError(
            "This method is disabled until upstream support is finalized: "
            "https://github.com/zarr-developers/zarr-python/issues/2407"
        )
        # get metadata from an arbitrary FOV
        # deterministic because dicts are ordered
        example_position = next(iter(positions.values()))
        plate = open_ome_zarr(
            store_path,
            layout="hcs",
            mode="w-",
            channel_names=example_position.channel_names,
            axes=example_position.axes,
            version=example_position.version,
        )
        for name, src_pos in positions.items():
            if not isinstance(src_pos, Position):
                raise TypeError(
                    f"Expected item type {type(Position)}, "
                    f"got {type(src_pos)}"
                )
            name = normalize_path(name)
            if name in plate.zgroup:
                raise FileExistsError(
                    f"Duplicate name '{name}' after path normalization."
                )
            row, col, fov = name.split("/")
            dst_pos = plate.create_position(row, col, fov)
            # overwrite position group
            _ = zarr.copy_store(
                src_pos.zgroup.store,
                plate.zgroup.store,
                source_path=src_pos.zgroup.name,
                dest_path=name,
                if_exists="replace",
            )
            dst_pos._parse_meta()
            dst_pos.metadata.omero.name = fov
            dst_pos.dump_meta()
        return plate

    def __init__(
        self,
        group: zarr.Group,
        parse_meta: bool = True,
        channel_names: list[str] | None = None,
        axes: list[AxisMeta] | None = None,
        name: str | None = None,
        acquisitions: list[AcquisitionMeta] | None = None,
        version: Literal["0.4", "0.5"] = "0.4",
        overwriting_creation: bool = False,
    ):
        super().__init__(
            group=group,
            parse_meta=parse_meta,
            channel_names=channel_names,
            axes=axes,
            version=version,
            overwriting_creation=overwriting_creation,
        )
        self._name = name
        self._acquisitions = (
            [AcquisitionMeta(id=0)] if not acquisitions else acquisitions
        )

    def _parse_meta(self):
        if plate_meta := self.maybe_wrapped_ome_attrs.get("plate"):
            _logger.debug(f"Loading HCS metadata from file: {plate_meta}")
            if "version" not in plate_meta:
                plate_meta["version"] = self.version
            self.metadata = PlateMeta(**plate_meta)
        else:
            self._warn_invalid_meta()
        for attr in ("_channel_names", "axes"):
            if not hasattr(self, attr):
                self._first_pos_attr(attr)

    def _first_pos_attr(self, attr: str):
        """Get attribute value from the first position."""
        name = " ".join(attr.split("_")).strip()
        msg = f"Cannot determine {name}:"
        try:
            row_grp = next(self.zgroup.groups())[1]
            well_grp = next(row_grp.groups())[1]
            pos_grp = next(well_grp.groups())[1]
        except StopIteration:
            _logger.warning(f"{msg} No position is found in the dataset.")
            return
        try:
            pos = Position(pos_grp)
            setattr(self, attr, getattr(pos, attr))
        except AttributeError:
            _logger.warning(f"{msg} Invalid metadata at the first position")

    def dump_meta(self, field_count: bool = False):
        """Dumps metadata JSON to the `.zattrs` file.

        Parameters
        ----------
        field_count : bool, optional
            Whether to count all the FOV/positions
            and populate the metadata field 'plate.field_count';
            this operation can be expensive if there are many positions,
            by default False
        """
        if field_count:
            self.metadata.field_count = len(list(self.positions()))
        ome = {"plate": self.metadata.model_dump(**TO_DICT_SETTINGS)}
        self._dump_ome(ome)

    def _auto_idx(
        self,
        name: str,
        index: int | None,
        axis_name: Literal["row", "column"],
    ):
        if index is not None:
            return index
        elif not hasattr(self, "metadata"):
            return 0
        else:
            part = ["row", "column"].index(axis_name)
            all_indices = []
            for well_index in self.metadata.wells:
                index = getattr(well_index, f"{axis_name}_index")
                if well_index.path.split("/")[part] == name:
                    return index
                all_indices.append(index)
            return max(all_indices) + 1

    def _build_meta(
        self,
        first_row_meta: PlateAxisMeta,
        first_col_meta: PlateAxisMeta,
        first_well_meta: WellIndexMeta,
    ):
        """Initiate metadata attribute."""
        self.metadata = PlateMeta(
            version=self.version,
            name=self._name,
            acquisitions=self._acquisitions,
            rows=[first_row_meta],
            columns=[first_col_meta],
            wells=[first_well_meta],
        )

    def create_well(
        self,
        row_name: str,
        col_name: str,
        row_index: int | None = None,
        col_index: int | None = None,
    ):
        """Creates a new well group in the plate.
        The new well will have empty group metadata,
        which will not be created until a position is written.

        Parameters
        ----------
        row_name : str
            Name key of the row
        col_name : str
            Name key of the column
        row_index : int, optional
            Index of the row,
            will be set by the sequence of creation if not provided,
            by default None
        col_index : int, optional
            Index of the column,
            will be set by the sequence of creation if not provided,
            by default None

        Returns
        -------
        Well
            Well node object
        """
        # normalize input
        row_name = normalize_path(row_name)
        col_name = normalize_path(col_name)
        if row_name in self:
            if col_name in self[row_name]:
                raise FileExistsError(
                    f"Well '{row_name}/{col_name}' already exists."
                )
        row_meta = PlateAxisMeta(name=row_name)
        col_meta = PlateAxisMeta(name=col_name)
        row_index = self._auto_idx(row_name, row_index, "row")
        col_index = self._auto_idx(col_name, col_index, "column")
        # build well metadata
        well_index_meta = WellIndexMeta(
            path="/".join([row_name, col_name]),
            row_index=row_index,
            column_index=col_index,
        )
        if not hasattr(self, "metadata"):
            self._build_meta(row_meta, col_meta, well_index_meta)
        else:
            self.metadata.wells.append(well_index_meta)
        # create new row if needed
        if row_name not in self:
            row_grp = self.zgroup.create_group(
                row_meta.name, overwrite=self._overwrite
            )
            if row_meta not in self.metadata.rows:
                self.metadata.rows.append(row_meta)
        else:
            row_grp = self[row_name].zgroup
        if col_meta not in self.metadata.columns:
            self.metadata.columns.append(col_meta)
        # create well
        well_grp = row_grp.create_group(col_name, overwrite=self._overwrite)
        self.dump_meta()
        return Well(group=well_grp, parse_meta=False, **self._child_attrs)

    def create_positions(
        self, positions: list[PositionSpec]
    ) -> list[Position]:
        """Creates multiple position groups in the plate efficiently.

        This is a vectorized version of :py:meth:`create_position` that creates
        multiple positions in a single call. Wells are created as needed and
        metadata is written in batches for better performance.

        Parameters
        ----------
        positions : list[PositionSpec]
            List of position specifications. Each tuple can have 3-6 elements:

            - 3 elements: ``(row_name, col_name, pos_name)``
            - 4 elements: ``(row_name, col_name, pos_name, row_index)``
            - 5 elements: ``(row_name, col_name, pos_name, row_index,
              col_index)``
            - 6 elements: ``(row_name, col_name, pos_name, row_index,
              col_index, acq_index)``

            Where:

            - row_name (str): Name key of the row
            - col_name (str): Name key of the column
            - pos_name (str): Name key of the position
            - row_index (int | None): Index of the row (auto-assigned if None
              or omitted)
            - col_index (int | None): Index of the column (auto-assigned if
              None or omitted)
            - acq_index (int): Index of the acquisition (defaults to 0 if
              omitted)

        Returns
        -------
        list[Position]
            List of created Position node objects

        See Also
        --------
        create_position : Create a single position group

        Examples
        --------
        Create multiple positions with automatic row/column indexing:

        >>> plate.create_positions([
        ...     ("A", "1", "0"),
        ...     ("A", "1", "1"),
        ...     ("A", "2", "0"),
        ... ])

        Create positions with explicit row/column indices:

        >>> plate.create_positions([
        ...     ("B", "3", "0", 1, 2),      # row_index=1, col_index=2
        ...     ("B", "3", "1", 1, 2),      # same well indices
        ... ])

        Create positions with specific acquisition indices:

        >>> plate.create_positions([
        ...     ("B", "3", "0", 1, 2, 0),   # acquisition 0
        ...     ("B", "3", "1", 1, 2, 1),   # acquisition 1
        ... ])
        """
        positions = deepcopy(positions)  # We may mutate contents
        wells = {}  # Track wells by path to avoid duplicate objects
        positions_out = []
        for r, c, p, *args in positions:
            # Parse out arguments
            well_args = args[:2]
            acquisition_index = 0  # Default value for create_position
            if len(args) == 3:
                acquisition_index = args[2]
            elif len(args) > 3:
                raise ValueError(
                    "Passed too many fields for a position: "
                    f"{(r, c, p, *args)}"
                )
            r = normalize_path(r)
            c = normalize_path(c)
            well_path = os.path.join(r, c)

            # Get or create well, ensuring we reuse the same object
            if well_path in wells:
                well = wells[well_path]
            elif well_path in self.zgroup:
                well = self[well_path]
                wells[well_path] = well
            else:
                well = self.create_well(r, c, *well_args)
                wells[well_path] = well

            positions_out.append(
                well._create_position_nosync(p, acquisition=acquisition_index)
            )
        for well in wells.values():
            well.dump_meta()
        return positions_out

    def create_position(
        self,
        row_name: str,
        col_name: str,
        pos_name: str,
        row_index: int | None = None,
        col_index: int | None = None,
        acq_index: int = 0,
    ):
        """Creates a new position group in the plate.
        The new position will have empty group metadata,
        which will not be created until an image is written.

        Parameters
        ----------
        row_name : str
            Name key of the row
        col_name : str
            Name key of the column
        row_index : int, optional
            Index of the row,
            will be set by the sequence of creation if not provided,
            by default None
        col_index : int, optional
            Index of the column,
            will be set by the sequence of creation if not provided,
            by default None
        acq_index : int, optional
            Index of the acquisition, by default 0

        Returns
        -------
        Position
            Position node object
        """
        row_name = normalize_path(row_name)
        col_name = normalize_path(col_name)
        well_path = os.path.join(row_name, col_name)
        if well_path in self.zgroup:
            well = self[well_path]
        else:
            well = self.create_well(
                row_name, col_name, row_index=row_index, col_index=col_index
            )
        return well.create_position(pos_name, acquisition=acq_index)

    def rows(self) -> Generator[tuple[str, Row], None, None]:
        """Returns a generator that iterate over the name and value
        of all the rows in the plate.

        Yields
        ------
        tuple[str, Row]
            Name and row object.
        """
        yield from self.iteritems()

    def wells(self) -> Generator[tuple[str, Well], None, None]:
        """Returns a generator that iterate over the path and value
        of all the wells (along rows, columns) in the plate.

        Yields
        ------
        [str, Well]
            Path and well object.
        """
        for _, row in self.rows():
            for _, well in row.wells():
                yield well.zgroup.path, well

    def positions(self) -> Generator[tuple[str, Position], None, None]:
        """Returns a generator that iterate over the path and value
        of all the positions (along rows, columns, and wells) in the plate.

        Yields
        ------
        [str, Position]
            Path and position object.
        """
        for _, well in self.wells():
            for _, position in well.positions():
                yield position.zgroup.path, position

    def rename_well(self, old: str, new: str):
        """Rename a well.

        Parameters
        ----------
        old : str
            Old name of well, e.g. "A/1"
        new : str
            New name of well, e.g. "B/2"
        """

        # normalize inputs
        old = normalize_path(old)
        new = normalize_path(new)
        old_row, old_column = old.split("/")
        new_row, new_column = new.split("/")
        new_row_meta = PlateAxisMeta(name=new_row)
        new_col_meta = PlateAxisMeta(name=new_column)

        # self.zgroup.move(old, new) # Not Implemented

        # raises ValueError if old well does not exist
        # or if new well already exists
        if old not in self.zgroup:
            raise FileNotFoundError(f"Well '{old}' does not exist.")
        if new in self.zgroup:
            raise FileExistsError(f"Well '{new}' already exists.")

        store_path = self.zgroup.store.root
        assert store_path.is_dir()

        old_path = store_path / old
        assert old_path.is_dir()

        new_path = store_path / new
        assert not new_path.parent.is_dir()

        shutil.move(
            str(old_path.parent), str(new_path.parent)
        )  # rename row path
        shutil.move(
            str(new_path.parent / old_column), str(new_path)
        )  # rename column path

        assert new in self.zgroup

        # update well metadata
        old_well_index = [
            well_name.path for well_name in self.metadata.wells
        ].index(old)
        self.metadata.wells[old_well_index].path = new
        new_well_names = [well.path for well in self.metadata.wells]

        # update row/col metadata
        # check for new row/col
        if new_row not in [row.name for row in self.metadata.rows]:
            self.metadata.rows.append(new_row_meta)
        if new_column not in [col.name for col in self.metadata.columns]:
            self.metadata.columns.append(new_col_meta)

        # check for empty row/col
        if old_row not in [well.split("/")[0] for well in new_well_names]:
            # delete empty row from zarr
            del self.zgroup[old_row]
            self.metadata.rows = [
                row for row in self.metadata.rows if row.name != old_row
            ]
        if old_column not in [well.split("/")[1] for well in new_well_names]:
            self.metadata.columns = [
                col for col in self.metadata.columns if col.name != old_column
            ]

        self.dump_meta()


def _check_file_mode(
    store_path: Path,
    mode: Literal["r", "r+", "a", "w", "w-"],
    disable_path_checking: bool,
) -> bool:
    if mode == "a":
        mode = "r+" if store_path.exists() else "w-"
    parse_meta = False
    if mode in ("r", "r+"):
        parse_meta = True
    elif mode == "w-":
        if store_path.exists():
            raise FileExistsError(store_path)
    elif mode == "w":
        if store_path.exists():
            if (
                ".zarr" not in str(store_path.resolve())
                and not disable_path_checking
            ):
                raise ValueError(
                    "Cannot overwrite a path that does not contain '.zarr', "
                    "use `disable_path_checking=True` if you are sure that "
                    f"{store_path} should be overwritten."
                )
            _logger.warning(f"Overwriting data at {store_path}")
    else:
        raise ValueError(f"Invalid persistence mode '{mode}'.")
    return parse_meta


def _detect_layout(meta_keys: list[str]) -> Literal["fov", "hcs"]:
    if "plate" in meta_keys:
        return "hcs"
    elif "multiscales" in meta_keys:
        return "fov"
    else:
        raise KeyError(
            "Dataset metadata keys ('plate'/'multiscales') not in "
            f"the found store metadata keys: {meta_keys}. "
            "Is this a valid OME-Zarr dataset?"
        )


@overload
def open_ome_zarr(
    store_path: str | Path,
    layout: Literal["auto"],
    mode: Literal["r", "r+", "a", "w", "w-"] = "r",
    channel_names: list[str] | None = None,
    axes: list[AxisMeta] | None = None,
    version: Literal["0.4", "0.5"] = "0.4",
    disable_path_checking: bool = False,
    **kwargs,
) -> Plate | Position | TiledPosition: ...


@overload
def open_ome_zarr(
    store_path: str | Path,
    layout: Literal["fov"],
    mode: Literal["r", "r+", "a", "w", "w-"] = "r",
    channel_names: list[str] | None = None,
    axes: list[AxisMeta] | None = None,
    version: Literal["0.4", "0.5"] = "0.4",
    disable_path_checking: bool = False,
    **kwargs,
) -> Position: ...


@overload
def open_ome_zarr(
    store_path: str | Path,
    layout: Literal["tiled"],
    mode: Literal["r", "r+", "a", "w", "w-"] = "r",
    channel_names: list[str] | None = None,
    axes: list[AxisMeta] | None = None,
    version: Literal["0.4", "0.5"] = "0.4",
    disable_path_checking: bool = False,
    **kwargs,
) -> TiledPosition: ...


@overload
def open_ome_zarr(
    store_path: str | Path,
    layout: Literal["hcs"],
    mode: Literal["r", "r+", "a", "w", "w-"] = "r",
    channel_names: list[str] | None = None,
    axes: list[AxisMeta] | None = None,
    version: Literal["0.4", "0.5"] = "0.4",
    disable_path_checking: bool = False,
    **kwargs,
) -> Plate: ...


def open_ome_zarr(
    store_path: str | Path,
    layout: Literal["auto", "fov", "hcs", "tiled"] = "auto",
    mode: Literal["r", "r+", "a", "w", "w-"] = "r",
    channel_names: list[str] | None = None,
    axes: list[AxisMeta] | None = None,
    version: Literal["0.4", "0.5"] = "0.4",
    disable_path_checking: bool = False,
    **kwargs,
) -> Plate | Position | TiledPosition:
    """Convenience method to open OME-Zarr stores.

    Parameters
    ----------
    store_path : str | Path
        File path to the Zarr store to open
    layout: Literal["auto", "fov", "hcs", "tiled"], optional
        NGFF store layout:
        "auto" will infer layout from existing metadata
        (cannot be used for creation);
        "fov" opens a single position/FOV node;
        "hcs" opens the high-content-screening multi-fov hierarchy;
        "tiled" opens a "fov" layout with tiled image array
        (cannot be automatically inferred since this not NGFF-specified);
        by default "auto"
    mode : Literal["r", "r+", "a", "w", "w-"], optional
        Persistence mode:
        'r' means read only (must exist);
        'r+' means read/write (must exist);
        'a' means read/write (create if doesn't exist);
        'w' means create (overwrite if exists);
        'w-' means create (fail if exists),
        by default "r".
    channel_names : list[str], optional
        Channel names used to create a new data store,
        ignored for existing stores,
        by default None
    axes : list[AxisMeta], optional
        OME axes metadata, by default None:

        .. code-block:: text

            [AxisMeta(name='T', type='time', unit='second'),
            AxisMeta(name='C', type='channel', unit=None),
            AxisMeta(name='Z', type='space', unit='micrometer'),
            AxisMeta(name='Y', type='space', unit='micrometer'),
            AxisMeta(name='X', type='space', unit='micrometer')]

    version : Literal["0.4", "0.5"], optional
        OME-NGFF version, by default "0.4"
    disable_path_checking : bool, optional
        Whether to allow overwriting a path that does not contain '.zarr',
        by default False

        .. warning::
            This can lead to severe data loss
            if the input path is not checked carefully.

    kwargs : dict, optional
        Keyword arguments to underlying NGFF node constructor,
        by default None


    Returns
    -------
    Dataset
        NGFF node object
        (:py:class:`iohub.ngff.Position`,
        :py:class:`iohub.ngff.Plate`,
        or :py:class:`iohub.ngff.TiledPosition`)
    """
    store_path = Path(store_path)
    parse_meta = _check_file_mode(
        store_path, mode, disable_path_checking=disable_path_checking
    )
    root = _open_store(store_path, mode, version)
    meta_keys = root.attrs.keys() if parse_meta else []
    if "ome" in meta_keys:
        meta_keys = root.attrs["ome"].keys()
        version = root.attrs["ome"].get("version", version)
    if layout == "auto":
        if parse_meta:
            layout = _detect_layout(meta_keys)
        else:
            raise ValueError(
                "Store layout must be specified when creating a new dataset."
            )
    msg = f"Specified layout '{layout}' does not match existing metadata."
    if layout in ("fov", "tiled"):
        if parse_meta and "multiscales" not in meta_keys:
            raise ValueError(msg)
        node = TiledPosition if layout == "tiled" else Position
    elif layout == "hcs":
        if parse_meta and "plate" not in meta_keys:
            raise ValueError(msg)
        node = Plate
    else:
        raise ValueError(f"Unknown layout: {layout}")
    return node(
        group=root,
        parse_meta=parse_meta,
        channel_names=channel_names,
        axes=axes,
        version=version,
        **kwargs,
    )
