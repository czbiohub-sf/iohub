# TODO: remove this in the future (PEP deferred for 3.11, now 3.12?)
from __future__ import annotations

import os, logging
from numcodecs import Blosc
import zarr
import numpy as np
from pydantic import ValidationError

from iohub.ngff_meta import *
from iohub.lf_utils import channel_display_settings

from typing import TYPE_CHECKING, Union, Tuple, List, Dict, Literal, Generator
from numpy.typing import NDArray, DTypeLike

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath


def _pad_shape(shape: tuple[int], target: int = 5):
    """Pad shape tuple to a target length."""
    pad = target - len(shape)
    return (1,) * pad + shape


def open_store(
    store_path: StrOrBytesPath,
    mode: Literal["r", "r+", "a", "w", "w-"],
    version: Literal["0.1", "0.4"],
    synchronizer=None,
):
    if not os.path.isdir(store_path) and mode in ("r", "r+"):
        raise FileNotFoundError(
            f"Dataset directory not found at {store_path}."
        )
    if not version == "0.4":
        logging.warn(
            "\n".join(
                "IOHub is only tested against OME-NGFF v0.4.",
                f"Requested version {version} may not work properly.",
            )
        )
        dimension_separator = None
    else:
        dimension_separator = "/"
    try:
        store = zarr.DirectoryStore(
            store_path, dimension_separator=dimension_separator
        )
        root = zarr.open_group(store, mode=mode, synchronizer=synchronizer)
    except:
        raise FileNotFoundError(
            f"Cannot open Zarr root group at {store_path}."
        )
    return root


class NGFFNode:
    """A node (group level in Zarr) in an NGFF dataset."""

    _MEMBER_TYPE = None

    def __init__(
        self,
        group: zarr.Group,
        parse_meta: bool = True,
        version: Literal["0.1", "0.4"] = "0.4",
        overwriting_creation: bool = False,
    ):
        self._group = group
        self._overwrite = overwriting_creation
        self._version = version
        if parse_meta:
            self._parse_meta()

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
    def version(self):
        """NGFF version"""
        return self._version

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

    def __getitem__(self, key):
        key = zarr.util.normalize_storage_path(key)
        if key in self._member_names:
            return self._MEMBER_TYPE(self._group[key])
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __delitem__(self, key):
        key = zarr.util.normalize_storage_path(key)
        if key in self._member_names:
            del self[key]

    def __contains__(self, key):
        key = zarr.util.normalize_storage_path(key)
        return key in self._member_names

    def __iter__(self):
        for key in self._member_names:
            yield key

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

    def _warn_invalid_meta(self):
        msg = "Zarr group at {} does not have valid metadata for {}".format(
            self._group.path, type(self)
        )
        logging.warn(msg)

    def _parse_meta(self):
        """Parse and set NGFF metadata from `.zattrs`."""
        raise NotImplementedError

    def dump_meta(self):
        """Dumps metadata JSON to the `.zattrs` file."""
        self.zattrs.update(**self.metadata.dict(**TO_DICT_SETTINGS))

    def close(self):
        """Close Zarr store."""
        self._group.store.close()


class ImageArray(zarr.Array):
    """Container object for image stored as a zarr array (up to 5D)"""

    def __init__(self, zarray: zarr.Array):
        super().__init__(
            store=zarray._store,
            path=zarray._path,
            read_only=zarray._read_only,
            chunk_store=zarray._chunk_store,
            synchronizer=zarray._synchronizer,
            cache_metadata=zarray._cache_metadata,
            cache_attrs=zarray._attrs.cache,
            partial_decompress=zarray._partial_decompress,
            write_empty_chunks=zarray._write_empty_chunks,
            zarr_version=zarray._version,
            meta_array=zarray._meta_array,
        )
        self._get_dims()

    def _get_dims(self):
        (
            self.frames,
            self.channels,
            self.slices,
            self.height,
            self.width,
        ) = _pad_shape(self.shape, target=5)

    def numpy(self):
        """Return the whole image as an in-RAM NumPy array.
        `self.numpy()` is equivalent to `self[:]`."""
        return self[:]

    def downscale(self):
        raise NotImplementedError

    def tensorstore(self):
        raise NotImplementedError


class Position(NGFFNode):
    """The Zarr group level directly containing multiscale image arrays."""

    _MEMBER_TYPE = ImageArray
    _DEFAULT_AXES = [
        AxisMeta(name="T", type="time", unit="second"),
        AxisMeta(name="C", type="channel"),
        *[
            AxisMeta(name=i, type="space", unit="micrometer")
            for i in ("Z", "Y", "X")
        ],
    ]

    def __init__(
        self,
        group: zarr.Group,
        parse_meta: bool = True,
        channel_names: list[str] = None,
        axes: list[AxisMeta] = None,
        overwriting_creation: bool = False,
    ):
        if channel_names:
            self._channel_names = channel_names
        elif not parse_meta:
            raise ValueError(
                "Channel names need to be provided or in metadata."
            )
        super().__init__(
            group, parse_meta, overwriting_creation=overwriting_creation
        )
        self.axes = axes if axes else self._DEFAULT_AXES

    def _parse_meta(self):
        multiscales = self.zattrs.get("multiscales")
        omero = self.zattrs.get("omero")
        if multiscales and omero:
            try:
                self.metadata = ImagesMeta(
                    multiscales=multiscales, omero=omero
                )
                self._channel_names = [
                    c.label for c in self.metadata.omero.channels
                ]
                self.axes = self.metadata.multiscales[0].axes
            except ValidationError:
                self._warn_invalid_meta()
        else:
            self._warn_invalid_meta()

    @property
    def _storage_options(self):
        return {
            "compressor": Blosc(
                cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE
            ),
            "overwrite": self._overwrite,
        }

    @property
    def _member_names(self):
        return self.array_keys()

    @property
    def channel_names(self):
        return self._channel_names

    def __getitem__(self, key: Union[int, str]):
        """Get an image array member of the position.
        E.g. Raw-coordinates image, a multi-scale level, or labels

        Parameters
        ----------
        key : Union[int, str]
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
        key = zarr.util.normalize_storage_path(key)
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
        for key in self.array_keys():
            yield key, self[key]

    def create_image(
        self,
        name: str,
        data: NDArray,
        chunks: tuple[int] = None,
        transform: List[TransformationMeta] = None,
    ):
        """Create a new image array in the position.

        Parameters
        ----------
        name : str
            Name key of the new image.
        data : NDArray
            Image data.
        chunks : tuple[int], optional
            Chunk size, by default None.
            ZYX stack size will be used if not specified.
        transform : List[TransformationMeta], optional
            List of coordinate transformations, by default None.
            Should be specified for a non-native resolution level.

        Returns
        -------
        ImageArray
            Container object for image stored as a zarr array (up to 5D)
        """
        if not chunks:
            chunks = data.shape[-min(3, len(data.shape)) :]
            chunks = _pad_shape(chunks, target=len(data.shape))
        img_arr = ImageArray(
            self._group.array(
                name, data, chunks=chunks, **self._storage_options
            )
        )
        self._create_image_meta(img_arr.basename, transform=transform)
        return img_arr

    def _create_image_meta(
        self,
        name: str,
        transform: List[TransformationMeta] = None,
        extra_meta: dict = None,
    ):
        if not transform:
            transform = [TransformationMeta(type="identity")]
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
                        coordinateTransformations=transform,
                        metadata=extra_meta,
                    )
                ],
                omero=self._omero_meta(id=0, name=self._group.basename),
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
        clims: List[Tuple[float, float, float, float]] = None,
    ):
        if not clims:
            clims = [None] * 4
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

    def _find_axis(self, axis_name):
        for i, axis in enumerate(self.axes):
            if axis.name == axis_name:
                return i
        return None

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
            raise ValueError(f"Channel name {chan_name} already exists.")
        self._channel_names.append(chan_name)
        if resize_arrays:
            ch_ax = self._find_axis("channel")
            if ch_ax is None:
                raise KeyError(
                    "Axis 'channel' does not exist."
                    + "Please update `self.axes` first."
                )
            for _, img in self.images():
                shape = img.shape
                if ch_ax < len(shape):
                    shape[ch_ax] += 1
                # prepend axis
                elif ch_ax == len(shape):
                    shape = _pad_shape(shape, target=len(shape) + 1)
                else:
                    raise IndexError(
                        f"Cannot infer channel axis for shape {shape}."
                    )
                img.resize(shape)
        if "omero" in self.metadata.dict().keys():
            self.metadata.omero.channels.append(
                channel_display_settings(chan_name)
            )
            self.dump_meta()


class Well(NGFFNode):
    _MEMBER_TYPE = Position
    pass


class Row(NGFFNode):
    _MEMBER_TYPE = Well
    pass


class Plate(NGFFNode):
    def __init__(self, group: zarr.Group, parse_meta: bool = True):
        super().__init__(group, parse_meta)


class Dataset:
    """Mix in file mode class method for `NGFFNode` subclasses"""

    @classmethod
    def open(
        cls,
        store_path: StrOrBytesPath,
        mode: Literal["r", "r+", "a", "w", "w-"] = "r",
        channel_names: List[str] = None,
        axes: list[AxisMeta] = None,
        version: Literal["0.1", "0.4"] = "0.4",
        synchronizer: Union[
            zarr.ThreadSynchronizer, zarr.ProcessSynchronizer
        ] = None,
    ):
        """Convenience method to open Zarr stores.
        Uses default parameters to initiate readers/writers. Initiate manually to alter them.

        Parameters
        ----------
        store_path : StrOrBytesPath
            Path to the Zarr store to open
        mode : Literal["r+", "a", "w-"], optional
            mode : Literal["r", "r+", "a", "w", "w-"], optional
            Persistence mode:
            'r' means read only (must exist);
            'r+' means read/write (must exist);
            'a' means read/write (create if doesn't exist);
            'w' means create (overwrite if exists);
            'w-' means create (fail if exists),
            by default "r".
        channel_names : List[str], optional
            Channel names used to create a new data store, ignored for existing stores,
            by default None
        axes : list[AxisMeta], optional
            OME axes metadata, by default:
            ```
            [AxisMeta(name='T', type='time', unit='second'),
            AxisMeta(name='C', type='channel', unit=None),
            AxisMeta(name='Z', type='space', unit='micrometer'),
            AxisMeta(name='Y', type='space', unit='micrometer'),
            AxisMeta(name='X', type='space', unit='micrometer')]
            ````
        version : Literal["0.1", "0.4"], optional
            OME-NGFF version, by default "0.4"
        synchronizer : object, optional
            Zarr thread or process synchronizer, by default None


        Returns
        -------
        Dataset
            NGFF dataset object (`OMEZarr` or `HCSZarr')
        """
        if mode == "a":
            mode = ("w-", "r+")[int(os.path.exists(store_path))]
        parse_meta = False
        if mode in ("r", "r+"):
            parse_meta = True
        elif mode == "w-":
            if os.path.exists(store_path):
                raise FileExistsError(store_path)
        elif mode == "w":
            logging.warn(f"Overwriting data at {store_path}")
        else:
            raise ValueError(f"Invalid persistence mode '{mode}'.")
        root = open_store(store_path, mode, version, synchronizer)
        return cls(
            root=root,
            parse_meta=parse_meta,
            channel_names=channel_names,
            axes=axes,
        )

    def __init__(
        self,
        root: zarr.Group,
        parse_meta: bool = True,
        channel_names: list[str] = None,
        axes: list[AxisMeta] = None,
    ):
        # this should call `Position.__init__()` for `OMEZarr`
        # or `Plate.__init__()` for `HCSZarr`
        super().__init__(
            group=root,
            parse_meta=parse_meta,
            channel_names=channel_names,
            axes=axes,
        )


class OMEZarr(Dataset, Position):
    """Generic OME-Zarr dataset container.

    Parameters
    ----------
    root : zarr.Group
        The root group of the Zarr store, dimension separator should be '/'
    channel_names: List[str]
        Names of all the channels present in data ordered according to channel indices
    version : Literal["0.1", "0.4"], optional
        OME-NGFF version, default to "0.4" if not provided
    arr_name : str, optional
        Base name of the arrays, by default '0'
    axes : List[AxisMeta], optional
        OME axes metadata, by default:
        ```
        [AxisMeta(name='T', type='time', unit='second'),
        AxisMeta(name='C', type='channel', unit=None),
        AxisMeta(name='Z', type='space', unit='micrometer'),
        AxisMeta(name='Y', type='space', unit='micrometer'),
        AxisMeta(name='X', type='space', unit='micrometer')]
        ````
    """

    def __init__(
        self,
        root: zarr.Group,
        parse_meta: bool = True,
        channel_names: list[str] = None,
        axes: list[AxisMeta] = None,
        version: Literal["0.1", "0.4"] = "0.4",
    ):
        super().__init__(
            root=root,
            parse_meta=parse_meta,
            channel_names=channel_names,
            axes=axes,
        )
        self._version = version

    def require_array(
        self,
        group: zarr.Group,
        name: str,
        zyx_shape: Tuple[int],
        dtype: DTypeLike,
        chunks: Tuple[int] = None,
    ):
        """Create a new array filled with zeros if it does not exist.

        Parameters
        ----------
        group : zarr.Group
            Parent zarr group
        name : str
            name key of the array
        zyx_shape : Tuple[int]
            Shape of the z-stack (Z, Y, X) in the array
        dtype : DTypeLike
            Data type
        chunks : Tuple[int], optional
            Chunk size for the new array if not present, by default a z-stack (1, 1, Z, Y, X)

        Returns
        -------
        Array
            Zarr array. Zero-filled with shape (1, 1, Z, Y, X) if created.

        Raises
        ------
        ValueError
        """
        if len(zyx_shape) != 3:
            raise ValueError("Shape {shape} does not have 3 dimensions.")
        value = group.get(name)
        if value and not isinstance(value, zarr.Array):
            raise FileExistsError(
                f"Name '{name}' maps to a non-array object of type {type(value)}."
            )
        elif isinstance(value, zarr.Array):
            if value.shape[-3:] == zyx_shape:
                return value
            else:
                raise FileExistsError(
                    f"An array exists with incompatible shape {value.shape}."
                )
        elif value is None:
            if not chunks:
                chunks = (1, 1) + zyx_shape
            return group.zeros(
                name,
                shape=(1, 1, *zyx_shape),
                chunks=chunks,
                dtype=dtype,
                **self._storage_options,
            )

    def write_zstack(
        self,
        data: NDArray,
        group: zarr.Group,
        time_index: int,
        channel_index: int,
        transform: List[TransformationMeta] = None,
        name: str = None,
        auto_meta=True,
        additional_meta: dict = None,
    ):
        """Write a z-stack with OME-NGFF metadata.

        Parameters
        ----------
        data : NDArray
            Image data with shape (Z, Y, X)
        group : zarr.Group
            Parent Zarr group
        time_index : int
            Time index
        channel_index : int
            Channel index
        transform: List[TransformationMeta], optional
            Coordinate transformations (see iohub.ngff_meta.TransformationMeta) for the array, by default identity
        name : str, optional
            Name key of the 5D array, by default None
        auto_meta : bool, optional
            Whether to track and store metadata automatically, by default True
        additional_meta : dict, optional
            Additional metadata, by default None
        """
        if len(data.shape) < 3:
            raise ValueError("Data has less than 3 dimensions.")
        if time_index < 0 or channel_index < 0:
            raise ValueError("Time and channel indices must be non-negative.")
        if not name:
            name = self.arr_name
        zyx_shape = data.shape[-3:]
        # get 5D array
        array = self.require_array(
            group, name, zyx_shape=zyx_shape, dtype=data.dtype
        )
        if time_index >= array.shape[0] or channel_index >= array.shape[1]:
            array.resize(
                max(time_index + 1, array.shape[0]),
                max(channel_index + 1, array.shape[1]),
                *zyx_shape,
            )
        # write data
        array[time_index, channel_index] = data
        if auto_meta:
            self._dump_zstack_meta(name, transform, additional_meta)

    def _dump_zstack_meta(
        self,
        name: str,
        transform: List[TransformationMeta],
        additional_meta: dict,
    ):
        dataset_meta = self._dataset_meta(name, transform=transform)
        if "multiscales" not in self.root.attrs:
            multiscales = [
                MultiScaleMeta(
                    version=self.version,
                    axes=self.axes,
                    datasets=[dataset_meta],
                    metadata=additional_meta,
                )
            ]
            omero = self._omero_meta(0, self.root)
            self.images_meta = ImagesMeta(multiscales=multiscales, omero=omero)
        else:
            if (
                dataset_meta.path
                not in self.images_meta.multiscales[0].get_dataset_paths()
            ):
                self.images_meta.multiscales[0].datasets.append(dataset_meta)
        self.root.attrs.put(self.images_meta.dict(**TO_DICT_SETTINGS))


class HCSWriter(Dataset, Plate):
    """High-content screening OME-Zarr writer instance for an existing Zarr store.

    Parameters
    ----------
    root : zarr.Group
        The root group of the Zarr store, dimension separator should be '/'
    channel_names: List[str]
        Names of all the channels present in data ordered according to channel indices
    plate_name: str, optional
        Name of the plate, by default None
    version : Literal["0.1", "0.4"], optional
        OME-NGFF version, by default 0.4
    arr_name : str, optional
        Base name of the arrays, by default '0'
    axes : List[AxisMeta], optional
        OME axes metadata, by default:
        `
        [{'name': 'T', 'type': 'time', 'unit': 'second'},
        {'name': 'C', 'type': 'channel'},
        {'name': 'Z', 'type': 'space', 'unit': 'micrometer'},
        {'name': 'Y', 'type': 'space', 'unit': 'micrometer'},
        {'name': 'X', 'type': 'space', 'unit': 'micrometer'}]
        `
    """

    @classmethod
    def from_reader(
        cls,
        reader,
        detect_arr_name: bool = True,
        detect_layout: bool = True,
    ):
        # TODO: update reader API to use `reader.store` instead of `reader.root.store`
        reader.root.store.close()
        root = zarr.open(
            zarr.DirectoryStore(
                str(reader.root.store.path), dimension_separator="/"
            ),
            mode="a",
        )
        writer = cls(
            root=root,
            channel_names=reader.channel_names,
            plate_name=root.attrs.get("name"),
            version=reader.plate_meta.version,
            axes=reader.axes,
            acquisitions=reader.plate_meta.acquisitions,
        )
        if detect_arr_name:
            writer.arr_name = reader.arr_name
        if detect_layout:
            writer.rows = reader.rows_meta
            writer.columns = reader.columns_meta
            writer.wells = reader.wells_meta
            writer.positions = reader.positions_meta
            writer.plate_meta = reader.plate_meta
        return writer

    def __init__(
        self,
        root: zarr.Group,
        channel_names: List[str],
        plate_name: str = None,
        version: Literal["0.1", "0.4"] = "0.4",
        arr_name: str = "0",
        axes: List[AxisMeta] = None,
        acquisitions: List[AcquisitionMeta] = None,
    ):
        super().__init__(root, channel_names, version, arr_name, axes)
        self.plate_name = plate_name
        self.plate_meta: PlateMeta = None
        self.acquisitions = (
            [AcquisitionMeta(id=0)] if not acquisitions else acquisitions
        )
        self.rows: Dict[str, Union[PlateAxisMeta, int, float]] = {}
        self.columns: Dict[str, Union[PlateAxisMeta, int, float]] = {}
        self.wells: Dict[str, Union[WellIndexMeta, int, float]] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}

    @property
    def row_names(self) -> List[str]:
        """Row names in the plate (sub-groups under root)"""
        return self._rel_keys("")

    def get_cols_in_row(self, row_name: str) -> List[str]:
        """Get column names in a row (wells).

        Parameters
        ----------
        row_name : str
            Name path of the parent row, e.g. `"A"`, `"H"`

        Returns
        -------
        List[str]
            list of the names of the columns
        """
        return self._rel_keys(row_name)

    def get_pos_in_well(self, well_name: str) -> List[str]:
        """Get column names in a row (wells).

        Parameters
        ----------
        well_name : str
            Name path of the well relative to the root, e.g. `"A/1"`, `"H/12"`

        Returns
        -------
        List[str]
            list of the names of the positions
        """
        return self._rel_keys(well_name)

    @property
    def plate_layout(self) -> Dict[str, List[str]]:
        """Names of rows and columns of non-empty wells.

        Returns
        -------
        Dict[str, List[str]]
            {"A": ["1", ...],...}
        """
        return {
            row_name: self.get_cols_in_row() for row_name in self.row_names
        }

    def require_row(
        self, name: str, index: int = None, overwrite: bool = False
    ):
        """Creates a row in the hierarchy (first level below zarr root) if it does not exist.

        Parameters
        ----------
        name : str
            Name of the row
        index : int, optional
            Unique index of the new row, by default incremented by 1
        overwrite : bool, optional
            Delete all existing data in the row group, by default false

        Returns
        -------
        Group
            Zarr group for the required row
        """
        if name in self.rows:
            if index:
                existing_id = self.rows[name]["id"]
                if existing_id != index:
                    raise ValueError(
                        f"Requested index {index} conflicts with existing row {name} index {existing_id}."
                    )
        else:
            if not index:
                index = len(self.rows)
            self.rows[name] = {
                "id": index,
                "meta": PlateAxisMeta(name=name),
            }
        return self.root.require_group(name, overwrite=overwrite)

    def require_well(
        self,
        row_name: str,
        col_name: str,
        row_index: int = None,
        col_index: int = None,
        overwrite: bool = False,
    ):
        """Creates a well ('row_name/col_name') in the hierarchy if it does not exist.
        Will also create the parent row if it does not exist

        Parameters
        ----------
        row_name : str
            Name of the parent row
        col_name : str
            Name of the column
        row_index: int, optional
            Unique index of the new row, by default incremented by 1
        col_index: int, optional
            Unique index of the new column, by default incremented by 1
        overwrite : bool, optional
            Delete all existing data in the row group, by default false

        Returns
        -------
        Group
            Zarr group for the required well
        """
        well_name = os.path.join(row_name, col_name)
        row = self.require_row(row_name, index=row_index)
        if col_name in self.columns and col_index:
            _id = self.columns[col_name]["id"]
            if _id != col_index:
                raise ValueError(
                    f"Requested index {col_index} conflicts with existing column {col_name} index {_id}."
                )
        else:
            if not col_index:
                col_index = len(self.columns)
            self.columns[col_name] = {
                "id": col_index,
                "meta": PlateAxisMeta(name=col_name),
            }
        well = row.require_group(col_name, overwrite=overwrite)
        if well.name not in self.wells:
            self.wells[well.name] = {
                "meta": WellIndexMeta(
                    path=well_name,
                    rowIndex=self.rows[row_name]["id"],
                    columnIndex=self.columns[col_name]["id"],
                ),
                "positions": [],
                "image_meta_list": [],
            }
        return well

    def require_position(
        self, row: str, column: str, fov: str, acq_id: int = 0, **kwargs
    ):
        """Create a row, a column, and a FOV/position if they do not exist.

        Parameters
        ----------
        row : str
            Name key of the row, e.g. 'A'
        column : str
            Name key of the column, e.g. '12'
        fov : str
            Name key of the FOV/position, e.g. '0'
        acq_id : int, optional
            Acquisition ID, by default 0
        **kwargs :
            Keyword arguments for `require_well()` and `require_position`
        """
        well = self.require_well(row, column, **kwargs)
        position = well.require_group(fov)
        self.positions[position.name] = {"id": len(self.positions)}
        if position.name not in self.wells[well.name]["positions"]:
            image_meta = ImageMeta(acquisition=acq_id, path=position.basename)
            self.wells[well.name]["image_meta_list"].append(image_meta)
            self.wells[well.name]["positions"].append(position.name)
        return position

    def write_zstack(
        self,
        data: NDArray,
        group: zarr.Group,
        time_index: int,
        channel_index: int,
        transform: List[TransformationMeta] = None,
        name: str = None,
        auto_meta=True,
        additional_meta: dict = None,
    ):
        if not name:
            name = self.arr_name
        super().write_zstack(
            data,
            group,
            time_index,
            channel_index,
            name=name,
            auto_meta=False,
        )
        if auto_meta:
            self._dump_zstack_meta(group, name, transform, additional_meta)
        well = self.root.get(os.path.dirname(group.name))
        self._dump_well_meta(well)
        self._dump_plate_meta()

    def _dump_zstack_meta(
        self,
        position: zarr.Group,
        name: str,
        transform: List[TransformationMeta],
        additional_meta: dict,
    ):
        dataset_meta = self._dataset_meta(name, transform=transform)
        if "attrs" not in self.positions[position.name]:
            multiscales = [
                MultiScaleMeta(
                    version=self.version,
                    axes=self.axes,
                    datasets=[dataset_meta],
                    metadata=additional_meta,
                )
            ]
            id = self.positions[position.name]["id"]
            omero = self._omero_meta(id, position)
            images_meta = ImagesMeta(multiscales=multiscales, omero=omero)
            self.positions[position.name]["attrs"] = images_meta
        else:
            if (
                dataset_meta.path
                not in self.positions[position.name]["attrs"]
                .multiscales[0]
                .get_dataset_paths()
            ):
                self.positions[position.name]["attrs"].multiscales[
                    0
                ].datasets.append(dataset_meta)
        position.attrs.put(
            self.positions[position.name]["attrs"].dict(**TO_DICT_SETTINGS)
        )

    def _dump_plate_meta(self):
        self.plate_meta = PlateMeta(
            version=self.version,
            name=self.plate_name,
            acquisitions=self.acquisitions,
            rows=[row["meta"] for _, row in self.rows.items()],
            columns=[col["meta"] for _, col in self.columns.items()],
            wells=[well["meta"] for _, well in self.wells.items()],
            field_count=len(self.positions),
        )
        self.root.attrs["plate"] = self.plate_meta.dict(**TO_DICT_SETTINGS)

    def _dump_well_meta(self, well: zarr.Group):
        well_group_meta = WellGroupMeta(
            version=self.version,
            images=self.wells[well.name]["image_meta_list"],
        )
        well.attrs["well"] = well_group_meta.dict(**TO_DICT_SETTINGS)
