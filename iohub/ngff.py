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

    def __len__(self):
        return len(self._member_names)

    def __getitem__(self, key):
        key = zarr.util.normalize_storage_path(key)
        levels = key.count("/")
        item_type = self._MEMBER_TYPE
        for _ in range(levels):
            item_type = item_type._MEMBER_TYPE
        if key in self._member_names:
            return item_type(self._group[key])
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __delitem__(self, key):
        """.. Warning: this does NOT clean up metadata!"""
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

    def iteritems(self):
        for key in self._member_names:
            try:
                yield key, self[key]
            except:
                logging.warn(
                    "Skipped item at {}: invalid {}.".format(
                        key, type(self._MEMBER_TYPE)
                    )
                )

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
        raise NotImplementedError

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
    version : Literal["0.1", "0.4"]
        OME-NGFF specification version
    zgroup : Group
        Root Zarr group holding arrays
    zattr : Attributes
        Zarr attributes of the group
    channel_names : List[str]
        Name of the channels
    axes : List[AxisMeta]
        Axes metadata
    """

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
        version: Literal["0.1", "0.4"] = "0.4",
        overwriting_creation: bool = False,
    ):
        if channel_names:
            self._channel_names = channel_names
        elif not parse_meta:
            raise ValueError(
                "Channel names need to be provided or in metadata."
            )
        super().__init__(
            group,
            parse_meta,
            version=version,
            overwriting_creation=overwriting_creation,
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

    def dump_meta(self):
        """Dumps metadata JSON to the `.zattrs` file."""
        self.zattrs.update(**self.metadata.dict(**TO_DICT_SETTINGS))

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
        yield from self.iteritems()

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

    def _find_axis(self, axis_type):
        for i, axis in enumerate(self.axes):
            if axis.type == axis_type:
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
        if "omero" in self.metadata.dict().keys():
            self.metadata.omero.channels.append(
                channel_display_settings(chan_name)
            )
            self.dump_meta()


class Well(NGFFNode):
    """The Zarr group level containing position groups.

    Parameters
    ----------
    group : zarr.Group
        Zarr heirarchy group object
    parse_meta : bool, optional
        Whether to parse NGFF metadata in `.zattrs`, by default True
    version : Literal["0.1", "0.4"]
        OME-NGFF specification version
    overwriting_creation : bool, optional
        Whether to overwrite or error upon creating an existing child item,
        by default False

    Attributes
    ----------
    version : Literal["0.1", "0.4"]
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
        version: Literal["0.1", "0.4"] = "0.4",
        overwriting_creation: bool = False,
    ):
        super().__init__(
            group=group,
            parse_meta=parse_meta,
            version=version,
            overwriting_creation=overwriting_creation,
        )

    def _parse_meta(self):
        well_group_meta = self.zattrs.get("well")
        if well_group_meta:
            self.metadata = WellGroupMeta(**well_group_meta)
        else:
            self._warn_invalid_meta()

    def dump_meta(self):
        """Dumps metadata JSON to the `.zattrs` file."""
        self.zattrs.update({"well": self.metadata.dict(**TO_DICT_SETTINGS)})

    @property
    def _member_names(self):
        return self.group_keys()

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

    def create_position(self, name: str, acquisition: int = 0):
        """Creates a new position group in the well group.

        Parameters
        ----------
        name : str
            Name key of the new position
        acquisition : int, optional
            The index of the acquisition, by default 0
        """
        pos_grp = self._group.create_group(name, overwrite=self._overwrite)
        # build metadata
        image_meta = ImageMeta(acquisition=acquisition, path=pos_grp.basename)
        if not hasattr(self, "metadata"):
            self.metadata = WellGroupMeta(images=[image_meta])
        else:
            self.metadata.images.append(image_meta)
        self.dump_meta()
        return self[name]

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
    version : Literal["0.1", "0.4"]
        OME-NGFF specification version
    overwriting_creation : bool, optional
        Whether to overwrite or error upon creating an existing child item,
        by default False

    Attributes
    ----------
    version : Literal["0.1", "0.4"]
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
        version: Literal["0.1", "0.4"] = "0.4",
        overwriting_creation: bool = False,
    ):
        super().__init__(
            group=group,
            parse_meta=parse_meta,
            version=version,
            overwriting_creation=overwriting_creation,
        )

    @property
    def _member_names(self):
        return self.group_keys()

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

    def __init__(
        self,
        group: zarr.Group,
        parse_meta: bool = True,
        channel_names: List[str] = None,
        axes: list[AxisMeta] = None,
        name: str = None,
        acquisitions: List[AcquisitionMeta] = None,
        version: Literal["0.1", "0.4"] = "0.4",
        overwriting_creation: bool = False,
    ):
        super().__init__(
            group=group,
            parse_meta=parse_meta,
            version=version,
            overwriting_creation=overwriting_creation,
        )
        self._channel_names = channel_names
        self.axes = axes
        self._name = name
        self._acquisitions = (
            [AcquisitionMeta(id=0)] if not acquisitions else acquisitions
        )
        self._rows = {}
        self._cols = {}

    def _parse_meta(self):
        plate_meta = self.zattrs.get("plate")
        if plate_meta:
            self.metadata = PlateMeta(**plate_meta)

    @property
    def channel_names(self):
        return self._channel_names

    def dump_meta(self, field_count: bool = False):
        """Dumps metadata JSON to the `.zattrs` file.

        Parameters
        ----------
        field_count : bool, optional
            Whether to count all the FOV/positions
            and populate the metadata field 'plate.field_count';
            this operation can be expensive if the number of positions is large,
            by default False
        """
        if field_count:
            self.metadata.field_count = len([kv for kv in self.positions()])
        self.zattrs.update({"plate": self.metadata.dict(**TO_DICT_SETTINGS)})

    @staticmethod
    def _auto_idx(name: str, known: dict[str, int]):
        idx = known.get(name)
        if idx:
            return idx
        else:
            used = known.values()
            return max(used) + 1 if used else 0

    def _build_meta(
        self, first_row_meta: PlateAxisMeta, first_col_meta: PlateAxisMeta
    ):
        """Initiate metadata attribute if not present."""
        if not hasattr(self, "metadata"):
            self.metadata = PlateMeta(
                version=self.version,
                name=self._name,
                acquisitions=self._acquisitions,
                rows=[first_row_meta],
                columns=[first_col_meta],
            )

    def create_well(
        self,
        row_name: str,
        col_name: str,
        row_index: int = None,
        col_index: int = None,
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
        col_meta = PlateAxisMeta(name=col_grp.basename)
        # create new row
        if row_name not in self:
            row_grp = self.zgroup.create_group(
                row_name, overwrite=self._overwrite
            )
            row_meta = PlateAxisMeta(name=row_name)
            self._build_meta(row_meta, col_meta)
            self.metadata.rows.append(row_meta)
            self._rows[row_name] = row_index
        col_grp = row_grp.create_group(col_name, overwrite=self._overwrite)
        self._cols[col_name] = col_index
        well_index_meta = WellIndexMeta(
            path=os.path.join(row_grp.basename, col_grp.basename),
            row_index=row_index,
            column_index=col_index,
        )
        self.metadata.wells.append(well_index_meta)
        self.dump_meta()
        return self[row_name][col_name]

    def rows(self):
        """Returns a generator that iterate over the name and value
        of all the rows in the plate.

        Yields
        ------
        tuple[str, Row]
            Name and row object.
        """
        yield from self.iteritems()

    def positions(self):
        """Returns a generator that iterate over the path and value
        of all the positions (along rows, columns, and wells) in the plate.

        Yields
        ------
        [str, Position]
            Path and position object.
        """
        for _, row in self.rows():
            for _, well in row.wells():
                for _, position in well:
                    yield position.zgroup.path, position


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
        **kwargs,
    ):
        """Convenience method to open NGFF data stores.

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
            Channel names used to create a new data store,
            ignored for existing stores,
            by default None
        axes : list[AxisMeta], optional
            OME axes metadata, by default None:
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
        kwargs : dict, optional
            Keyword arguments to underlying NGFF node constructor,
            by default None


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
            **kwargs,
        )

    def __init__(self, *args, **kwargs):
        # this should call `Position.__init__()` for `OMEZarr`
        # or `Plate.__init__()` for `HCSZarr`
        super().__init__(*args, **kwargs)


class OMEZarr(Dataset, Position):
    """Single-FOV OME-Zarr dataset container.

    Parameters
    ----------
    root : zarr.Group
        The root group of the Zarr store,
    parse_meta : bool, optional
        Whether to parse metadata from file, by default True
    channel_names: List[str]
        Names of all the channels present in data
        ordered according to channel indices
    axes : List[AxisMeta], optional
        OME axes metadata, by default None:
        ```
        [AxisMeta(name='T', type='time', unit='second'),
        AxisMeta(name='C', type='channel', unit=None),
        AxisMeta(name='Z', type='space', unit='micrometer'),
        AxisMeta(name='Y', type='space', unit='micrometer'),
        AxisMeta(name='X', type='space', unit='micrometer')]
        ````
    version : Literal["0.1", "0.4"], optional
        OME-NGFF version, by default "0.4"
    overwriting_creation : bool
        Whether to overwrite or error upon creating an existing child item,
        by default False
    """

    def __init__(
        self,
        root: zarr.Group,
        parse_meta: bool = True,
        channel_names: list[str] = None,
        axes: list[AxisMeta] = None,
        version: Literal["0.1", "0.4"] = "0.4",
        overwriting_creation: bool = False,
    ):
        super().__init__(
            group=root,
            parse_meta=parse_meta,
            channel_names=channel_names,
            axes=axes,
            version=version,
            overwriting_creation=overwriting_creation,
        )


class HCSZarr(Dataset, Plate):
    """High-content screening OME-Zarr dataset container.


    Parameters
    ----------
    root : zarr.Group
        The root group of the Zarr store,
    parse_meta : bool, optional
        Whether to parse metadata from file, by default True
    channel_names: List[str]
        Names of all the channels present in data
        ordered according to channel indices
    axes : List[AxisMeta], optional
        OME axes metadata, default to:
        ```
        [AxisMeta(name='T', type='time', unit='second'),
        AxisMeta(name='C', type='channel', unit=None),
        AxisMeta(name='Z', type='space', unit='micrometer'),
        AxisMeta(name='Y', type='space', unit='micrometer'),
        AxisMeta(name='X', type='space', unit='micrometer')]
        ````
    version : Literal["0.1", "0.4"], optional
        OME-NGFF version, by default "0.4"
    overwriting_creation : bool
        Whether to overwrite or error upon creating an existing child item,
        by default False
    plate_name: str, optional
        Name of the plate, default to filename if not provided
    acquisitions: list[AcquisitionMeta], optional
        List of acquisitions, by default None
    """

    def __init__(
        self,
        root: zarr.Group,
        parse_meta: bool = True,
        channel_names: list[str] = None,
        axes: list[AxisMeta] = None,
        version: Literal["0.1", "0.4"] = "0.4",
        overwriting_creation: bool = False,
        plate_name: str = None,
        acquisitions: list[AcquisitionMeta] = None,
    ):
        super().__init__(
            group=root,
            parse_meta=parse_meta,
            channel_names=channel_names,
            axes=axes,
            name=plate_name,
            acquisitions=acquisitions,
            version=version,
            overwriting_creation=overwriting_creation,
        )
