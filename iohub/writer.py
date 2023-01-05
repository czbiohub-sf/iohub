# TODO: remove this in the future (PEP deferred for 3.11, now 3.12?)
from __future__ import annotations

import os, logging
from numcodecs import Blosc
import zarr
from ome_zarr.format import format_from_version

from iohub.zarrfile import OMEZarrReader, HCSReader, _DEFAULT_AXES
from iohub.ngff_meta import *
from iohub.lf_utils import channel_display_settings

from typing import TYPE_CHECKING, Union, Tuple, List, Dict, Literal
from numpy.typing import NDArray, DTypeLike

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath


def new_zarr(
    store_path: StrOrBytesPath, mode: Literal["r", "r+", "a", "w", "w-"] = "a"
):
    """Open the root group of a new Zarr store at a give path.
    Create a OME-NGFF-compatible store on the local file system if not present.

    Parameters
    ----------
    store_path : StrOrBytesPath
        Path of the new store
    mode : Literal["r", "r+", "a", "w", "w-"], optional
        File mode (passed to `zarr.open()`) used to open the root group,
        by default "a" (read/write, create if not present)

    Returns
    -------
    Group
        Root Zarr group of a directory store with `/` as the dimension separator
    """
    if os.path.exists(store_path):
        raise FileExistsError(f"{store_path} already exists.")
    store = zarr.DirectoryStore(str(store_path), dimension_separator="/")
    return zarr.open(store, mode=mode)


class OMEZarrWriter:
    """Generic OME-Zarr writer instance for an existing Zarr store.

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
        `
        [{'name': 'T', 'type': 'time', 'unit': 'second'},
        {'name': 'C', 'type': 'channel'},
        {'name': 'Z', 'type': 'space', 'unit': 'micrometer'},
        {'name': 'Y', 'type': 'space', 'unit': 'micrometer'},
        {'name': 'X', 'type': 'space', 'unit': 'micrometer'}]
        `
    """

    _READER_TYPE = OMEZarrReader

    @classmethod
    def from_reader(cls, reader: OMEZarrReader):
        reader.store.close()
        root = zarr.open(
            zarr.DirectoryStore(
                str(reader.store.path), dimension_separator="/"
            ),
            mode="a",
        )
        writer = cls(
            root,
            reader.channel_names,
            version=reader.version,
            arr_name=reader.array_keys[0],
        )
        writer.images_meta = ImagesMeta(
            multiscales=root.attrs["multiscales"], omero=root.attrs["omero"]
        )
        writer.axes = reader.axes
        return writer

    @classmethod
    def open(
        cls,
        store_path: StrOrBytesPath,
        mode: Literal["r+", "a", "w-"] = "a",
        channel_names: List[str] = None,
        version: Literal["0.1", "0.4"] = "0.4",
    ):
        """Convenience method to open Zarr stores.
        Uses default parameters to initiate readers/writers. Initiate manually to alter them.

        Parameters
        ----------
        store_path : StrOrBytesPath
            Path to the Zarr store to open
        mode : Literal["r+", "a", "w-"], optional
            Persistence mode: 'r+' means read/write (must exist); 'a' means read/write (create if doesn't exist); 'w-' means create (fail if exists), by default 'a'
        channel_names : List[str], optional
            Channel names (must be provided to create a new data store), by default None
        version : Literal["0.1", "0.4"], optional
            OME-NGFF version, by default "0.4"

        Returns
        -------
        OMEZarrWriter
            writer instance
        """
        try:
            reader = cls._READER_TYPE(store_path, version=version)
            logging.info(f"Found existing OME-Zarr dataset at {store_path}")
            if mode in {"r+", "a"}:
                return cls.from_reader(reader)
            elif mode == "w-":
                raise FileExistsError(
                    f"Persistence mode 'w-' does not allow overwriting."
                )
        except:
            not_found_msg = f"OME-Zarr dataset not found at {store_path}"
            if mode == "r+":
                raise FileNotFoundError(not_found_msg)
            elif mode in {"a", "w-"}:
                logging.info(not_found_msg)
                if not channel_names:
                    raise ValueError(
                        "Cannot initiate writer without channel names."
                    )
                root = new_zarr(store_path)
                logging.info(f"Creating new data store at {store_path}")
                return cls(root, channel_names, version=version)

    def __init__(
        self,
        root: zarr.Group,
        channel_names: List[str],
        version: Literal["0.1", "0.4"] = None,
        arr_name: str = "0",
        axes: List[AxisMeta] = None,
    ):
        self.root = root
        self._channel_names = channel_names
        self.version = version
        self.fmt = format_from_version(str(version)) if version else "0.4"
        self.arr_name = arr_name
        self.axes = axes if axes else _DEFAULT_AXES
        self._overwrite = False

    @property
    def channel_names(self):
        return self._channel_names

    @property
    def _storage_options(self):
        return {
            "compressor": Blosc(
                cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE
            ),
            "overwrite": self._overwrite,
        }

    def _rel_keys(self, rel_path: str) -> List[str]:
        return [name for name in self.root[rel_path].group_keys()]

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
            Chunk size for the new array if not present, by default a z-stack (Z, Y, X)

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
                chunks = zyx_shape
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

    def _dataset_meta(
        self,
        name: str,
        transform: List[TransformationMeta] = None,
    ):
        if not transform:
            transform = [TransformationMeta(type="identity")]
        dataset_meta = DatasetMeta(
            path=name, coordinate_transformations=transform
        )
        return dataset_meta

    def _omero_meta(
        self,
        id: int,
        group: zarr.Group,
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
            name=group.name,
            channels=channels,
            rdefs=RDefsMeta(default_t=0, default_z=0),
        )
        return omero_meta

    def append_channel(self, chan_name: str):
        """Append a channel to the end of the channel list.

        Parameters
        ----------
        chan_name : str
            Name of the new channel
        """
        if chan_name in self._channel_names:
            raise ValueError(
                f"Channel name {chan_name} already exists in the dataset."
            )
        self._channel_names.append(chan_name)
        if "omero" in self.root.attrs:
            self.images_meta.omero.channels.append(
                channel_display_settings(chan_name)
            )
            self.root.attrs["omero"] = self.images_meta.omero.dict(
                **TO_DICT_SETTINGS
            )

    def close(self):
        self.root.store.close()


class HCSWriter(OMEZarrWriter):
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

    _READER_TYPE = HCSReader

    @classmethod
    def from_reader(
        cls,
        reader: HCSReader,
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
        acq_id : str, optional
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
        position: zarr.Group,
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
            position,
            time_index,
            channel_index,
            name=name,
            auto_meta=False,
        )
        if auto_meta:
            self._dump_zstack_meta(position, name, transform, additional_meta)
        well = self.root.get(os.path.dirname(position.name))
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
