import os, logging
from numcodecs import Blosc
import numpy as np
import zarr
from ome_zarr.reader import Reader
from ome_zarr.format import format_from_version

from iohub.ngff_meta import *

from typing import TYPE_CHECKING, Union, Tuple, List, Dict, Literal
from numpy.typing import NDArray, DTypeLike

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath


def _channel_display_settings(
    chan_name: str,
    clim: Tuple[float, float, float, float] = None,
    first_chan: bool = False,
):
    """This will create a dictionary used for OME-zarr metadata.  Allows custom contrast limits and channel
    names for display. Defaults everything to grayscale.

    Parameters
    ----------
    chan_name : str
        Desired name of the channel for display
    clim : Tuple[float, float, float, float], optional
        Contrast limits (start, end, min, max)
    first_chan : bool, optional
        Whether or not this is the first channel of the dataset (display will be set to active), by default False

    Returns
    -------
    dict
        Display settings adherent to ome-zarr standards
    """
    U16_FMAX = float(np.iinfo(np.uint16).max)
    channel_settings = {
        "Retardance": (0.0, 100.0, 0.0, 1000.0),
        "Orientation": (0.0, np.pi, 0.0, np.pi),
        "Phase3D": (-0.2, 0.2, -10, 10),
        "BF": (0.0, 5.0, 0.0, U16_FMAX),
        "S0": (0.0, 1.0, 0.0, U16_FMAX),
        "S1": (-0.5, 0.5, -10.0, 10.0),
        "S2": (-0.5, 0.5, -10.0, 10.0),
        "S3": (-1.0, 1.0, -10.0, -10.0),
        "Other": (0, U16_FMAX, 0.0, U16_FMAX),
    }
    if not clim:
        if chan_name in channel_settings.keys():
            clim = channel_settings[chan_name]
        else:
            clim = channel_settings["Other"]
    window = WindowDict(start=clim[0], end=clim[1], min=clim[2], max=clim[3])
    return ChannelMeta(
        active=first_chan,
        coefficient=1.0,
        color="FFFFFF",
        family="linear",
        inverted=False,
        label=chan_name,
        window=window,
    )


class OMEZarrWriter:
    """Generic OME-Zarr writer instance for an existing Zarr store.

    Parameters
    ----------
    root : zarr.Group
        The root group of the Zarr store, dimension separator should be '/'
    channel_names: List[str]
        Names of all the channels present in data ordered according to channel indices
    version : Literal["0.1", "0.4"], optional
        OME-NGFF version, by default 0.4
    arr_name : str, optional
        Base name of the arrays, by default '0'
    axes : Union[str, List[str], Dict[str, str]], optional
        OME axes metadata, by default:
        `
        [{'name': 'T', 'type': 'time', 'unit': 'second'},
        {'name': 'C', 'type': 'channel'},
        {'name': 'Z', 'type': 'space', 'unit': 'micrometer'},
        {'name': 'Y', 'type': 'space', 'unit': 'micrometer'},
        {'name': 'X', 'type': 'space', 'unit': 'micrometer'}]
        `
    """

    _DEFAULT_AXES = [
        AxisMeta(name="T", type="time", unit="second"),
        AxisMeta(name="C", type="channel"),
        *[
            AxisMeta(name=i, type="space", unit="micrometer")
            for i in ("Z", "Y", "X")
        ],
    ]

    @classmethod
    def from_ome_reader(cls, reader: Reader):
        # TODO: get metadata from reader
        writer = cls()
        return writer

    def __init__(
        self,
        root: zarr.Group,
        channel_names: List[str],
        version: Literal["0.1", "0.4"] = "0.4",
        arr_name: str = "0",
        axes: Union[str, List[str], List[Dict[str, str]]] = None,
    ):
        self.root = root
        self.channel_names = channel_names
        self.version = version
        self.fmt = format_from_version(str(version))
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.arr_name = arr_name
        self.axes = axes if axes else self._DEFAULT_AXES
        self._overwrite = False

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

    def require_position(
        self,
        name: str,
        overwrite: bool = False,
    ):
        """Creates a new position group if it does not exist.

        Parameters
        ----------
        name : str
            Name (absolute path under the store root) of position group
        overwrite : bool, optional
            Delete all existing data in the position group, by default False


        Returns
        -------
        Group
            Zarr group for the required position
        """
        if name not in self.positions:
            self.positions[name] = {"id": len(self.positions)}
        return self.root.require_group(name, overwrite=overwrite)

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
        """Write a z-stack with OME-NGFF metadata.

        Parameters
        ----------
        data : NDArray
            Image data with shape (Z, Y, X)
        position : zarr.Group
            Parent position group
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
            position, name, zyx_shape=zyx_shape, dtype=data.dtype
        )
        if time_index >= array.shape[0] or channel_index >= array.shape[1]:
            t = max(time_index, array.shape[0])
            c = max(channel_index, array.shape[1])
            array.resize(t, c, *zyx_shape)
        # write data
        array[time_index, channel_index] = data
        if auto_meta:
            self._dump_zstack_meta(position, name, transform, additional_meta)

    def _dump_zstack_meta(
        self,
        position: zarr.Group,
        name: str,
        transform: List[TransformationMeta],
        additional_meta: dict,
    ):
        dataset_meta = self._dataset_meta(name, transform=transform)
        pos_meta = self.positions[position.name]
        if "attrs" not in pos_meta:
            multiscales = MultiScalesMeta(
                version=self.version,
                axes=self.axes,
                datasets=[dataset_meta],
                metadata=additional_meta,
            )
            omero = self._omero_meta(position)
            images_meta = ImagesMeta(multiscales=multiscales, omero=omero)
            pos_meta["attrs"] = images_meta
        else:
            if (
                dataset_meta.path
                not in pos_meta["attrs"].multiscales.get_dataset_paths()
            ):
                pos_meta["attrs"].multiscales.datasets.append(dataset_meta)
        position.attrs.put(pos_meta["attrs"].dict(**TO_DICT_SETTINGS))

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
        position: zarr.Group,
        clims: List[Tuple[float, float, float, float]] = None,
    ):
        id = self.positions[position.name]["id"]
        if not clims:
            clims = [None] * 4
        channels = []
        for i, (channel_name, clim) in enumerate(
            zip(self.channel_names, clims)
        ):
            if i == 0:
                first_chan = True
            channels.append(
                _channel_display_settings(
                    channel_name, clim=clim, first_chan=first_chan
                )
            )
        omero_meta = OMEROMeta(
            version=self.version,
            id=id,
            name=position.name,
            channels=channels,
            rdefs=RDefsMeta(default_t=0, default_z=0),
        )
        return omero_meta


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
    axes : Union[str, List[str], Dict[str, str]], optional
        OME axes metadata, by default:
        `
        [{'name': 'T', 'type': 'time', 'unit': 'second'},
        {'name': 'C', 'type': 'channel'},
        {'name': 'Z', 'type': 'space', 'unit': 'micrometer'},
        {'name': 'Y', 'type': 'space', 'unit': 'micrometer'},
        {'name': 'X', 'type': 'space', 'unit': 'micrometer'}]
        `
    """

    def __init__(
        self,
        root: zarr.Group,
        channel_names: List[str],
        plate_name: str = None,
        version: Literal["0.1", "0.4"] = "0.4",
        arr_name: str = "0",
        axes: Union[str, List[str], List[Dict[str, str]]] = None,
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
        if not index:
            index = len(self.rows)
        if name in self.rows:
            _id = self.rows[name]["id"]
            if _id != index:
                raise ValueError(
                    f"Requested index {index} conflicts with existing row {name} index {_id}."
                )
        else:
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
        if not col_index:
            col_index = len(self.columns)
        if col_name in self.columns:
            _id = self.columns[col_name]["id"]
            if _id != col_index:
                raise ValueError(
                    f"Requested index {col_index} conflicts with existing column {col_name} index {_id}."
                )
        else:
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
        position = super().require_position(
            os.path.join(well.name, fov),
            overwrite=bool(kwargs.get("overwrite")),
        )
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


class DefaultZarr(HCSWriter):
    """
    This writer is based off creating a default HCS hierarchy for non-hcs datasets.
    Currently, we decide that all positions will live under individual columns under
    a single row.  i.e. this produces the following structure:

    Dataset.zarr
        ____> Row_0
            ---> Col_0
                ---> Pos_000
            ...

            --> Col_N
                ---> Pos_N

    We assume this structure in the metadata updating/position creation
    """

    def __init__(self, store, root_path):

        super().__init__(store, root_path)

        self.dataset_name = None
        self.plate_meta = dict()
        self.well_meta = dict()

    def init_hierarchy(self):
        """
        method to init the default hierarchy.
        Will create the first row and initialize metadata fields

        Returns
        -------

        """
        self.create_row(0)
        self.dataset_name = os.path.basename(self.root_path).strip(".zarr")

        self.plate_meta["plate"] = {
            "acquisitions": [
                {
                    "id": 1,
                    "maximumfieldcount": 1,
                    "name": "Dataset",
                    "starttime": 0,
                }
            ],
            "columns": [],
            "field_count": 1,
            "name": self.dataset_name,
            "rows": [],
            "version": "0.1",
            "wells": [],
        }

        self.plate_meta["plate"]["rows"].append({"name": self.rows[0]})

        self.well_meta["well"] = {"images": [], "version": "0.1"}
        self.well_meta = dict(self.well_meta)

    def create_position(self, position, name):
        """
        Creates a column and position subgroup given the index and name.  Name is
        provided by the main writer class

        Parameters
        ----------
        position:           (int) Index of the position to create
        name:               (str) name of the position subgroup

        Returns
        -------

        """

        # get row name and create a column
        row_name = self.rows[0]
        self.create_column(0, position)
        col_name = self.columns[position]

        if self.verbose:
            print(
                f"Creating and opening subgroup {row_name}/{col_name}/{name}"
            )

        # create position subgroup
        self.store[row_name][col_name].create_group(name)

        # update trackers
        self.current_pos_group = self.store[row_name][col_name][name]
        self.current_well_group = self.store[row_name][col_name]
        self.current_position = position

        # update ome-metadata
        self.positions[position] = {
            "name": name,
            "row": row_name,
            "col": col_name,
        }
        self._update_plate_meta(position)
        self._update_well_meta(position)

    def _update_plate_meta(self, pos):
        """
        Updates the plate metadata which lives at the highest level (top store).
        This metadata carries information on the rows/columns and their paths.

        Parameters
        ----------
        pos:            (int) Position index to update the metadata

        Returns
        -------

        """

        self.plate_meta["plate"]["columns"].append({"name": self.columns[pos]})
        self.plate_meta["plate"]["wells"].append(
            {"path": f"{self.rows[0]}/{self.columns[pos]}"}
        )
        self.store.attrs.put(self.plate_meta)

    def _update_well_meta(self, pos):
        """
        Updates the well metadata which lives at the column level.
        This metadata carries information about the positions underneath.
        Assumes only one position will ever be underneath this level.

        Parameters
        ----------
        pos:            (int) Index of the position to update

        Returns
        -------

        """

        self.well_meta["well"]["images"] = [
            {"path": self.positions[pos]["name"]}
        ]
        self.store[self.rows[0]][self.columns[pos]].attrs.put(self.well_meta)


class HCSZarr(HCSWriter):
    """
    This writer version will write new data based upon provided HCS metadata.
    Useful for when we are reconstructing data that already has a specific HCS heirarchy.
    allows for the process data structure to match the raw data structure for consistent
    user browsing.  Also works for copying the structure created by the DefaultZarr writer.

    Note, this doesn't make any new structural changes but strictly copies the one provided
    """

    def __init__(self, store, root_path, hcs_meta):

        super().__init__(store, root_path)
        self.hcs_meta = hcs_meta

    def init_hierarchy(self):
        """
        Creates the entire Row/Col/Pos hierarchy structure based upon the accepted
        HCS metadata. Creates everything row by row

        Returns
        -------

        """

        # check to make sure HCS metadata can be parsed / isn't missing critical info
        self._check_HCS_meta()

        plate_meta = self.hcs_meta["plate"]
        row_count = 0
        col_count = 0
        well_count = 0
        pos_count = 0

        # go through rows
        for row in plate_meta["rows"]:

            self.create_row(row_count, row["name"])

            # go through columns under this row
            for col in plate_meta["columns"]:

                self.create_column(row_count, col_count, col["name"])

                well_meta = self.hcs_meta["well"][well_count]

                # go through positions under this column
                for image in well_meta["images"]:
                    self.positions[pos_count] = {
                        "name": image["path"],
                        "row": row["name"],
                        "col": col["name"],
                    }
                    self.store[row["name"]][col["name"]].create_group(
                        image["path"]
                    )
                    pos_count += 1

                # place well_metadata provided from user
                self.store[row["name"]][col["name"]].attrs.put(
                    {"well": well_meta}
                )

                col_count += 1
                well_count += 1

            col_count = 0

        # place plate metadata as provided by the user
        self.store.attrs.put({"plate": plate_meta})

    def _check_HCS_meta(self):
        """
        checks to make sure the HCS metadata is formatted properly. HCS metadata should be passed in the following
        structure:

        {'plate':       {<plate_metadata>},
        'well':         [{well_meta}] # length must be equal to number of wells
        }

        Only the 'plate' metadata is required here.  All others are optional and will be used if present.
        Specifications on the specific metadata structures can be found here https://ngff.openmicroscopy.org/0.1/

        Returns
        -------

        """

        HCS_Defaults = {
            "acquisitions": [
                {
                    "id": 1,
                    "maxmimumfieldcount": 1,
                    "name": "Dataset",
                    "starttime": 0,
                }
            ],
            "field_count": 1,
            "name": os.path.basename(self.root_path).strip(".zarr"),
            "version": "0.1",
        }

        # Check to see if all of the required keys are present
        if "plate" not in self.hcs_meta.keys():
            raise KeyError(f"HCS metadata missing plate metadata")

        if "well" not in self.hcs_meta.keys():
            raise KeyError(f"HCS metadata missing well metadata")

        plate_keys = self.hcs_meta["plate"].keys()
        for key in plate_keys:
            if key in HCS_Defaults and key not in plate_keys:
                self.hcs_meta["plate"][key] = HCS_Defaults[key]

        if "rows" not in plate_keys:
            raise KeyError("rows key is missing from plate metadata")

        if "columns" not in plate_keys:
            raise KeyError("columns key is missing from plate metadata")

        # create wells data based on rows, columns if not present
        if "wells" not in self.hcs_meta["plate"].keys():
            self.hcs_meta["plate"]["wells"] = [
                {"path": f"{row}/{col}"}
                for row in self.hcs_meta["plate"]["rows"]
                for col in self.hcs_meta["plate"]["columns"]
            ]

        # Check Plate Meta
        n_col = len(self.hcs_meta["plate"]["columns"])
        n_row = len(self.hcs_meta["plate"]["rows"])
        n_well = len(self.hcs_meta["plate"]["wells"])
        if n_col * n_row != n_well:
            raise ValueError(
                "Plate metadata error: Numbers of rows/columns does not match number of wells"
            )

        cnt = 0
        for row in self.hcs_meta["plate"]["rows"]:
            for col in self.hcs_meta["plate"]["columns"]:
                row_name = row["name"]
                col_name = col["name"]
                well_path = self.hcs_meta["plate"]["wells"][cnt]["path"]
                if well_path != f"{row_name}/{col_name}":
                    raise ValueError(
                        f"Plate metadata error: well path {well_path} \
                                        does not match row {row_name}, col {col_name}"
                    )
                cnt += 1

        # check to make sure well metadata is correct
        if "well" in self.hcs_meta.keys():
            for well in self.hcs_meta["well"]:
                if len(well["images"]) > self.hcs_meta["plate"]["field_count"]:
                    raise ValueError(
                        "Well metadata error: number of FOV exceeds maximum field count in plate metadata"
                    )

    def create_position(self, position, name=None):
        """
        Assumes all of the positions were created upon _init_hierarchy called before this function.
        Will raise an error if the user tries to create a position with this writer version

        Parameters
        ----------
        position:           (int) position index to create
        name:               (str) or None.  Not used.

        Returns
        -------

        """
        try:
            self.open_position(position)
        except:
            raise ValueError(
                "HCS Writer already initialized positions. Cannot create a position that already exists"
            )
