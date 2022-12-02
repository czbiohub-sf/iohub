import os
from numcodecs import Blosc
import numpy as np


class WriterBase:
    """
    ABC for all writer types
    """

    def __init__(self, store, root_path):

        # init common attributes
        self.store = store
        self.root_path = root_path
        self.current_pos_group = None
        self.current_position = None
        self.current_well_group = None
        self.verbose = False
        self.dtype = None

        # set hardcoded compressor
        self.__compressor = Blosc(
            cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE
        )

        # maps to keep track of hierarchies
        self.rows = dict()
        self.columns = dict()
        self.positions = dict()

    # Silence print statements
    def set_verbosity(self, verbose: bool):
        self.verbose = verbose

    # Initialize zero array
    def init_array(
        self, data_shape, chunk_size, dtype, chan_names, clims, overwrite=False
    ):
        """

        Initializes the zarr array under the current position subgroup.
        array level is called 'arr_0' in the hierarchy.  Sets omero/multiscales metadata based upon
        chan_names and clims

        Parameters
        ----------
        data_shape:         (tuple)  Desired Shape of your data (T, C, Z, Y, X).  Must match data
        chunk_size:         (tuple) Desired Chunk Size (T, C, Z, Y, X).  Chunking each image would be (1, 1, 1, Y, X)
        dtype:              (str or np.dtype) Data Type, i.e. 'uint16' or np.uint16
        chan_names:         (list) List of strings corresponding to your channel names.  Used for OME-zarr metadata
        clims:              (list) list of tuples corresponding to contrast limtis for channel.  OME-Zarr metadata
                                    tuple can be of (start, end, min, max) or (start, end)
        overwrite:          (bool) Whether or not to overwrite the existing data that may be present.

        Returns
        -------

        """

        self.dtype = np.dtype(dtype)

        self.set_channel_attributes(chan_names, clims)
        self.current_pos_group.zeros(
            "arr_0",
            shape=data_shape,
            chunks=chunk_size,
            dtype=dtype,
            compressor=self.__compressor,
            overwrite=overwrite,
        )

    def write(self, data, t, c, z):
        """
        Write data to specified index of initialized zarr array

        :param data: (nd-array), data to be saved. Must be the shape that matches indices (T, C, Z, Y, X)
        :param t: (list), index or index slice of the time dimension
        :param c: (list), index or index slice of the channel dimension
        :param z: (list), index or index slice of the z dimension

        """

        shape = np.shape(data)

        if self.current_pos_group.__len__() == 0:
            raise ValueError("Array not initialized")

        if not isinstance(t, int) and not isinstance(t, slice):
            raise TypeError("t specification must be either int or slice")

        if not isinstance(c, int) and not isinstance(c, slice):
            raise TypeError("c specification must be either int or slice")

        if not isinstance(z, int) and not isinstance(z, slice):
            raise TypeError("z specification must be either int or slice")

        if isinstance(t, int) and isinstance(c, int) and isinstance(z, int):

            if len(shape) > 2:
                raise ValueError("Index dimensions exceed data dimensions")
            else:
                self.current_pos_group["arr_0"][t, c, z] = data

        else:
            self.current_pos_group["arr_0"][t, c, z] = data

    def create_channel_dict(self, chan_name, clim=None, first_chan=False):
        """
        This will create a dictionary used for OME-zarr metadata.  Allows custom contrast limits and channel
        names for display.  Defaults everything to grayscale.

        Parameters
        ----------
        chan_name:          (str) Desired name of the channel for display
        clim:               (tuple) contrast limits (start, end, min, max)
        first_chan:         (bool) whether or not this is the first channel of the dataset (display will be set to active)

        Returns
        -------
        dict_:              (dict) dictionary adherent to ome-zarr standards

        """

        if chan_name == "Retardance":
            min = clim[2] if clim else 0.0
            max = clim[3] if clim else 1000.0
            start = clim[0] if clim else 0.0
            end = clim[1] if clim else 100.0
        elif chan_name == "Orientation":
            min = clim[2] if clim else 0.0
            max = clim[3] if clim else np.pi
            start = clim[0] if clim else 0.0
            end = clim[1] if clim else np.pi

        elif chan_name == "Phase3D":
            min = clim[2] if clim else -10.0
            max = clim[3] if clim else 10.0
            start = clim[0] if clim else -0.2
            end = clim[1] if clim else 0.2

        elif chan_name == "BF":
            min = clim[2] if clim else 0.0
            max = clim[3] if clim else 65535.0
            start = clim[0] if clim else 0.0
            end = clim[1] if clim else 5.0

        elif chan_name == "S0":
            min = clim[2] if clim else 0.0
            max = clim[3] if clim else 65535.0
            start = clim[0] if clim else 0.0
            end = clim[1] if clim else 1.0

        elif chan_name == "S1":
            min = clim[2] if clim else 10.0
            max = clim[3] if clim else -10.0
            start = clim[0] if clim else -0.5
            end = clim[1] if clim else 0.5

        elif chan_name == "S2":
            min = clim[2] if clim else -10.0
            max = clim[3] if clim else 10.0
            start = clim[0] if clim else -0.5
            end = clim[1] if clim else 0.5

        elif chan_name == "S3":
            min = clim[2] if clim else -10
            max = clim[3] if clim else 10
            start = clim[0] if clim else -1.0
            end = clim[1] if clim else 1.0

        else:
            min = clim[2] if clim else 0.0
            max = clim[3] if clim else 65535.0
            start = clim[0] if clim else 0.0
            end = clim[1] if clim else 65535.0

        dict_ = {
            "active": first_chan,
            "coefficient": 1.0,
            "color": "FFFFFF",
            "family": "linear",
            "inverted": False,
            "label": chan_name,
            "window": {"end": end, "max": max, "min": min, "start": start},
        }

        return dict_

    def create_row(self, idx, name=None):
        """
        Creates a row in the hierarchy (first level below zarr store). Option to name
        this row.  Default is Row_{idx}.  Keeps track of the row name + row index for later
        metadata creation

        Parameters
        ----------
        idx:            (int) Index of the row (order in which it is placed)
        name:           (str) Optional name to replace default row name

        Returns
        -------

        """

        row_name = f"Row_{idx}" if not name else name
        row_path = os.path.join(self.root_path, row_name)

        # check if the user is trying to create a row that already exsits
        if os.path.exists(row_path):
            raise FileExistsError(
                f"A row subgroup with the name {row_name} already exists"
            )
        else:
            self.store.create_group(row_name)
            self.rows[idx] = row_name

    def create_column(self, row_idx, idx, name=None):
        """
        Creates a column in the hierarchy (second level below zarr store, one below row). Option to name
        this column.  Default is Col_{idx}.  Keeps track of the column name + column index for later
        metadata creation

        Parameters
        ----------
        row_idx:        (int) Index of the row to place the column underneath
        idx:            (int) Index of the column (order in which it is placed)
        name:           (str) Optional name to replace default column name

        Returns
        -------

        """

        col_name = f"Col_{idx}" if not name else name
        row_name = self.rows[row_idx]
        col_path = os.path.join(
            os.path.join(self.root_path, row_name), col_name
        )

        # check to see if the user is trying to create a row that already exists
        if os.path.exists(col_path):
            raise FileExistsError(
                f"A column subgroup with the name {col_name} already exists"
            )
        else:
            self.store[self.rows[row_idx]].create_group(col_name)
            self.columns[idx] = col_name

    def open_position(self, position: int):
        """
        Opens a position based upon the position index.  It will navigate the rows/column to
        find where this position is based off of the generation position map which keeps track
        of this information.  It will set current_pos_group to this position for writing the data

        Parameters
        ----------
        position:           (int) Index of the position you wish to open

        Returns
        -------

        """

        # get row, column, and path to the well
        row_name = self.positions[position]["row"]
        col_name = self.positions[position]["col"]
        well_path = os.path.join(
            os.path.join(self.root_path, row_name), col_name
        )

        # check to see if this well exists (row/column)
        if os.path.exists(well_path):
            pos_name = self.positions[position]["name"]
            pos_path = os.path.join(well_path, pos_name)

            # check to see if the position exists
            if os.path.exists(pos_path):

                if self.verbose:
                    print(f"Opening subgroup {row_name}/{col_name}/{pos_name}")

                # update trackers to note the current status of the writer
                self.current_pos_group = self.store[row_name][col_name][
                    pos_name
                ]
                self.current_well_group = self.store[row_name][col_name]
                self.current_position = position

            else:
                raise FileNotFoundError(
                    f"Could not find zarr position subgroup at {row_name}/{col_name}/{pos_name}\
                                                    Check spelling or create position subgroup with create_position"
                )
        else:
            raise FileNotFoundError(
                f"Could not find zarr position subgroup at {row_name}/{col_name}/\
                                                Check spelling or create column/position subgroup with create_position"
            )

    def set_root(self, root):
        """
        set the root path of the zarr store.  Used in the main writer class.

        Parameters
        ----------
        root:               (str) path to the zarr store (folder ending in .zarr)

        Returns
        -------

        """
        self.root_path = root

    def set_store(self, store):
        """
        Sets the zarr store.  Used in the main writer class

        Parameters
        ----------
        store:              (Zarr StoreObject) Opened zarr store at the highest level

        Returns
        -------

        """
        self.store = store

    def get_zarr(self):
        return self.current_pos_group

    def set_channel_attributes(self, chan_names, clims=None):
        """
        A method for creating ome-zarr metadata dictionary.
        Channel names are defined by the user, everything else
        is pre-defined.

        Parameters
        ----------
        chan_names:     (list) List of channel names in the order of the channel dimensions
                                i.e. if 3D Phase is C = 0, list '3DPhase' first.

        clims:          (list of tuples) contrast limits to display for every channel

        """

        rdefs = {
            "defaultT": 0,
            "model": "color",
            "projection": "normal",
            "defaultZ": 0,
        }

        multiscale_dict = [{"datasets": [{"path": "arr_0"}], "version": "0.1"}]
        dict_list = []

        if clims and len(chan_names) < len(clims):
            raise ValueError(
                "Contrast Limits specified exceed the number of channels given"
            )

        for i in range(len(chan_names)):
            if clims:
                if len(clims[i]) == 2:
                    if "float" in self.dtype.name:
                        clim = (
                            float(clims[i][0]),
                            float(clims[i][1]),
                            -1000,
                            1000,
                        )
                    else:
                        info = np.iinfo(self.dtype)
                        clim = (
                            float(clims[i][0]),
                            float(clims[i][1]),
                            info.min,
                            info.max,
                        )
                elif len(clims[i]) == 4:
                    clim = (
                        float(clims[i][0]),
                        float(clims[i][1]),
                        float(clims[i][2]),
                        float(clims[i][3]),
                    )
                else:
                    raise ValueError(
                        "clim specification must a tuple of length 2 or 4"
                    )

            first_chan = True if i == 0 else False
            if not clims or i >= len(clims):
                dict_list.append(
                    self.create_channel_dict(
                        chan_names[i], first_chan=first_chan
                    )
                )
            else:
                dict_list.append(
                    self.create_channel_dict(
                        chan_names[i], clim, first_chan=first_chan
                    )
                )

        full_dict = {
            "multiscales": multiscale_dict,
            "omero": {"channels": dict_list, "rdefs": rdefs, "version": 0.1},
        }

        self.current_pos_group.attrs.put(full_dict)

    def init_hierarchy(self):
        pass

    def create_position(self, position: int, name: str):
        pass


class DefaultZarr(WriterBase):
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


class HCSZarr(WriterBase):
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
