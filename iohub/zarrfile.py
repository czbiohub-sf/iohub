# TODO: remove this in the future (PEP deferred for 3.11, now 3.12?)
from __future__ import annotations

import os, logging
import zarr
import numpy as np
from copy import copy
from ome_zarr.io import parse_url
from ome_zarr.format import format_from_version

from iohub.reader_base import ReaderBase
from iohub.ngff_meta import *

from typing import TYPE_CHECKING, Union, Tuple, List, Dict, Literal

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath


_DEFAULT_AXES = [
    AxisMeta(name="T", type="time", unit="second"),
    AxisMeta(name="C", type="channel"),
    *[
        AxisMeta(name=i, type="space", unit="micrometer")
        for i in ("Z", "Y", "X")
    ],
]


class OMEZarrReader(ReaderBase):
    def __init__(
        self,
        store_path: StrOrBytesPath,
        version: Literal["0.1", "0.4"] = "0.4",
    ):
        super().__init__()
        # check zarr store
        if not version == "0.4":
            logging.warn(
                "\n".join(
                    "The OMEZarrReader is only tested against OME-NGFF v0.4.",
                    f"Requested version {version} may not work properly.",
                )
            )
        location = parse_url(
            store_path, mode="r", fmt=format_from_version(version)
        )
        if not location:
            raise FileNotFoundError(
                f"OME-Zarr store not found at {store_path}."
            )
        if not location.exists():
            raise FileNotFoundError(
                "Array and group metadata not found. Is it an empty store?"
            )
        self.version = version
        self.store = location.store
        self.root = zarr.group(self.store)
        array_keys = list(self.root.array_keys())
        if array_keys:
            self.array_keys = array_keys
        else:
            raise FileNotFoundError(
                "Array not found at top level. Is this an HCS store?"
            )
        try:
            channels: list = self.root.attrs.get("omero").get("channels")
            self.channel_names = [c["label"] for c in channels]
        except KeyError:
            logging.warn(
                "OMERO channel metadata not found. Channel names cannot be determined."
            )
        try:
            self.axes = [
                AxisMeta(**ax)
                for ax in self.root.attrs["multiscales"][0]["axes"]
            ]
        except KeyError:
            logging.warn("Axes meta data not found.")


class ZarrReader(ReaderBase):

    """
    .. deprecated:: 0.0.1
          `ZarrReader` will be removed in future iohub releases, it is replaced by
          `HCSReader` to enforce upgrade to version 0.4 of the OME-Zarr specification.

    Reader for HCS ome-zarr arrays.  OME-zarr structure can be found here: https://ngff.openmicroscopy.org/0.1/
    Also collects the HCS metadata so it can be later copied.
    """

    def __init__(self, zarrfile: str):
        super().__init__()

        # zarr files (.zarr) are directories
        if not os.path.isdir(zarrfile):
            raise ValueError("file does not exist")

        self.zf = zarrfile

        try:
            self.store = zarr.open(self.zf, "r")
        except:
            raise FileNotFoundError("Supplies path is not a valid zarr store")

        self.root = self.store  # TODO: fix `self.store`
        try:
            row = self.store[list(self.store.group_keys())[0]]
            col = row[list(row.group_keys())[0]]
            pos = col[list(col.group_keys())[0]]
            self.arr_name = list(pos.array_keys())[0]
        except IndexError:
            raise IndexError("Incompatible zarr format")

        self.plate_meta = self.store.attrs.get("plate")
        self._get_rows()
        self._get_columns()
        self._get_wells()
        self.position_map = dict()
        self._get_positions()

        # structure of zarr array
        first_arr_shape = self.root[self.position_map[0]["well"]][
            self.position_map[0]["name"]
        ][self.arr_name].shape
        (
            self.frames,
            self.channels,
            self.slices,
            self.height,
            self.width,
        ) = np.pad(
            first_arr_shape,
            (5 - len(first_arr_shape), 0),
            "constant",
            constant_values=(1),
        )
        self.positions = len(self.position_map)
        self.channel_names = []
        self.stage_positions = 0
        self.z_step_size = None

        # initialize metadata
        self.mm_meta = None

        try:
            self._set_mm_meta()
        except TypeError:
            self.mm_meta = None

        self._generate_hcs_meta()

        # get channel names from omero metadata if no MM meta present
        if len(self.channel_names) == 0:
            self._get_channel_names()

    def _get_rows(self):
        """
        Function to get the rows of the zarr hierarchy from HCS metadata

        Returns
        -------

        """
        rows = []
        for row in self.plate_meta["rows"]:
            rows.append(row["name"])
        self.rows = rows

    def _get_columns(self):
        """
        Function to get the columns of the zarr hierarchy from HCS metadata

        Returns
        -------

        """
        columns = []
        for column in self.plate_meta["columns"]:
            columns.append(column["name"])
        self.columns = columns

    def _get_wells(self):
        """
        Function to get the wells (Row/Col) of the zarr hierarchy from HCS metadata

        Returns
        -------

        """
        wells = []
        for well in self.plate_meta["wells"]:
            wells.append(well["path"])
        self.wells = wells

    def _get_positions(self):
        """
        Gets the position names and paths from HCS metadata

        Returns
        -------

        """

        idx = 0
        # Assumes that the positions are indexed in the order of Row-->Well-->FOV
        for well in self.wells:
            for pos in self.store[well].attrs.get("well").get("images"):
                name = pos["path"]
                self.position_map[idx] = {"name": name, "well": well}
                idx += 1

    def _generate_hcs_meta(self):
        """
        Pulls the HCS metadata and organizes it into a dictionary structure
        that can be easily read by the WaveorderWriter.

        Returns
        -------

        """
        self.hcs_meta = dict()
        self.hcs_meta["plate"] = self.plate_meta

        well_metas = []
        for well in self.wells:
            meta = self.store[well].attrs.get("well")
            well_metas.append(meta)

        self.hcs_meta["well"] = well_metas

    def _set_mm_meta(self):
        """
        Sets the micromanager summary metadata based on MM version

        Returns
        -------

        """
        self.mm_meta = self.store.attrs.get("Summary")
        mm_version = self.mm_meta["MicroManagerVersion"]

        if mm_version != "pycromanager":
            if "beta" in mm_version:
                if self.mm_meta["Positions"] > 1:
                    self.stage_positions = []

                    for p in range(len(self.mm_meta["StagePositions"])):
                        pos = self._simplify_stage_position_beta(
                            self.mm_meta["StagePositions"][p]
                        )
                        self.stage_positions.append(pos)

            # elif mm_version == '1.4.22':
            #     for ch in self.mm_meta['ChNames']:
            #         self.channel_names.append(ch)
            else:
                if self.mm_meta["Positions"] > 1:
                    self.stage_positions = []

                    for p in range(self.mm_meta["Positions"]):
                        pos = self._simplify_stage_position(
                            self.mm_meta["StagePositions"][p]
                        )
                        self.stage_positions.append(pos)

                # for ch in self.mm_meta['ChNames']:
                #     self.channel_names.append(ch)

        self.z_step_size = self.mm_meta["z-step_um"]

    def _get_channel_names(self):

        well = self.hcs_meta["plate"]["wells"][0]["path"]
        pos = self.hcs_meta["well"][0]["images"][0]["path"]

        omero_meta = self.store[well][pos].attrs.asdict()["omero"]

        for chan in omero_meta["channels"]:
            self.channel_names.append(chan["label"])

    def _simplify_stage_position(self, stage_pos: dict):
        """
        flattens the nested dictionary structure of stage_pos and removes superfluous keys

        Parameters
        ----------
        stage_pos:      (dict) dictionary containing a single position's device info

        Returns
        -------
        out:            (dict) flattened dictionary
        """

        out = copy(stage_pos)
        out.pop("DevicePositions")
        for dev_pos in stage_pos["DevicePositions"]:
            out.update({dev_pos["Device"]: dev_pos["Position_um"]})
        return out

    def _simplify_stage_position_beta(self, stage_pos: dict):
        """
        flattens the nested dictionary structure of stage_pos and removes superfluous keys
        for MM2.0 Beta versions

        Parameters
        ----------
        stage_pos:      (dict) dictionary containing a single position's device info

        Returns
        -------
        new_dict:       (dict) flattened dictionary

        """

        new_dict = {}
        new_dict["Label"] = stage_pos["label"]
        new_dict["GridRow"] = stage_pos["gridRow"]
        new_dict["GridCol"] = stage_pos["gridCol"]

        for sub in stage_pos["subpositions"]:
            values = []
            for field in ["x", "y", "z"]:
                if sub[field] != 0:
                    values.append(sub[field])
            if len(values) == 1:
                new_dict[sub["stageName"]] = values[0]
            else:
                new_dict[sub["stageName"]] = values

        return new_dict

    def get_image_plane_metadata(self, p, c, z):
        """
        For the sake of not keeping an enormous amount of metadta, only the microscope conditions
        for the first timepoint are kept in the zarr metadata during write.  User can only query image
         plane metadata at p, c, z

        Parameters
        ----------
        p:          (int) Position index
        c:          (int) Channel index
        z:          (int) Z-slice index

        Returns
        -------
        (dict) Image Plane Metadata at given coordinate w/ T = 0

        """
        coord_str = f"({p}, 0, {c}, {z})"
        return self.store.attrs.get("ImagePlaneMetadata").get(coord_str)

    def get_zarr(self, position):
        """
        Returns the position-level zarr group array (not in memory)

        Parameters
        ----------
        position:       (int) position index

        Returns
        -------
        (ZarrArray) Zarr array containing the (T, C, Z, Y, X) array at given position

        """
        pos_info = self.position_map[position]
        well = pos_info["well"]
        pos = pos_info["name"]
        return self.store[well][pos][self.arr_name]

    def get_array(self, position):
        """
        Gets the (T, C, Z, Y, X) array at given position

        Parameters
        ----------
        position:       (int) position index

        Returns
        -------
        (nd-Array) numpy array of size (T, C, Z, Y, X) at specified position

        """
        pos = self.get_zarr(position)
        return pos[:]

    def get_image(self, p, t, c, z):
        """
        Returns the image at dimension P, T, C, Z

        Parameters
        ----------
        p:          (int) index of the position dimension
        t:          (int) index of the time dimension
        c:          (int) index of the channel dimension
        z:          (int) index of the z dimension

        Returns
        -------
        image:      (nd-array) image at the given dimension of shape (Y, X)
        """

        pos = self.get_zarr(p)

        return pos[t, c, z]

    def get_num_positions(self) -> int:
        return self.positions


class HCSReader(ZarrReader):
    def __init__(self, store_path: StrOrBytesPath):
        super().__init__(store_path)
        self._get_axes_meta()
        self.plate_meta = PlateMeta(**self.root.attrs["plate"])

    def _get_rows(self):
        self.rows = []
        self.rows_meta = {}
        for i, row in enumerate(self.plate_meta["rows"]):
            self.rows.append(row["name"])
            self.rows_meta[row["name"]] = {
                "id": i,
                "meta": PlateAxisMeta(**row),
            }

    def _get_columns(self):
        self.columns = []
        self.columns_meta = {}
        for i, column in enumerate(self.plate_meta["columns"]):
            self.columns.append(column["name"])
            self.columns_meta[column["name"]] = {
                "id": i,
                "meta": PlateAxisMeta(**column),
            }

    def _get_wells(self):
        self.wells = []
        self.wells_meta = {}
        self.positions_meta = {}
        for well in self.plate_meta["wells"]:
            well_name = well["path"]
            self.wells.append(well_name)
            self.wells_meta[well_name] = {
                "meta": WellIndexMeta(**well),
                "positions": [],
                "image_meta_list": [
                    ImageMeta(**image_meta)
                    for image_meta in self.root[well_name].attrs["well"][
                        "images"
                    ]
                ],
            }
            for _, position in self.root[well_name].groups():
                self.wells_meta[well_name]["positions"].append(position.name)
                pos_attrs = ImagesMeta(**position.attrs)
                self.positions_meta[position.name] = {
                    "attrs": pos_attrs,
                    "id": pos_attrs.omero.id,
                }

    def _get_axes_meta(self):
        first_position = self.root[next(iter(self.positions_meta))]
        ms = first_position.attrs.get("multiscales")[0]
        warning = "Axes metadata not found. Using default."
        if ms:
            try:
                self.axes = MultiScaleMeta(**ms).axes
                return
            except KeyError:
                logging.warn(warning)
        else:
            logging.warn(warning)
        self.axes = _DEFAULT_AXES
