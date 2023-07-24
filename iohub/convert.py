import copy
import logging
import os
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from iohub._version import version as iohub_version
from iohub.ngff import Position, TransformationMeta, open_ome_zarr
from iohub.reader import (
    MicromanagerOmeTiffReader,
    MicromanagerSequenceReader,
    read_micromanager,
)

__all__ = ["TIFFConverter"]


def _create_grid_from_coordinates(
    xy_coords: list[tuple[float, float]], rows: int, columns: int
):
    """Function to create a grid from XY-position coordinates.
    Useful for generating HCS Zarr metadata.

    Parameters
    ----------
    xy_coords : list[tuple[float, float]]
        (X, Y) stage position list in the order in which it was acquired.
    rows : int
        number of rows in the grid-like acquisition
    columns : int
        number of columns in the grid-like acquisition

    Returns
    -------
    NDArray
        A grid-like array mimicking the shape of the acquisition where the
        value in the array corresponds to the position index at that location.
    """

    coords = dict()
    coords_list = []
    for idx, pos in enumerate(xy_coords):
        coords[idx] = pos
        coords_list.append(pos)

    # sort by X and then by Y
    coords_list.sort(key=lambda x: x[0])
    coords_list.sort(key=lambda x: x[1])

    # reshape XY coordinates into their proper 2D shape
    grid = np.reshape(coords_list, (rows, columns, 2))
    pos_index_grid = np.zeros((rows, columns), "uint16")
    keys = list(coords.keys())
    vals = list(coords.values())

    for row in range(rows):
        for col in range(columns):
            # append position index (key) into a final grid
            # by indexed into the coordinate map (values)
            pos_index_grid[row, col] = keys[vals.index(list(grid[row, col]))]

    return pos_index_grid


class TIFFConverter:
    """Convert Micro-Manager TIFF formats
    (single-page TIFF, OME-TIFF, ND-TIFF) into HCS OME-Zarr.
    Each FOV will be written to a separate well in the plate layout.

    Parameters
    ----------
    input_dir : str
        Input directory
    output_dir : str
        Output directory
    data_type : Literal['singlepagetiff', 'ometiff', 'ndtiff'], optional
        Input data type, by default None
    grid_layout : bool, optional
        Whether to lay out the positions in a grid-like format
        based on how the data was acquired
        (useful for tiled acquisitions), by default False
    chunks : tuple[int], optional
        Chunk size of the output Zarr arrays, by default None
        (chunk by XY planes, this is the fastest at converting time)
    label_positions : bool, optional
        Dump postion labels in MM metadata to Omero metadata,
        by default False
    scale_voxels : bool, optional
        Write voxel size (XY pixel size and Z-step) as scaling transform,
        by default True
    hcs_plate : bool, optional
        Create NGFF HCS layout based on position names from the
        HCS Site Generator in Micro-Manager (only available for OME-TIFF),
        and is ignored for other formats, by default None
        (attempt to apply to OME-TIFF datasets, disable this with ``False``)
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        data_type: Literal["singlepagetiff", "ometiff", "ndtiff"] = None,
        grid_layout: int = False,
        chunks: tuple[int] = None,
        label_positions: bool = False,
        scale_voxels: bool = True,
        hcs_plate: bool = None,
    ):
        logging.debug("Checking output.")
        if not output_dir.strip("/").endswith(".zarr"):
            raise ValueError("Please specify .zarr at the end of your output")
        self.output_dir = output_dir
        logging.info("Initializing data.")
        self.reader = read_micromanager(
            input_dir, data_type, extract_data=False
        )
        logging.debug("Finished initializing data.")
        self.summary_metadata = (
            self.reader.mm_meta["Summary"] if self.reader.mm_meta else None
        )
        self.save_name = os.path.basename(output_dir)
        logging.debug("Getting dataset summary information.")
        self.coord_map = dict()
        self.p = self.reader.get_num_positions()
        if self.p is None and isinstance(
            self.reader, MicromanagerSequenceReader
        ):
            # single page tiff reader may not return total positions
            # for `get_num_positions()`
            self.p = self.reader.num_positions
        self.t = self.reader.frames
        self.c = self.reader.channels
        self.z = self.reader.slices
        self.y = self.reader.height
        self.x = self.reader.width
        self.dim = (self.p, self.t, self.c, self.z, self.y, self.x)
        self.prefix_list = []
        self.label_positions = label_positions
        self.hcs_plate = hcs_plate
        self._check_hcs_sites()
        self._get_pos_names()
        logging.info(
            f"Found Dataset {self.save_name} with "
            f"dimensions (P, T, C, Z, Y, X): {self.dim}"
        )
        self._gen_coordset()
        self.metadata = dict()
        self.metadata["iohub_version"] = iohub_version
        self.metadata["Summary"] = self.summary_metadata
        if grid_layout:
            if hcs_plate:
                raise ValueError(
                    "grid_layout and hcs_plate must not be both true"
                )
            logging.info("Generating HCS plate level grid.")
            try:
                self.position_grid = _create_grid_from_coordinates(
                    *self._get_position_coords()
                )
            except ValueError:
                self._make_default_grid()
        else:
            self._make_default_grid()
        self.chunks = chunks if chunks else (1, 1, 1, self.y, self.x)
        self.transform = self._scale_voxels() if scale_voxels else None

    def _check_hcs_sites(self):
        is_mmstack = isinstance(self.reader, MicromanagerOmeTiffReader)
        if self.hcs_plate:
            if is_mmstack:
                self.hcs_sites = self.reader.hcs_position_labels
            else:
                raise ValueError(
                    f"HCS plate position not supported for {type(self.reader)}"
                )
        elif self.hcs_plate is None and is_mmstack:
            try:
                self.hcs_sites = self.reader.hcs_position_labels
                self.hcs_plate = True
            except ValueError:
                logging.debug(
                    "HCS sites not detected, "
                    "dumping all position into a single row."
                )

    def _make_default_grid(self):
        self.position_grid = np.expand_dims(
            np.arange(self.p, dtype=int), axis=0
        )

    def _gen_coordset(self):
        """Generates a coordinate set in the dimensional order
        to which the data was acquired.
        This is important for keeping track of where
        we are in the tiff file during conversion

        Returns
        -------
        list(tuples) w/ length [N_images]

        """

        # if acquisition information is not present
        # make an arbitrary dimension order
        if (
            not self.summary_metadata
            or "AxisOrder" not in self.summary_metadata.keys()
        ):
            self.p_dim = 0
            self.t_dim = 1
            self.c_dim = 2
            self.z_dim = 3

            self.dim_order = ["position", "time", "channel", "z"]

            # Assume data was collected slice first
            dims = [
                self.reader.slices,
                self.reader.channels,
                self.reader.frames,
                self.reader.get_num_positions(),
            ]

        # get the order in which the data was collected to minimize i/o calls
        else:
            # 4 possible dimensions: p, c, t, z
            n_dim = 4
            hashmap = {
                "position": self.p,
                "time": self.t,
                "channel": self.c,
                "z": self.z,
            }

            self.dim_order = copy.copy(self.summary_metadata["AxisOrder"])

            dims = []
            for i in range(n_dim):
                if i < len(self.dim_order):
                    dims.append(hashmap[self.dim_order[i]])
                else:
                    dims.append(1)

            # Reverse the dimension order and gather dimension indices
            self.dim_order.reverse()
            self.p_dim = self.dim_order.index("position")
            self.t_dim = self.dim_order.index("time")
            self.c_dim = self.dim_order.index("channel")
            self.z_dim = self.dim_order.index("z")

        # create array of coordinate tuples with innermost dimension
        # being the first dim acquired
        self.coords = [
            (dim3, dim2, dim1, dim0)
            for dim3 in range(dims[3])
            for dim2 in range(dims[2])
            for dim1 in range(dims[1])
            for dim0 in range(dims[0])
        ]

    def _get_position_coords(self):
        row_max = 0
        col_max = 0
        coords_list = []

        # TODO: read rows, cols directly from XY corods
        # TODO: account for non MM2gamma meta?
        if not self.reader.stage_positions:
            raise ValueError("Stage positions not available.")
        for idx, pos in enumerate(self.reader.stage_positions):
            stage_pos = pos.get("XYStage")
            if stage_pos is None:
                raise ValueError(
                    f"Stage position is not available for position {idx}"
                )
            coords_list.append(pos["XYStage"])
            row = pos["GridRow"]
            col = pos["GridCol"]
            row_max = row if row > row_max else row_max
            col_max = col if col > col_max else col_max

        return coords_list, row_max + 1, col_max + 1

    def _perform_image_check(self, zarr_img: NDArray, tiff_img: NDArray):
        if not np.array_equal(zarr_img, tiff_img):
            raise ValueError(
                "Converted Zarr image does not match the raw data. "
                "Conversion Failed."
            )

    def _get_pos_names(self):
        """Append a list of pos names in ascending order
        (order in which they were acquired).
        """
        if not self.label_positions:
            self.pos_names = ["0"] * self.p
        else:
            self.pos_names = []
            for p in range(self.p):
                if self.p > 1:
                    try:
                        name = self.summary_metadata["StagePositions"][p][
                            "Label"
                        ]
                    except KeyError:
                        name = None
                else:
                    name = None
                self.pos_names.append(name)

    def _get_image_array(self, p: int, t: int, c: int, z: int):
        try:
            return np.asarray(self.reader.get_image(p, t, c, z))
        except KeyError:
            # Converter will log a warning and
            # fill zeros if the image does not exist
            return None

    def _get_coord_reorder(self, coord):
        return (
            coord[self.p_dim],
            coord[self.t_dim],
            coord[self.c_dim],
            coord[self.z_dim],
        )

    def _get_channel_names(self):
        cns = self.reader.channel_names
        if not cns and isinstance(self.reader, MicromanagerSequenceReader):
            logging.warning(
                "Using an empty channel name for the single channel."
            )
            cns = [str(i) for i in range(self.c)]
        return cns

    def _scale_voxels(self):
        z_um = self.reader.z_step_size
        if self.z_dim > 1 and not z_um:
            logging.warning(
                "Z step size is not available. "
                "Setting the Z axis scaling factor to 1."
            )
            z_um = 1.0
        xy_warning = (
            " Setting X and Y scaling factors to 1."
            " Suppress this warning by setting `scale-voxels` to false."
        )
        if isinstance(self.reader, MicromanagerSequenceReader):
            logging.warning(
                "Pixel size detection is not supported for single-page TIFFs."
                + xy_warning
            )
            xy_um = 1.0
        else:
            try:
                xy_um = self.reader.xy_pixel_size
            except AttributeError as e:
                logging.warning(str(e) + xy_warning)
                xy_um = 1.0
        return [
            TransformationMeta(
                type="scale", scale=[1.0, 1.0, z_um, xy_um, xy_um]
            )
        ]

    def _init_zarr_arrays(self):
        self.writer = open_ome_zarr(
            self.output_dir,
            layout="hcs",
            mode="w-",
            channel_names=self._get_channel_names(),
            version="0.4",
        )
        self.zarr_position_names = []
        arr_kwargs = {
            "name": "0",
            "shape": (
                self.t if self.t != 0 else 1,
                self.c if self.c != 0 else 1,
                self.z if self.z != 0 else 1,
                self.y,
                self.x,
            ),
            "dtype": self.reader.dtype,
            "chunks": self.chunks,
            "transform": self.transform,
        }
        if self.hcs_plate:
            self._init_hcs_arrays(arr_kwargs)
        else:
            self._init_grid_arrays(arr_kwargs)

    def _init_hcs_arrays(self, arr_kwargs):
        for row, col, fov in self.hcs_sites:
            self._create_zeros_array(row, col, fov, arr_kwargs)
        logging.info(
            "Created HCS NGFF layout from Micro-Manager HCS position labels."
        )
        self.writer.print_tree()

    def _init_grid_arrays(self, arr_kwargs):
        for row, columns in enumerate(self.position_grid):
            for column in columns:
                self._create_zeros_array(row, column, "0", arr_kwargs)

    def _create_zeros_array(
        self, row_name: str, col_name: str, pos_name: str, arr_kwargs: dict
    ) -> Position:
        pos = self.writer.create_position(row_name, col_name, pos_name)
        self.zarr_position_names.append(pos.zgroup.name)
        _ = pos.create_zeros(**arr_kwargs)
        pos.metadata.omero.name = self.pos_names[
            len(self.zarr_position_names) - 1
        ]
        pos.dump_meta()

    def run(self, check_image: bool = True):
        """Runs the conversion.

        Parameters
        ----------
        check_image : bool, optional
            Whether to check that the written Zarr array has the same
            pixel values as in TIFF files, by default True
        """
        logging.debug("Setting up Zarr store.")
        self._init_zarr_arrays()
        bar_format = (
            "Status: |{bar:16}|{n_fmt}/{total_fmt} "
            "(Time Remaining: {remaining}), {rate_fmt}{postfix}]"
        )
        # Run through every coordinate and convert in acquisition order
        logging.info("Converting Images...")
        for coord in tqdm(self.coords, bar_format=bar_format):
            coord_reorder = self._get_coord_reorder(coord)
            img_raw = self._get_image_array(*coord_reorder)
            if img_raw is None or not getattr(img_raw, "shape", ()):
                # Leave incomplete datasets zero-filled
                logging.warning(
                    f"Cannot load image at PTCZ={coord_reorder}, "
                    "filling with zeros. Check if the raw data is incomplete."
                )
                continue
            pos_name = self.zarr_position_names[coord_reorder[0]]
            zarr_img = self.writer[pos_name]["0"]
            zarr_img[coord_reorder[1:]] = img_raw
            if check_image:
                self._perform_image_check(zarr_img[coord_reorder[1:]], img_raw)
        self.writer.zgroup.attrs.update(self.metadata)
        self.writer.close()
