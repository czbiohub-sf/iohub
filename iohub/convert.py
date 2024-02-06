import json
import logging
import os
from typing import Literal

import numpy as np
from dask.array import to_zarr
from tqdm import tqdm
from tqdm.contrib.itertools import product

from iohub._version import version as iohub_version
from iohub.ngff import Position, TransformationMeta, open_ome_zarr
from iohub.reader import (
    MMStack,
    # MicromanagerSequenceReader,
    NDTiffDataset,
    read_micromanager,
)

__all__ = ["TIFFConverter"]
MAX_CHUNK_SIZE = 500e6  # in bytes


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
        Input directory path
    output_dir : str
        Output zarr directory path
    data_type : Literal['singlepagetiff', 'ometiff', 'ndtiff'], optional
        Input data type, by default None
    grid_layout : bool, optional
        Whether to lay out the positions in a grid-like format
        based on how the data was acquired
        (useful for tiled acquisitions), by default False
    chunks : tuple[int] or Literal['XY', 'XYZ'], optional
        Chunk size of the output Zarr arrays, by default None
        (chunk by XY planes, this is the fastest at converting time)
    hcs_plate : bool, optional
        Create NGFF HCS layout based on position names from the
        HCS Site Generator in Micro-Manager (only available for OME-TIFF),
        and is ignored for other formats, by default None
        (attempt to apply to OME-TIFF datasets, disable this with ``False``)

    Notes
    -----
    When converting ND-TIFF, the image plane metadata for all frames
    are aggregated into a file named ``image_plane_metadata.json``,
    and placed under the root Zarr group (alongside plate metadata).
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        data_type: Literal["singlepagetiff", "ometiff", "ndtiff"] = None,
        grid_layout: int = False,
        chunks: tuple[int] | Literal["XY", "XYZ"] = None,
        hcs_plate: bool = None,
    ):
        logging.debug("Checking output.")
        if not output_dir.strip("/").endswith(".zarr"):
            raise ValueError("Please specify .zarr at the end of your output")
        self.output_dir = output_dir
        logging.info("Initializing data.")
        self.reader = read_micromanager(input_dir, data_type)
        if reader_type := type(self.reader) not in (
            MMStack,
            NDTiffDataset,
        ):
            raise TypeError(
                f"Reader type {reader_type} not supported for conversion."
            )
        logging.debug("Finished initializing data.")
        self.summary_metadata = (
            self.reader.mm_meta["Summary"] if self.reader.mm_meta else None
        )
        self.save_name = os.path.basename(output_dir)
        logging.debug("Getting dataset summary information.")
        self.coord_map = dict()
        self.p = len(self.reader)
        self.t = self.reader.frames
        self.c = self.reader.channels
        self.z = self.reader.slices
        self.y = self.reader.height
        self.x = self.reader.width
        self.dtype = self.reader.dtype
        self.dim = (self.p, self.t, self.c, self.z, self.y, self.x)
        self.prefix_list = []
        self.hcs_plate = hcs_plate
        self._check_hcs_sites()
        self._get_pos_names()
        logging.info(
            f"Found Dataset {self.save_name} with "
            f"dimensions (P, T, C, Z, Y, X): {self.dim}"
        )
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
        self.chunks = self._gen_chunks(chunks)
        self.transform = self._scale_voxels()

    def _check_hcs_sites(self):
        if self.hcs_plate:
            self.hcs_sites = self.reader.hcs_position_labels
        elif self.hcs_plate is None:
            try:
                self.hcs_sites = self.reader.hcs_position_labels
                self.hcs_plate = True
            except ValueError:
                logging.debug(
                    "HCS sites not detected, "
                    "dumping all position into a single row."
                )

    def _make_default_grid(self):
        if isinstance(self.reader, NDTiffDataset):
            self.position_grid = np.array([self.pos_names])
        else:
            self.position_grid = np.expand_dims(
                np.arange(self.p, dtype=int), axis=0
            )

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

    def _get_pos_names(self):
        """Append a list of pos names in ascending order
        (order in which they were acquired).
        """
        self.pos_names = []
        for p in range(self.p):
            try:
                name = self.reader.stage_positions[p]["Label"]
            except (IndexError, KeyError):
                name = str(p)
            self.pos_names.append(name)

    def _gen_chunks(self, input_chunks):
        if not input_chunks:
            logging.debug("No chunk size specified, using ZYX.")
            chunks = [1, 1, self.z, self.y, self.x]
        elif isinstance(input_chunks, tuple):
            chunks = list(input_chunks)
        elif isinstance(input_chunks, str):
            if input_chunks.lower() == "xy":
                chunks = [1, 1, 1, self.y, self.x]
            elif input_chunks.lower() == "xyz":
                chunks = [1, 1, self.z, self.y, self.x]
            else:
                raise ValueError(f"{input_chunks} chunks are not supported.")
        else:
            raise TypeError(
                f"Chunk type {type(input_chunks)} is not supported."
            )

        # limit chunks to MAX_CHUNK_SIZE bytes
        bytes_per_pixel = np.dtype(self.dtype).itemsize
        # it's OK if a single image is larger than MAX_CHUNK_SIZE
        while (
            chunks[-3] > 1
            and np.prod(chunks, dtype=np.int64) * bytes_per_pixel
            > MAX_CHUNK_SIZE
        ):
            chunks[-3] = np.ceil(chunks[-3] / 2).astype(int)

        logging.debug(f"Zarr store chunk size will be set to {chunks}.")

        return tuple(chunks)

    def _get_channel_names(self):
        cns = self.reader.channel_names
        if not cns:
            logging.warning(
                "Cannot find channel names, using indices instead."
            )
            cns = [str(i) for i in range(self.c)]
        return cns

    def _scale_voxels(self):
        return [
            TransformationMeta(
                type="scale", scale=[1.0, 1.0, *self.reader.zyx_scale]
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
            "dtype": self.dtype,
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

    def _convert_ndtiff(self):
        bar_format_positions = (
            "Converting Positions: |{bar:16}|{n_fmt}/{total_fmt} "
            "(Time Remaining: {remaining}), {rate_fmt}{postfix}]"
        )
        bar_format_time_channel = (
            "Converting Timepoints/Channels: |{bar:16}|{n_fmt}/{total_fmt} "
            "(Time Remaining: {remaining}), {rate_fmt}{postfix}]"
        )
        for p_idx in tqdm(range(self.p), bar_format=bar_format_positions):
            position_image_plane_metadata = {}

            # ndtiff_pos_idx, ndtiff_t_idx, and ndtiff_channel_idx
            # may be None
            ndtiff_pos_idx = (
                self.pos_names[p_idx]
                if self.reader.str_position_axis
                else p_idx
            )
            try:
                ndtiff_pos_idx, *_ = self.reader._check_coordinates(
                    ndtiff_pos_idx, 0, 0, 0
                )
            except ValueError:
                # Log warning and continue if some positions were not
                # acquired in the dataset
                logging.warning(
                    f"Cannot load data at position {ndtiff_pos_idx}, "
                    "filling with zeros. Raw data may be incomplete."
                )
                continue

            dask_arr = self.reader.get_zarr(position=ndtiff_pos_idx)
            zarr_pos_name = self.zarr_position_names[p_idx]
            zarr_arr = self.writer[zarr_pos_name]["0"]

            for t_idx, c_idx in product(
                range(self.t),
                range(self.c),
                bar_format=bar_format_time_channel,
                position=1,
                leave=False,
            ):
                ndtiff_channel_idx = (
                    self.reader.channel_names[c_idx]
                    if self.reader.str_channel_axis
                    else c_idx
                )
                # set ndtiff_t_idx and ndtiff_z_idx to None if these axes were
                # not acquired
                (
                    _,
                    ndtiff_t_idx,
                    ndtiff_channel_idx,
                    ndtiff_z_idx,
                ) = self.reader._check_coordinates(
                    ndtiff_pos_idx, t_idx, ndtiff_channel_idx, 0
                )
                # Log warning and continue if some T/C were not acquired in the
                # dataset
                if not self.reader.dataset.has_image(
                    position=ndtiff_pos_idx,
                    time=ndtiff_t_idx,
                    channel=ndtiff_channel_idx,
                    z=ndtiff_z_idx,
                ):
                    logging.warning(
                        f"Cannot load data at timepoint {t_idx},  channel "
                        f"{c_idx}, filling with zeros. Raw data may be "
                        "incomplete."
                    )
                    continue

                data_slice = (slice(t_idx, t_idx + 1), slice(c_idx, c_idx + 1))
                to_zarr(
                    dask_arr[data_slice].rechunk(self.chunks),
                    zarr_arr,
                    region=data_slice,
                )

                for z_idx in range(self.z):
                    # this function will handle z_idx=0 when no z stacks
                    # acquired
                    image_metadata = self.reader.get_image_metadata(
                        ndtiff_pos_idx,
                        ndtiff_t_idx,
                        ndtiff_channel_idx,
                        z_idx,
                    )
                    # T/C/Z
                    frame_key = "/".join(
                        [str(i) for i in (t_idx, c_idx, z_idx)]
                    )
                    position_image_plane_metadata[frame_key] = image_metadata

            logging.info("Writing ND-TIFF image plane metadata...")
            # image plane metadata is save in
            # output_dir/row/well/fov/img/image_plane_metadata.json,
            # e.g. output_dir/A/1/FOV0/0/image_plane_metadata.json
            with open(
                os.path.join(
                    self.output_dir, zarr_arr.path, "image_plane_metadata.json"
                ),
                mode="x",
            ) as metadata_file:
                json.dump(
                    position_image_plane_metadata, metadata_file, indent=4
                )

    def __call__(self):
        """Runs the conversion."""
        logging.debug("Setting up Zarr store.")
        self._init_zarr_arrays()
        bar_format_images = (
            "Converting Images: |{bar:16}|{n_fmt}/{total_fmt} "
            "(Time Remaining: {remaining}), {rate_fmt}{postfix}]"
        )
        # Run through every coordinate and convert in acquisition order
        logging.info("Converting Images...")
        match self.reader:
            case NDTiffDataset():
                self._convert_ndtiff()
            case MMStack():
                for p_idx in tqdm(range(self.p), bar_format=bar_format_images):
                    da = self.reader[p_idx].xdata.data
                    zarr_pos_name = self.zarr_position_names[p_idx]
                    zarr_arr = self.writer[zarr_pos_name]["0"]
                    to_zarr(da.rechunk(self.chunks), zarr_arr)
            case _:
                raise NotImplementedError()

        self.writer.zgroup.attrs.update(self.metadata)
        self.writer.close()
