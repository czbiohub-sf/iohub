import json
import logging
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
from dask.array import to_zarr
from tqdm import tqdm
from tqdm.contrib.itertools import product
from tqdm.contrib.logging import logging_redirect_tqdm

from iohub.ngff.models import TransformationMeta
from iohub.ngff.nodes import Position, open_ome_zarr
from iohub.reader import MMStack, NDTiffDataset, read_images

__all__ = ["TIFFConverter"]
_logger = logging.getLogger(__name__)

MAX_CHUNK_SIZE = 500e6  # in bytes


def _create_grid_from_coordinates(
    xy_coords: list[tuple[float, float]], rows: int, columns: int
):
    """Create a grid from XY-position coordinates.

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
    (OME-TIFF, ND-TIFF) into HCS OME-Zarr.
    Each FOV will be written to a separate well in the plate layout.

    Parameters
    ----------
    input_dir : str | Path
        Input directory path
    output_dir : str | Path
        Output zarr directory path
    grid_layout : bool, optional
        Whether to lay out the positions in a grid-like format
        based on how the data was acquired
        (useful for tiled acquisitions), by default False
    chunks : tuple[int] or Literal['XY', 'XYZ'], optional
        Chunk size of the output Zarr arrays, by default None
        (chunk by XYZ volumes or 500 MB size limit, whichever is smaller)
    hcs_plate : bool, optional
        Create NGFF HCS layout based on position names from the
        HCS Site Generator in Micro-Manager (only available for OME-TIFF),
        and is ignored for other formats, by default None
        (attempt to apply to OME-TIFF datasets, disable this with ``False``)

    Notes
    -----
    The image plane metadata for each FOV is aggregated into a JSON file,
    and placed under the Zarr array directory
    (e.g. ``/row/column/fov/0/image_plane_metadata.json``).
    """

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        grid_layout: int = False,
        chunks: tuple[int] | Literal["XY", "XYZ"] = None,
        hcs_plate: bool = None,
    ):
        _logger.debug("Checking output.")
        output_dir = Path(output_dir)
        if ".zarr" not in output_dir.suffixes:
            raise ValueError("Please specify .zarr at the end of your output")
        self.output_dir = output_dir
        _logger.info("Initializing data.")
        self.reader = read_images(input_dir)
        if reader_type := type(self.reader) not in (
            MMStack,
            NDTiffDataset,
        ):
            raise TypeError(
                f"Reader type {reader_type} not supported for conversion."
            )
        _logger.debug("Finished initializing data.")
        self.summary_metadata = self.reader.micromanager_summary
        self.save_name = output_dir.name
        _logger.debug("Getting dataset summary information.")
        self.coord_map = dict()
        self.p = len(self.reader)
        self.t = self.reader.frames
        self.c = self.reader.channels
        self.z = self.reader.slices
        self.y = self.reader.height
        self.x = self.reader.width
        self.dim = (self.p, self.t, self.c, self.z, self.y, self.x)
        self.prefix_list = []
        self.hcs_plate = hcs_plate
        self._check_hcs_sites()
        self._get_pos_names()
        _logger.info(
            f"Found Dataset {input_dir} with "
            f"dimensions (P, T, C, Z, Y, X): {self.dim}"
        )
        self.metadata = dict()
        self.metadata["iohub_version"] = version("iohub")
        self.metadata["Summary"] = self.summary_metadata
        if grid_layout:
            if hcs_plate:
                raise ValueError(
                    "grid_layout and hcs_plate must not be both true"
                )
            _logger.info("Generating HCS plate level grid.")
            try:
                self.position_grid = _create_grid_from_coordinates(
                    *self._get_position_coords()
                )
            except ValueError as e:
                _logger.warning(f"Failed to generate grid layout: {e}")
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
                _logger.debug(
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
        """Get the position coordinates from the reader metadata.

        Raises:
            ValueError: If stage positions are not available.

        Returns:
            list: XY stage position coordinates.
            int: Number of grid rows.
            int: Number of grid columns.
        """
        rows = set()
        cols = set()
        xy_coords = []

        # TODO: account for non MM2gamma meta?
        if not self.reader.stage_positions:
            raise ValueError("Stage positions not available.")
        for idx, pos in enumerate(self.reader.stage_positions):
            try:
                xy_stage = pos["DefaultXYStage"]
                stage_pos = pos[xy_stage]
            except KeyError:
                raise ValueError(
                    f"Stage position is not available for position {idx}"
                )
            xy_coords.append(stage_pos)
            try:
                rows.add(pos["GridRow"])
                cols.add(pos["GridCol"])
            except KeyError:
                raise ValueError(
                    f"Grid indices not available for position {idx}"
                )

        return xy_coords, len(rows), len(cols)

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
            _logger.debug("No chunk size specified, using ZYX.")
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
        bytes_per_pixel = np.dtype(self.reader.dtype).itemsize
        # it's OK if a single image is larger than MAX_CHUNK_SIZE
        while (
            chunks[-3] > 1
            and np.prod(chunks, dtype=np.int64) * bytes_per_pixel
            > MAX_CHUNK_SIZE
        ):
            chunks[-3] = np.ceil(chunks[-3] / 2).astype(int)

        _logger.debug(f"Zarr store chunk size will be set to {chunks}.")

        return tuple(chunks)

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
            channel_names=self.reader.channel_names,
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
        _logger.info(
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

    def _convert_image_plane_metadata(self, fov, zarr_name: str):
        position_image_plane_metadata = {}
        sorted_keys = []
        for t_idx, c_idx in product(
            range(self.t),
            range(self.c),
            desc="Converting frame metadata",
            unit="frame",
            leave=False,
            ncols=80,
        ):
            c_key = c_idx
            if isinstance(self.reader, NDTiffDataset):
                c_key = (
                    self.reader.channel_names[c_idx]
                    if self.reader.str_channel_axis
                    else c_idx
                )
            missing_data_warning_issued = False
            for z_idx in range(self.z):
                metadata = fov.frame_metadata(t=t_idx, c=c_key, z=z_idx)
                if metadata is None:
                    if not missing_data_warning_issued:
                        missing_data_warning_issued = True
                        _logger.warning(
                            f"Cannot load data at P: {zarr_name}, T: {t_idx}, "
                            f"C: {c_idx}, filling with zeros. Raw data may be "
                            "incomplete."
                        )
                    continue
                if not sorted_keys:
                    # Sort keys, ordering keys without dashes first
                    sorted_keys = sorted(
                        metadata.keys(), key=lambda x: ("-" in x, x)
                    )

                sorted_metadata = {key: metadata[key] for key in sorted_keys}
                # T/C/Z
                frame_key = "/".join([str(i) for i in (t_idx, c_idx, z_idx)])
                position_image_plane_metadata[frame_key] = sorted_metadata
        with open(
            self.output_dir / zarr_name / "image_plane_metadata.json",
            mode="x",
        ) as metadata_file:
            json.dump(position_image_plane_metadata, metadata_file, indent=4)

    def __call__(self) -> None:
        """
        Runs the conversion.

        Examples
        --------
        >>> from iohub.convert import TIFFConverter
        >>> converter = TIFFConverter("input/path/", "output/path/")
        >>> converter()
        """
        _logger.debug("Setting up Zarr store.")
        self._init_zarr_arrays()
        # Run through every coordinate and convert in acquisition order
        _logger.debug("Converting images.")
        with logging_redirect_tqdm():
            for zarr_pos_name, (_, fov) in tqdm(
                zip(self.zarr_position_names, self.reader, strict=True),
                total=len(self.reader),
                desc="Converting images",
                unit="FOV",
                ncols=80,
            ):
                zarr_img = self.writer[zarr_pos_name]["0"]
                to_zarr(fov.xdata.data.rechunk(self.chunks), zarr_img)
                self._convert_image_plane_metadata(fov, zarr_img.path)
        self.writer.zgroup.attrs.update(self.metadata)
        self.writer.close()
        self.reader.close()
