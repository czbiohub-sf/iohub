# %%
# This script writes a high content screening (HCS) OME-Zarr dataset
# with a single FOV and a single scaling level per well,
# and adds an extra well-position to an existing dataset.
# It can be run as a plain Python script, or as interactive cells in some IDEs.

import os

import numpy as np

from iohub.ngff import open_ome_zarr

# %%
# Set storage path

store_path = f'{os.path.expanduser("~/")}hcs.zarr'

# %%
# Write 5D data to multiple wells.
# While the NGFF specification allows for arbitrary names,
# the ome-zarr-py library (thus the napari-ome-zarr plugin)
# only load positions and arrays with name '0'.

position_list = (
    ("A", "1", "0"),
    ("H", 10, 0),
    ("Control", "Blank", 0),
)

with open_ome_zarr(
    store_path,
    layout="hcs",
    mode="a",
    channel_names=["DAPI", "GFP", "Brightfield"],
) as dataset:
    # create and write to positions
    for row, col, fov in position_list:
        position = dataset.create_position(row, col, fov)
        position["0"] = np.random.randint(
            0, np.iinfo(np.uint16).max, size=(5, 3, 2, 32, 32), dtype=np.uint16
        )
    # print dataset summary
    dataset.print_tree()

# %%
# Append a channel to all the positions

with open_ome_zarr(store_path, mode="r+") as dataset:
    for name, position in dataset.positions():
        print(f"Appending a channel to position: {name}")
        position.append_channel("Segmentation", resize_arrays=True)
        position["0"][:, 3] = np.random.randint(
            0, np.iinfo(np.uint16).max, size=(5, 2, 32, 32), dtype=np.uint16
        )
    dataset.print_tree()

# %%
# Try viewing the images with napari-ome-zarr
