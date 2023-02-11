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
# only load arrays with name '0'.

position_list = (
    (row, col, fov)
    for row in "ABCDEFGH"
    for col in range(1, 13)
    for fov in range(2)
)

with open_ome_zarr(
    store_path,
    layout="hcs",
    mode="w",
    channel_names=["DAPI", "GFP", "Brightfield"],
) as dataset:
    # create and write to positions
    for i, (row, col, fov) in enumerate(position_list):
        position = dataset.create_position(row, col, fov)
        position["0"] = i*np.ones((2, 3, 5, 32, 32)).astype(np.uint16) 
        # 2 timepoints, 3 channels, 5 z-slices, 32x32 image
    # print dataset summary
    dataset.print_tree()
# Try viewing the hcs.zarr dataset in napari.

# %%
# Append a channel to all the positions

with open_ome_zarr(store_path, mode="r+") as dataset:
    for name, position in dataset.positions():
        print(f"Appending a channel to position: {name}")
        position.append_channel("Segmentation", resize_arrays=True)
        position["0"][:, 3] = np.random.randint(
            0, 2, size=(2, 5, 32, 32), dtype=np.uint16
        )
    dataset.print_tree()

# Try viewing the hcs.zarr dataset in napari.

# %%
