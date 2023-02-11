# %%
# This script writes a tiled (multi-FOV) zarr dataset.
# It is not spcified by the NGFF specification, but is a common use case.
# We repurpose the HCS layout from the NGFF specification to make multi-position data to view with napari-ome-zarr.


import os

import numpy as np

from iohub.ngff import open_ome_zarr

# %%
# Set storage path

store_path = f'{os.path.expanduser("~/")}tiles.zarr'

# %%
# Write 5D data to multiple positions.
# While the NGFF specification allows for arbitrary names,
# the ome-zarr-py library (thus the napari-ome-zarr plugin)
# only load arrays with name '0'.

position_list = ((row, col) for row in ("row_0", "row_1") for col in range(10))

with open_ome_zarr(
    store_path,
    layout="tiles",
    mode="w",
    channel_names=["DAPI", "GFP", "Brightfield"],
) as dataset:
    # create and write to positions
    for i, (row, col) in enumerate(position_list):
        position = dataset.create_position(row, col)
        # Following usage (set the data property) is more intuitive. Can we implement this?
        # position.data = i*np.ones((2, 3, 5, 32, 32)).astype(np.uint16) 
        position["0"] = i*np.ones((2, 3, 5, 32, 32)).astype(np.uint16)
        # 2 timepoints, 3 channels, 5 z-slices, 32x32 image at each position
    # print dataset summary
    dataset.print_tree()
# Try viewing the tiles.zarr dataset in napari or each position individually.