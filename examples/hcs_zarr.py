# %%
# This script writes a high content screening (HCS) OME-Zarr dataset
# with a single FOV and a single scaling level per well,
# and adds an extra well-position to an existing dataset.
# It can be run as a plain Python script, or as interactive cells in some IDEs.

import numpy as np

from iohub.ngff import HCSZarr

# %%
# Write 5D data to multiple wells

position_list = (
    ("A", "1", "0"),
    ("H", 12, 0),
    ("Control", "Blank", 0),
)

with HCSZarr.open(
    "hcs.zarr", mode="a", channel_names=["DAPI", "GFP"]
) as dataset:
    for row, col, fov in position_list:
        position = dataset.create_position(row, col, fov)
        position[0] = np.random.randint(
            0, np.iinfo(np.uint16).max, size=(5, 3, 2, 32, 32), dtype=np.uint16
        )

# %%
# Append a channel to all the positions

with HCSZarr.open("hcs.zarr", mode="r+") as dataset:
    for name, position in dataset.positions():
        print(name)
        position.append_channel("New", resize_arrays=True)
        position[0][:, 2] = np.random.randint(
            0, np.iinfo(np.uint16).max, size=(5, 3, 32, 32), dtype=np.uint16
        )

# %%
# Try viewing the images with napari-ome-zarr
