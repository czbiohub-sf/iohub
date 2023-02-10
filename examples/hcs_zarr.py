# %%
# This script writes a high content screening (HCS) OME-Zarr dataset
# with a single FOV and a single scaling level per well,
# and adds an extra well-position to an existing dataset.
# It can be run as a plain Python script, or as interactive cells in some IDEs.

import numpy as np

from iohub.ngff import HCSZarr

import os

# %%
# Write 5D data to multiple wells

position_list = (
    ("A", "1", "0"),
    ("H", 10, 0),
    ("Control", "Blank", 0),
)

file_path = f'{os.path.expanduser("~/")}hcs.zarr'

with HCSZarr.open(
    file_path, mode="a", channel_names=["DAPI", "GFP", "Brightfield"]
) as dataset:
    for row, col, fov in position_list:
        position = dataset.create_position(row, col, fov)
        position["raw"] = np.random.randint(
            0, np.iinfo(np.uint16).max, size=(5, 3, 2, 32, 32), dtype=np.uint16
        )
# "raw" is the custom name for the array in the OME-Zarr file.
# The default name is "0".
# Dragging above hcs.zarr into napari-ome-zarr doesn't load the array,
# since the plugin expects the array name to default to "0" .
# The following zarr will open with each position shown as a tile:

file_path = f'{os.path.expanduser("~/")}hcs_default.zarr'

with HCSZarr.open(
    file_path, mode="a", channel_names=["DAPI", "GFP", "Brightfield"]
) as dataset:
    for row, col, fov in position_list:
        position = dataset.create_position(row, col, fov)
        position["0"] = np.random.randint(
            0, np.iinfo(np.uint16).max, size=(5, 3, 2, 32, 32), dtype=np.uint16
        )

# Examine the shape, size, dtype, and chunks of the array at a position.
print(f'Array 0 @ Well A1, FOV 0:\n'
      f'shape:{dataset["A"]["1"]["0"]["0"].shape}\n'
      f'size:{dataset["A"]["1"]["0"]["0"].size}\n'
      f'dtype:{dataset["A"]["1"]["0"]["0"].dtype}\n'
      f'chunks:{dataset["A"]["1"]["0"]["0"].chunks}\n')

# %%
# Append a channel to all the positions

file_path = f'{os.path.expanduser("~/")}hcs_default.zarr'

with HCSZarr.open(file_path, mode="r+") as dataset:
    for name, position in dataset.positions():
        print(name)
        position.append_channel("Segmentation", resize_arrays=True)
        position['0'][:, 3] = np.random.randint(
            0, np.iinfo(np.uint16).max, size=(5, 2, 32, 32), dtype=np.uint16
        )

print(f'Array 0 @ Well A1, FOV 0:\n'
      f'shape:{dataset["A"]["1"]["0"]["0"].shape}\n'
      f'size:{dataset["A"]["1"]["0"]["0"].size}\n'
      f'dtype:{dataset["A"]["1"]["0"]["0"].dtype}\n'
      f'chunks:{dataset["A"]["1"]["0"]["0"].chunks}\n')

# %%
# Try viewing the images with napari-ome-zarr
