"""
Multi-FOV HCS OME-Zarr
======================

This script writes a high content screening (HCS) OME-Zarr dataset
with a single FOV and a single scaling level per well,
and adds an extra well-position to an existing dataset.
"""

# %%

import os
from tempfile import TemporaryDirectory

import numpy as np

from iohub.ngff import open_ome_zarr

# %%
# Set storage path

tmp_dir = TemporaryDirectory()
store_path = os.path.join(tmp_dir.name, "hcs.zarr")
print("Zarr store path", store_path)

# %%
# Write 5D data to multiple wells.
# Integer path names will be automatically converted to strings.
# While the NGFF specification (and iohub) allows for arbitrary names,
# the ome-zarr-py library and the napari-ome-zarr viewer
# can only load positions and arrays with name '0' for the whole plate.

position_list = (
    ("A", "1", "0"),
    ("H", 1, "0"),
    ("H", "12", "CannotVisualize"),
    ("Control", "Blank", 0),
)

with open_ome_zarr(
    store_path,
    layout="hcs",
    mode="w-",
    channel_names=["DAPI", "GFP", "Brightfield"],
) as dataset:
    # Create and write to positions
    # This affects the tile arrangement in visualization
    for row, col, fov in position_list:
        position = dataset.create_position(row, col, fov)
        position["0"] = np.random.randint(
            0, np.iinfo(np.uint16).max, size=(5, 3, 2, 32, 32), dtype=np.uint16
        )
    # Print dataset summary
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

# %%
# Clean up
tmp_dir.cleanup()
