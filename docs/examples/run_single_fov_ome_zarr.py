"""
Single-FOV OME-Zarr
===================

This script shows how to create a single-FOV, single-scale OME-Zarr dataset,
read data in read-only mode,
append an extra time point to an existing dataset,
and adding a new channel to an existing dataset.
"""

# %%
import os
from tempfile import TemporaryDirectory

import numpy as np

from iohub.ngff import open_ome_zarr

# %%
# Set storage path
tmp_dir = TemporaryDirectory()
store_path = os.path.join(tmp_dir.name, "ome.zarr")
print("Zarr store path", store_path)

# %%
# Write 5D data to a new Zarr store

tczyx = np.random.randint(
    0, np.iinfo(np.uint16).max, size=(5, 2, 3, 32, 32), dtype=np.uint16
)

with open_ome_zarr(
    store_path, layout="fov", mode="a", channel_names=["DAPI", "GFP"]
) as dataset:
    dataset["img"] = tczyx

# %%
# Opening in read-only mode prevents writing

with open_ome_zarr(store_path, layout="auto", mode="r") as dataset:
    img = dataset["img"]
    print(img)
    print(img.numpy())
    try:
        img[0, 0, 0, 0, 0] = 0
    except Exception as e:
        print(f"Writing was rejected: {e}")

# %%
# Append a new timepoint to an existing dataset

new_1czyx = np.random.randint(
    0, np.iinfo(np.uint16).max, size=(1, 2, 3, 32, 32), dtype=np.uint16
)

with open_ome_zarr(store_path, layout="fov", mode="r+") as dataset:
    img = dataset["img"]
    print(img.shape)
    img.append(new_1czyx, axis=0)
    print(img.shape)

# %%
# Modify channels

# Open the dataset used above
dataset = open_ome_zarr(store_path, mode="r+")
dataset.print_tree()

# Append a new channel and write a Z-stack
new_zyx = np.random.randint(
    0, np.iinfo(np.uint16).max, size=(3, 32, 32), dtype=np.uint16
)
dataset.append_channel("New", resize_arrays=True)
dataset["img"][0, 2] = new_zyx
print(dataset.channel_names)
dataset.print_tree()

# Rename the new channel
dataset.rename_channel("New", "Renamed")
print(dataset.channel_names)

# Write new data to the channel
new_tzyx = np.random.randint(
    0, np.iinfo(np.uint16).max, size=(6, 3, 32, 32), dtype=np.uint16
)
c_idx = dataset.get_channel_index("Renamed")
dataset["img"][:, c_idx] = new_tzyx

# Which is equivalent to:
if False:  # remove this line
    dataset.update_channel("Renamed", target="img", data=new_tzyx)

# Close the dataset
dataset.close()

# %%
# Try viewing the images with napari-ome-zarr

# %%
# Clean up
tmp_dir.cleanup()