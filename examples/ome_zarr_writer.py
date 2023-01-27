# %%
# This script shows how to write a single-FOV, single-scale OME-Zarr dataset,
# and how to append an extra channel to an existing dataset.
# It can be run as a plain Python script, or as interactive cells in some IDEs.

import numpy as np
from iohub.ngff import OMEZarr

# %%
# Write 5D data to a new Zarr store

tczyx = np.random.randint(
    0, np.iinfo(np.uint16).max, size=(5, 2, 3, 32, 32), dtype=np.uint16
)

with OMEZarr.open(
    "ome.zarr", mode="a", channel_names=["DAPI", "GFP"]
) as dataset:
    dataset[0] = tczyx

# %%
# Opening in read-only mode prevents writing

with OMEZarr.open("ome.zarr", mode="r") as dataset:
    img = dataset[0]
    print(img.numpy())
    try:
        img[0, 0, 0, 0, 0] = 0
    except:
        print("Writing was rejected.")

# %%
# Append a new timepoint to an existing dataset

new_1czyx = np.random.randint(
    0, np.iinfo(np.uint16).max, size=(1, 2, 3, 32, 32), dtype=np.uint16
)

with OMEZarr.open("ome.zarr", mode="r+") as dataset:
    img = dataset[0]
    print(img.shape)
    img.append(new_1czyx, axis=0)
    print(img.shape)

# %%
# Add a new channel and write a Z-stack

new_zyx = np.random.randint(
    0, np.iinfo(np.uint16).max, size=(3, 32, 32), dtype=np.uint16
)

dataset = OMEZarr.open("ome.zarr", mode="r+")
dataset.append_channel("New", resize_arrays=True)
dataset[0][0, 2] = new_zyx
dataset.close()

# %%
# Try viewing the images with napari-ome-zarr
