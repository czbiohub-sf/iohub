# %%
# RUNNING THE FOLLOWING CODE WILL WRITE A LARGE FILE
# This script shows how to store a larger-than-RAM array to a OME-Zarr dataset.
# The same methods works for HCS datasets too.
# It can be run as a plain Python script,
# or as interactive cells in some IDEs.


import numpy as np
from tqdm import tqdm

from iohub.ngff import open_ome_zarr

# %%
# FIXME: set Zarr store path here
store_path = ""

# %%
# shape and data type of the large array
# this array is about 10 GB, each time point is about 100 MB
# it may not be actually larger than RAM but enough for demo
# monitor the memory usage of python when the following runs
# and it should take significantly less than 10 GB
shape = (100, 2, 25, 1024, 1024)
dtype = np.uint16


# %%
# store this array by looping through the time points
if store_path:
    with open_ome_zarr(
        store_path, layout="fov", mode="w-", channel_names=["DAPI", "GFP"]
    ) as dataset:
        img = dataset.create_zeros(
            name="0",
            shape=shape,
            dtype=dtype,
            chunks=(1, 1, 1, 1024, 1024),  # chunk by XY planes
        )
        for t in tqdm(range(shape[0])):
            # write 4D image data for the time point
            img[t] = np.ones(shape[1:]) * t
        dataset.print_tree()
