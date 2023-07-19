"""
Writing a Large Array
=====================

This script shows how to store a larger-than-RAM array to a OME-Zarr dataset.
The same method works for HCS datasets too.
"""

# %%
# .. warning:: Executing this example will write a large file.
#     Modify the store path manually.
import numpy as np
from tqdm import tqdm

from iohub.ngff import open_ome_zarr

store_path = ""

# %%
# Shape and data type of the large array
# this array is about 10 GB, each time point is about 100 MB.
# It may not be actually larger than RAM but enough for demo.
# Monitor the memory usage of Python when the following runs
# and it should take significantly less than 10 GB.
shape = (100, 2, 25, 1024, 1024)
dtype = np.uint16


# %%
# .. note:: This will not run if the Zarr store path is not set above
#
# Store this array by looping through the time points
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
