# %%
# This script shows how to write a high content screening (HCS) OME-Zarr dataset
# with a single FOV and a single scaling level per well,
# and how to add an extra well-position to an existing dataset.
# It can be run as a plain Python script, or as interactive cells in some IDEs.

import zarr
import numpy as np
from iohub.zarrfile import HCSReader
from iohub.writer import create_zarr_store, HCSWriter

# %%
# Create a new zarr store and open it

store = create_zarr_store("hcs.zarr")
root = zarr.open(store, mode="a")

# %%
# Initialize a writer object
writer = HCSWriter(root, channel_names=["Retardance", "Orientation"])

writer.channel_names.append("Phase")
# %%
# This function writes some random 5-D data to a given position

def write_dummy_5d(writer: HCSWriter, position: zarr.Group):
    tczyx = np.random.randint(
        0, np.iinfo(np.uint16).max, size=(5, 3, 3, 32, 32), dtype=np.uint16
    )
    for t, czyx in enumerate(tczyx):
        for c, zyx in enumerate(czyx):
            writer.write_zstack(zyx, position, t, c)


# %%
# Define some positions (row, column, FOV)

positions = (
    ("A", "1", "0"),
    ("A", "12", "0"),
    ("H", "6", "0"),
)

# %%
# Write to the positions

for pos in positions:
    p = writer.require_position(*pos)
    write_dummy_5d(writer, p)

writer.close()

# %%
# The sections below shows how to add a new position
# to an existing store (that we just created).

#%%
# Populate a new writer from the reader.

reader = HCSReader('hcs.zarr')
writer = HCSWriter.from_reader(reader)

#%%
# Alternatively, use the convenience method 
# if default parameters does not need to be changed

while False: # FIXME: remove this clause and the following indentation
    writer = HCSWriter.open("hcs.zarr", mode="r+")


# %%
# Add an FOV in a new well

position = writer.require_position("D", "8", '0')
write_dummy_5d(writer, position)