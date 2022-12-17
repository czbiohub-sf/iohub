# %%
# This script shows how to write a single-FOV, single-scale OME-Zarr dataset,
# and how to append an extra channel to an existing dataset.
# It can be run as a plain Python script, or as interactive cells in some IDEs.

import zarr
import numpy as np
from iohub.writer import OMEZarrWriter

# %%
# Create a new zarr store

store = zarr.DirectoryStore("ome.zarr", dimension_separator="/")

# %%
# Initialize a writer object

writer = OMEZarrWriter(zarr.group(store), channel_names=["DAPI", "GFP"])

# %%
# Generate some random 5-D data

tczyx = np.random.randint(
    0, np.iinfo(np.uint16).max, size=(5, 2, 3, 32, 32), dtype=np.uint16
)

# %%
# Write Z-stacks with metadata

for t, czyx in enumerate(tczyx):
    for c, zyx in enumerate(czyx):
        writer.write_zstack(
            zyx, writer.root, time_index=t, channel_index=c, auto_meta=True
        )

# %%
# The sections below shows how to add a new channel 
# to an existing store (that we just created).
# It should run without the results from previous steps in RAM.

import zarr
import numpy as np
from iohub.writer import OMEZarrWriter
from iohub.zarrfile import OMEZarrReader

# %%
# Populate a new writer from the reader.

reader = OMEZarrReader("ome.zarr")
writer = OMEZarrWriter.from_reader(reader)
print(writer.channel_names)

# %%
# Add a new channel

writer.append_channel("Phase3D")

new_tczyx = np.random.randint(
    0, np.iinfo(np.uint16).max, size=(5, 1, 3, 32, 32), dtype=np.uint16
)

for t, new_czyx in enumerate(new_tczyx):
    writer.write_zstack(new_czyx[0], writer.root, t, 2)
