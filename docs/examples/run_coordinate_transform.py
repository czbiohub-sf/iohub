"""
Coordinate Transform
====================

This script writes two positions using the high content screening (HCS)
OME-Zarr dataset with two FOV in a single well with different
coordinate transformations (i.e translation and scaling)
"""

# %%
import os
from tempfile import TemporaryDirectory

import numpy as np

from iohub.ngff import TransformationMeta, open_ome_zarr

# %%
# Set storage path

tmp_dir = TemporaryDirectory()
store_path = os.path.join(tmp_dir.name, "transformed.zarr")
print("Zarr store path", store_path)

# %%
# Create two random sample images
tczyx_1 = np.random.randint(
    0, np.iinfo(np.uint16).max, size=(1, 3, 3, 32, 32), dtype=np.uint16
)
tczyx_2 = np.random.randint(
    0, np.iinfo(np.uint16).max, size=(1, 3, 3, 32, 32), dtype=np.uint16
)

# %%
# Coordinate Transformations (T,C,Z,Y,X)
# By default the translation is the identity matrix
coords_shift = [[1.0, 1.0, 1.0, 10.0, 10.0], [1.0, 1.0, 0.0, -10.0, -10.0]]
img_scaling = [[1.0, 1.0, 1.0, 0.5, 0.5]]

# %%
# Generate Transformation Metadata
translation = []
for shift in coords_shift:
    translation.append(
        TransformationMeta(type="translation", translation=shift)
    )
scaling = []
for scale in img_scaling:
    scaling.append(TransformationMeta(type="scale", scale=scale))

# %%
# Write 5D data to a new Zarr store
with open_ome_zarr(
    store_path,
    layout="hcs",
    mode="w-",
    channel_names=["DAPI", "GFP", "Brightfield"],
) as dataset:
    # Create and write to positions
    # This affects the tile arrangement in visualization
    position = dataset.create_position(0, 0, 0)
    position.create_image("0", tczyx_1, transform=[translation[0]])
    position = dataset.create_position(0, 0, 1)
    position.create_image("0", tczyx_2, transform=[translation[1], scaling[0]])
    # Print dataset summary
    dataset.print_tree()

# %%
# .. note:: To see the coordinate transforms,
#     open the positions individually using napari-ome-zarr.
#     This will duplicate the layers (channels).

# %%
# Clean up
tmp_dir.cleanup()