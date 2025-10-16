"""
Update OME-Zarr Version
=======================

This script shows how to write the same OME-Zarr image
using a new version.
"""

# %%
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from iohub.ngff import TransformationMeta, open_ome_zarr

# %%
# Set storage path
tmp_dir = TemporaryDirectory()
old_store_path = Path(tmp_dir.name) / "old.zarr"
new_store_path = Path(tmp_dir.name) / "new.zarr"

# %%
# Create a version 0.4 OME-Zarr dataset
random_image = np.random.randint(
    0, np.iinfo(np.uint16).max, size=(10, 2, 32, 128, 128), dtype=np.uint16
)
scale = [2.0, 3.0, 4.0, 5.0, 6.0]


with open_ome_zarr(
    old_store_path,
    layout="hcs",
    mode="w-",
    channel_names=["DAPI", "GFP"],
    version="0.4",
) as old_dataset:
    position = old_dataset.create_position("A", "1", "0")
    image = position.create_image(
        "0",
        random_image,
        chunks=(1, 1, 4, 32, 32),
        transform=[TransformationMeta(type="scale", scale=scale)],
    )

# %%
# Write the same image with version 0.5 and sharding

with open_ome_zarr(old_store_path, mode="r", layout="hcs") as old_dataset:
    with open_ome_zarr(
        new_store_path,
        layout="hcs",
        mode="w",
        channel_names=old_dataset.channel_names,
        version="0.5",
    ) as new_dataset:
        for name, old_position in old_dataset.positions():
            row, col, fov = name.split("/")
            new_position = new_dataset.create_position(row, col, fov)
            old_image = old_position["0"]
            new_image = new_position.create_image(
                "0",
                data=old_image.numpy(),
                chunks=(1, 1, 4, 32, 32),
                shards_ratio=(2, 1, 8, 4, 4),
                transform=old_position.metadata.multiscales[0]
                .datasets[0]
                .coordinate_transformations,
            )

# %%
# Read the new FOV to verify it was written correctly
with open_ome_zarr(new_store_path / "A/1/0", mode="r") as dataset:
    assert dataset.scale == scale
    image = dataset["0"]
    assert image.shards == (2, 1, 32, 128, 128)
    assert np.array_equal(image.numpy(), random_image)

# %%
# Clean up
tmp_dir.cleanup()
