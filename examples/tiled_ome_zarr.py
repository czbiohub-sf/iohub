# %%
# This script shows how to create a tiled single-resolution OME-Zarr dataset,
# make a tiles grid, and write data.

import os

import numpy as np

from iohub.ngff import open_ome_zarr

# %%
# Set storage path

store_path = f'{os.path.expanduser("~/")}ome.zarr'

# %%
# Write 5D data to a new Zarr store

# define grid (rows, columns)
grid_shape = (2, 3)
# define tile shape (5D array)
tile_shape = (5, 2, 3, 32, 32)


with open_ome_zarr(
    store_path, layout="tiled", mode="a", channel_names=["DAPI", "GFP"]
) as dataset:
    tiles = dataset.make_tiles(
        "tiled_raw", grid_shape=grid_shape, tile_shape=tile_shape
    )
    for row in range(grid_shape[0]):
        for column in range(grid_shape[1]):
            # each tile will be filled with different constant values
            data = np.zeros(shape=tile_shape) + row + column
            tiles.write_tile(data, row, column)


# %%
# Load the dataset

with open_ome_zarr(store_path, layout="tiled", mode="r") as dataset:
    # data store structure
    dataset.print_tree()
    # get tiled image array
    tiled = dataset["tiled_raw"]
    # check grid and tile shapes
    print(tiled.tiles, tiled.tile_shape)
    # read a tile
    tile_1_2 = tiled.get_tile(1, 2)

# %%
# Try viewing the images with napari-ome-zarr
