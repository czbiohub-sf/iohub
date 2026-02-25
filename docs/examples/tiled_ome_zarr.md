# Tiled OME-Zarr

This example shows how to create a tiled single-resolution OME-Zarr dataset,
make a tiles grid, and write data.

```python
import os
from tempfile import TemporaryDirectory

import numpy as np

from iohub.ngff import open_ome_zarr
```

## Write tiled data

```python
tmp_dir = TemporaryDirectory()
store_path = os.path.join(tmp_dir.name, "tiled.zarr")

# define grid (rows, columns)
grid_shape = (2, 3)
# define tile shape (5D array)
tile_shape = (5, 2, 3, 32, 32)

with open_ome_zarr(
    store_path, layout="tiled", mode="a", channel_names=["DAPI", "GFP"]
) as dataset:
    dtype = np.uint16
    tiles = dataset.make_tiles(
        "tiled_raw",
        grid_shape=grid_shape,
        tile_shape=tile_shape,
        dtype=dtype,
    )
    for row in range(grid_shape[0]):
        for column in range(grid_shape[1]):
            data = np.zeros(shape=tile_shape, dtype=dtype) + row + column
            tiles.write_tile(data, row, column)
```

## Load and inspect

```python
with open_ome_zarr(store_path, layout="tiled", mode="r") as dataset:
    dataset.print_tree()
    tiled = dataset["tiled_raw"]
    print(tiled.tiles, tiled.tile_shape)
    tile_1_2 = tiled.get_tile(1, 2)
```

## Clean up

```python
tmp_dir.cleanup()
```
