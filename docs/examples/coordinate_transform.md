# Coordinate Transform

This example writes two positions using the high content screening (HCS)
OME-Zarr dataset with two FOV in a single well with different
coordinate transformations (translation and scaling).

```python
import os
from tempfile import TemporaryDirectory

import numpy as np

from iohub.ngff import TransformationMeta, open_ome_zarr
```

## Create sample images and transformations

```python
tmp_dir = TemporaryDirectory()
store_path = os.path.join(tmp_dir.name, "transformed.zarr")

tczyx_1 = np.random.randint(
    0, np.iinfo(np.uint16).max, size=(1, 3, 3, 32, 32), dtype=np.uint16
)
tczyx_2 = np.random.randint(
    0, np.iinfo(np.uint16).max, size=(1, 3, 3, 32, 32), dtype=np.uint16
)

# Coordinate Transformations (T, C, Z, Y, X)
coords_shift = [
    [1.0, 1.0, 1.0, 10.0, 10.0],
    [1.0, 1.0, 0.0, -10.0, -10.0],
]
img_scaling = [[1.0, 1.0, 1.0, 0.5, 0.5]]

translation = [
    TransformationMeta(type="translation", translation=shift)
    for shift in coords_shift
]
scaling = [
    TransformationMeta(type="scale", scale=scale) for scale in img_scaling
]
```

## Write with coordinate transforms

```python
with open_ome_zarr(
    store_path,
    layout="hcs",
    mode="w-",
    channel_names=["DAPI", "GFP", "Brightfield"],
) as dataset:
    position = dataset.create_position("0", "0", "0")
    position.create_image("0", tczyx_1, transform=[translation[0]])
    position = dataset.create_position("0", "0", "1")
    position.create_image("0", tczyx_2, transform=[translation[1], scaling[0]])
    dataset.print_tree()
```

!!! note
    To see the coordinate transforms,
    open the positions individually using napari-ome-zarr.

## Clean up

```python
tmp_dir.cleanup()
```
