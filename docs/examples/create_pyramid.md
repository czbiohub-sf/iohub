# Multiscale Pyramid Creation

Create and compute a multiscale pyramid for efficient visualization.

```python
import os
from tempfile import TemporaryDirectory

import numpy as np

from iohub import open_ome_zarr
```

## Create data and compute pyramid

```python
tmp_dir = TemporaryDirectory()
store_path = os.path.join(tmp_dir.name, "pyramid.zarr")

data = np.random.randint(0, 255, size=(1, 2, 32, 256, 256), dtype=np.uint16)
print(f"Original data shape: {data.shape}")

with open_ome_zarr(
    store_path, layout="fov", mode="a", channel_names=["DAPI", "GFP"]
) as position:
    position.create_image("0", data)
    position.compute_pyramid(levels=3, method="mean")

    print("Pyramid levels:")
    for level in range(3):
        level_array = position[str(level)]
        scale = position.get_effective_scale(str(level))
        print(f"  Level {level}: {level_array.shape}, scale={scale[-3:]}")
```

## Label images get the same levels

A label image under `labels/` is a multiscale group too, and the viewer
expects it to have the image's levels. `PositionLabel.compute_pyramid` takes
the same arguments; pick a method that never invents label ids (`max`,
`stride` or `mode`), not `mean`.

```python
with open_ome_zarr(store_path, mode="r+") as position:
    labels = np.zeros((1, 32, 256, 256), dtype=np.uint32)
    labels[0, :, 64:128, 64:128] = 1
    label_image = position.create_label("cells", labels)
    label_image.compute_pyramid(levels=3, method="max")
    for level in range(3):
        print(f"  cells level {level}: {label_image[str(level)].shape}")
```

The `iohub compute-pyramid` command does both: `--method` for the image and
`--label-method` (default `max`) for every label image of the position.

## Clean up

```python
tmp_dir.cleanup()
```
