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

## Clean up

```python
tmp_dir.cleanup()
```
