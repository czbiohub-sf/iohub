"""
Tile + Blend an HCS Well (Multi-FOV)
=====================================

Create a synthetic HCS plate with 1 well and 4 FOVs arranged in a 2x2
grid with physical overlap, then composite the FOVs into a single mosaic
and tile+blend with ``map_tiles``.

This demonstrates the full pipeline:
FOV compositing (``Well.to_xarray``) → tiling (``Slicer``) → blending (``map_tiles``).
"""

# %%
import os
import warnings
from tempfile import TemporaryDirectory

import numpy as np

from iohub.ngff import open_ome_zarr
from iohub.ngff.models import TransformationMeta
from iohub.tile import Slicer, map_tiles

warnings.filterwarnings("ignore")

# %%
# Create a synthetic HCS plate
# ------------------------------
# 1 well ("A/1") with 4 FOVs in a 2x2 grid.
# Each FOV is 1t x 1c x 1z x 32y x 32x.
# FOVs overlap by 8 pixels (~25%) in both Y and X.
#
# Layout (pixel coordinates):
#
# .. code-block:: text
#
#     FOV 0: y=[0,32), x=[0,32)       FOV 1: y=[0,32), x=[24,56)
#     FOV 2: y=[24,56), x=[0,32)      FOV 3: y=[24,56), x=[24,56)
#
#     Mosaic: 56 x 56 pixels (with 8px overlap strips between FOVs)

tmp_dir = TemporaryDirectory()
plate_path = os.path.join(tmp_dir.name, "plate.zarr")

rng = np.random.default_rng(123)

# Pixel size (um/px) — use 1.0 for clean coordinate alignment
pixel_size = 1.0

# Grid step: 24 px → 8 px overlap per FOV pair (32 - 24 = 8)
grid_step = 24

# FOV grid positions: (row_idx, col_idx) → pixel origin (y, x)
fov_grid = {
    "000": (0, 0),
    "001": (0, 1),
    "010": (1, 0),
    "011": (1, 1),
}

with open_ome_zarr(plate_path, layout="hcs", mode="w-", channel_names=["GFP"]) as plate:
    for fov_name, (row_idx, col_idx) in fov_grid.items():
        pos = plate.create_position("A", "1", fov_name)
        data = rng.random((1, 1, 1, 32, 32), dtype=np.float32)
        pos.create_image("0", data, chunks=(1, 1, 1, 32, 32))

        # Set physical scale and translation so FOVs are placed on a grid
        y_offset = row_idx * grid_step * pixel_size
        x_offset = col_idx * grid_step * pixel_size
        pos.set_transform(
            "0",
            [
                TransformationMeta(
                    type="scale",
                    scale=[1.0, 1.0, 1.0, pixel_size, pixel_size],
                ),
                TransformationMeta(
                    type="translation",
                    translation=[0.0, 0.0, 0.0, y_offset, x_offset],
                ),
            ],
        )

print(f"Created plate at {plate_path}")

# %%
# Open and composite the well
# -----------------------------
# ``Well.to_xarray()`` composites all 4 FOVs into one mosaic.

plate = open_ome_zarr(plate_path, mode="r")
_, well = next(plate.wells())
mosaic = well.to_xarray(compositor="mean")

print(f"Mosaic shape: {mosaic.shape}")
print(f"Mosaic Y range: [{float(mosaic.y[0]):.2f}, {float(mosaic.y[-1]):.2f}] um")
print(f"Mosaic X range: [{float(mosaic.x[0]):.2f}, {float(mosaic.x[-1]):.2f}] um")

# %%
# Inspect the tiling
# --------------------

slicer = Slicer(mosaic, tile_size={"y": 24, "x": 24}, overlap={"y": 4, "x": 4})
print(f"\n{slicer}")
print(f"Tiles: {len(slicer)}")
print(f"Overlap edges: {slicer.graph.number_of_edges()}")

# %%
# Tile, process, and blend
# --------------------------
# Apply a function to each tile of the mosaic and blend back.


def process(tile):
    """Example: double the intensity."""
    return tile * 2


result = map_tiles(
    mosaic,
    fn=process,
    tile_size={"y": 24, "x": 24},
    overlap={"y": 4, "x": 4},
    weights="gaussian",
)
print(f"\nResult shape: {result.shape}")
print(f"Lazy: {hasattr(result.data, 'dask')}")

# %%
# Verify the result
# -------------------

values = result.values
expected = mosaic.values * 2
np.testing.assert_allclose(values, expected, atol=1e-4)
print("Round-trip check: PASSED")

# %%
# With overlap caching
# ----------------------

result_cached = map_tiles(
    mosaic,
    fn=process,
    tile_size={"y": 24, "x": 24},
    overlap={"y": 4, "x": 4},
    weights="gaussian",
    cache="persist",
)
np.testing.assert_allclose(result_cached.values, expected, atol=1e-4)
print("Cached round-trip: PASSED")

# %%
# Clean up

plate.close()
tmp_dir.cleanup()
