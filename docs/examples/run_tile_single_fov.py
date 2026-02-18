"""
Tile + Blend a Single FOV
==========================

Create a synthetic OME-Zarr FOV, then tile it with overlap,
apply a processing function to each tile, and blend the results
back into a single mosaic using ``map_tiles`` (xarray-native)
and ``tile_and_assemble`` (zarr output).
"""

# %%
import os
import warnings
from tempfile import TemporaryDirectory

import numpy as np

from iohub.ngff import open_ome_zarr
from iohub.tile import Slicer, map_tiles, tile_and_assemble

warnings.filterwarnings("ignore")

# %%
# Create a synthetic single-FOV OME-Zarr
# ----------------------------------------
# 1 timepoint, 2 channels, 4 Z-slices, 64x128 YX.

tmp_dir = TemporaryDirectory()
fov_path = os.path.join(tmp_dir.name, "fov.zarr")

rng = np.random.default_rng(42)
raw = rng.random((1, 2, 4, 64, 128), dtype=np.float32)

with open_ome_zarr(fov_path, layout="fov", mode="w-", channel_names=["GFP", "DAPI"]) as dataset:
    dataset.create_image("0", raw, chunks=(1, 1, 4, 64, 128))
    dataset.set_scale("0", "y", 0.325)
    dataset.set_scale("0", "x", 0.325)

print(f"Created FOV at {fov_path}")

# %%
# Open and inspect the data
# --------------------------

pos = open_ome_zarr(fov_path, mode="r")
data = pos.to_xarray()
print(f"Shape: {data.shape}  dims: {data.dims}")
print(f"Y range: [{float(data.y[0]):.2f}, {float(data.y[-1]):.2f}] um")
print(f"X range: [{float(data.x[0]):.2f}, {float(data.x[-1]):.2f}] um")

# %%
# Inspect the Slicer
# --------------------
# See how tiles are laid out with overlap.

slicer = Slicer(data, tile_size={"y": 32, "x": 64}, overlap={"y": 8, "x": 16})
print(slicer)
print(f"Neighborhood graph: {slicer.graph.number_of_edges()} overlap edges")

# %%
# map_tiles: xarray-native (no zarr output)
# -------------------------------------------
# Tile, apply a function, blend back. Result stays lazy until ``.values``.


def my_algorithm(tile):
    """Example: scale by 2 and add 1."""
    return tile * 2 + 1


result = map_tiles(
    data,
    fn=my_algorithm,
    tile_size={"y": 32, "x": 64},
    overlap={"y": 8, "x": 16},
    weights="gaussian",
)
print(f"Result shape: {result.shape}, lazy: {hasattr(result.data, 'dask')}")
print(f"Coords preserved: c={list(result.c.values)}")

# Trigger computation and verify
values = result.values
expected = raw * 2 + 1
np.testing.assert_allclose(values, expected, atol=1e-4)
print("Round-trip check: PASSED")

# %%
# map_tiles with overlap caching
# --------------------------------
# ``cache="persist"`` pre-loads overlap strips so they aren't read twice.
# ``cache="bfs"`` reorders tile processing for cache locality.

result_cached = map_tiles(
    data,
    fn=my_algorithm,
    tile_size={"y": 32, "x": 64},
    overlap={"y": 8, "x": 16},
    weights="gaussian",
    cache="persist",
)
np.testing.assert_allclose(result_cached.values, expected, atol=1e-4)
print("Cached round-trip: PASSED")

# %%
# tile_and_assemble: zarr output
# --------------------------------
# Same pipeline, but writes to zarr on disk.

out_path = os.path.join(tmp_dir.name, "result.zarr")
result_zarr = tile_and_assemble(
    data,
    fn=my_algorithm,
    tile_size={"y": 32, "x": 64},
    output=out_path,
    overlap={"y": 8, "x": 16},
    weights="gaussian",
)
print(f"Output zarr: {out_path}")
np.testing.assert_allclose(result_zarr.values, expected, atol=1e-4)
print("Zarr round-trip: PASSED")

# %%
# Identity round-trip with different blenders
# -----------------------------------------------
# Verify that blending is correct: ``fn=identity`` recovers the original.

for blender in ["uniform", "gaussian", "distance"]:
    r = map_tiles(
        data,
        fn=lambda t: t,
        tile_size={"y": 32, "x": 64},
        overlap={"y": 8, "x": 16},
        weights=blender,
    )
    maxerr = float(np.max(np.abs(r.values - raw)))
    print(f"  {blender:10s} identity max error: {maxerr:.2e}")

# %%
# Clean up

pos.close()
tmp_dir.cleanup()
