"""
Add Labels
==========

This script shows how to add labels to a single-FOV OME-Zarr dataset.
Demonstrates creating segmentation masks and annotations with NGFF-compliant
multiscale pyramids, colors, and properties.
"""

from pathlib import Path

# %%
import numpy as np

from iohub.ngff.nodes import open_ome_zarr

# Path for our example dataset
dataset_path = Path("data/example_with_labels.zarr")

image_shape = (1, 2, 1, 1024, 1024)
rng = np.random.default_rng(42)

print("Adding Labels to OME-Zarr FOV Dataset")
print("=" * 40)

# %% Create OME-Zarr dataset with sample image
print("1. Creating dataset with sample image...")
with open_ome_zarr(
    dataset_path,
    layout="fov",
    mode="w",
    channel_names=["DAPI", "GFP"],
    version="0.5",
) as position:
    # TCZYX: 1 time, 2 channels, 1 z-slice, 1024x1024
    sample_image = rng.integers(0, 1000, size=image_shape, dtype=np.uint16)
    position.create_image("0", sample_image)
    print(f"Created image: {sample_image.shape}")

# %% Create cell segmentation labels
with open_ome_zarr(dataset_path, mode="r+") as position:
    print("2. Adding cell segmentation labels...")

    # Create simple segmentation mask (TZYX: 1 time, 1 z-slice, 1024x1024)
    # Note: Labels use TZYX format (no channel dimension)
    segmentation = np.zeros((1, 1, 1024, 1024), dtype=np.uint16)

    # Add 3 simple circular cells
    y, x = np.ogrid[:1024, :1024]
    cells = [
        (250, 250, 100, 1),  # (center_y, center_x, radius, label_value)
        (500, 750, 80, 2),
        (750, 400, 120, 3),
    ]

    for center_y, center_x, radius, label_value in cells:
        mask = ((y - center_y) ** 2 + (x - center_x) ** 2) <= radius**2
        segmentation[0, 0, mask] = label_value

    # Define colors for visualization (RGBA integers 0-255)
    colors = {
        1: [255, 100, 100, 255],  # Light red
        2: [100, 255, 100, 255],  # Light green
        3: [100, 100, 255, 255],  # Light blue
    }

    # Define properties for each cell
    properties = [
        {"label-value": 1, "type": "normal", "area": 31416},
        {"label-value": 2, "type": "small", "area": 20106},
        {"label-value": 3, "type": "large", "area": 45239},
    ]

    # Create multiscale label image (2 pyramid levels)
    cells_label = position.create_label(
        name="cells",
        data=segmentation,
        colors=colors,
        properties=properties,
        pyramid_levels=2,  # Creates levels "0" and "1"
    )
    print(f"Created cells label: {cells_label.array_keys()} levels")

# %% Create annotation labels
with open_ome_zarr(dataset_path, layout="fov", mode="r+") as position:
    print("3. Adding annotation labels...")

    # Create annotation points (single level)
    annotations = np.zeros((1, 1, 1024, 1024), dtype=np.uint8)
    annotations[0, 0, 200, 200] = 1  # Point of interest
    annotations[0, 0, 800, 800] = 2  # Another point

    annotations_label = position.create_label(
        name="annotations",
        data=annotations,
        colors={1: [255, 255, 0, 255], 2: [255, 0, 255, 255]},
        pyramid_levels=1,  # Single resolution
    )
    print(f"Created annotations: {annotations_label.array_keys()} levels")

# 4. Read back and demonstrate access patterns
print("4. Reading labels back...")
with open_ome_zarr(dataset_path, layout="fov", mode="r") as position:
    available_labels = (
        list(position.labels_group.group_keys())
        if position.labels_group
        else []
    )
    print(f"   Available labels: {available_labels}")

    # Access specific labels
    cells = position.get_label("cells")
    print(f"   Cells pyramid: {cells.array_keys()}")

    # Get highest resolution data
    high_res_cells = cells.data  # Gets level "0"
    print(f"   Shape: {high_res_cells.shape}")

    # Access downscaled version (empty, like images)
    if "1" in cells:
        downscaled = cells["1"]
        print(f"   Downscaled shape: {downscaled.shape}")

        # Note: Downscaled levels are empty (same behavior as images)
        # Users can manually populate them if needed
        downscaled_data = downscaled.numpy()
        print(f"   Downscaled is empty: {np.all(downscaled_data == 0)}")
        print(
            "   Note: Fill manually if needed:",
            "   cells['1'][...] = your_downscaled_data",
        )

    # Iterate over all labels
    print("   All labels:")
    for name, label_image in position.labels():
        levels = label_image.array_keys()
        print(f"     {name}: {levels}")
