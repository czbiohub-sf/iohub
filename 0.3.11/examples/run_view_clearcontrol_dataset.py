"""
Viewing Clear Control
=====================

Example of opening a Clear Control dataset and viewing with napari.

Usage:

    $ python view_clearcontrol_dataset.py <OPTIONAL Clear Control dataset>

If the dataset path is not provided,
it creates a mock dataset of random integers.
"""

# %%
# Setup

import sys
import tempfile
import time

from iohub.clearcontrol import (
    ClearControlFOV,
    create_mock_clear_control_dataset,
)

# %%
# Parse optional Clear Control dataset path.
# Mock dataset is created if dataset path is not provided.

if len(sys.argv) < 2:
    print("Loading mock random noise dataset ...")
    path = f"{tempfile.gettempdir()}/dataset.cc"
    create_mock_clear_control_dataset(path)
else:
    path = sys.argv[1]

# %%
# Open Clear Control dataset.

s = time.time()
cc = ClearControlFOV(path, cache=True)
print("init time (secs)", time.time() - s)

# %%
# Time load time of a single volume.

s = time.time()
cc[0, 0]
print("single volume load time (secs)", time.time() - s)


# %%
# Load dataset using napari
if __name__ == "__main__":
    try:
        import napari

        s = time.time()
        napari.view_image(cc)
        print("napari load time (secs)", time.time() - s)

        napari.run()

    except ModuleNotFoundError:
        pass
