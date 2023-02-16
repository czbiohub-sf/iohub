# iohub

N-dimensional bioimaging produces data and metadata in various formats,
and iohub aims to become a unified Python interface to the most common formats
used at the Biohub and in the broader imaging community.

## Supported formats

### Read

- OME-Zarr
- Single-page TIFF, OME-TIFF, and NDTiff images written by Micro-Manager/pycro-manager
- Custom data formats used by Biohub microscopes (e.g., PTI, Mantis (WIP), DaXi (TBD))

### Write

- OME-Zarr
- Multi-page TIFF stacks organized in a directory hierarchy that mimics OME-NGFF (WIP)

## Quick start

Install iohub locally:

```sh
git clone https://github.com/czbiohub/iohub.git
pip install /path/to/iohub
```

> For more details about installation, see the [related section in the contribution guide](CONTRIBUTING.md#setting-up-developing-environment).

Load and modify an [official example OME-Zarr](https://zenodo.org/record/7274533#.Y-q9uOzMJqv) dataset:

```py
import numpy as np
from iohub.ngff import open_ome_zarr

with open_ome_zarr(
    "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
    mode="r",
    layout="auto",
) as dataset:
    dataset.print_tree()  # prints the hierarchy of the zarr store
    channel_names = dataset.channel_names
    print(channel_names)
    img_array = dataset[
        "B/03/0/0"
    ]  # lazy Zarr array for the raw image in the first position
    raw_data = img_array.numpy()  # loads a CZYX 4D array into RAM
    print(raw_data.mean())  # does some analysis

# %%
with open_ome_zarr(
    "max_intensity_projection.zarr",
    mode="w-",
    layout="hcs",
    channel_names=channel_names,
) as dataset:
    new_fov = dataset.create_position(
        "B", "03", "0"
    )  # creates fov with the same path
    new_fov["0"] = raw_data.max(axis=1).reshape(
        (1, 1, 1, *raw_data.shape[2:])
    )  # max projection along Z axis and prepend dims to 5D
    dataset.print_tree()  # checks that new data has been written
```

For more API usage examples, refer to these [example scripts](https://github.com/czbiohub/iohub/tree/main/examples).

## Why iohub?

This project is inspired by the existing Python libraries for bioimaging data I/O,
including [ome-zarr-py](https://github.com/ome/ome-zarr-py), [tifffile](https://github.com/cgohlke/tifffile) and [aicsimageio](https://github.com/AllenCellModeling/aicsimageio).
They support some of the most widely adopted and/or promising formats in microscopy,
such as OME-Zarr and OME-Tiff.

iohub bridges the gaps among them with the following features:

- Efficient reading of data in various TIFF-based formats produced by the Micro-Manager/Pycro-Manager acquisition stack.
- Efficient and customizable conversion of data and metadata from Tiff to OME-Zarr.
- Pythonic and atomic access of OME-Zarr data with parallelized analysis in mind.
- OME-Zarr metadata is automatically constructed and updated for writing,
and verified against the specification when reading.
- Adherence to the latest OME-NGFF specification (v0.4) whenever possible.
