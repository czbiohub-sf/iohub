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
pip install iohub
```

> For more details about installation, see the [related section in the contribution guide](CONTRIBUTING.md#setting-up-developing-environment).

See usage, see [examples](https://github.com/czbiohub/iohub/tree/main/examples).
