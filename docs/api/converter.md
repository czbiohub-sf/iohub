# Convert to OME-Zarr

Convert Micro-Manager TIFF formats and Nikon ND2 datasets to OME-Zarr.

!!! note
    There is also a CLI command for conversion.
    See the [CLI Reference](../cli.md) or run `iohub convert --help`.

## Supported inputs

- Micro-Manager OME-TIFF
- Micro-Manager NDTiff
- Nikon ND2

## Performance

Pixels are copied by a pool of worker threads (`num_workers`, default 4). Each
worker converts one *frame slab* — a block of whole 2D frames of one FOV, bounded in
T, C and Z.

NDTiff v3 datasets are read directly from the byte offsets in `NDTiff.index`
(one contiguous read per volume, `O_DIRECT` where the filesystem allows it, plane
metadata decoded from the same bytes), so throughput scales with `num_workers`
until the storage saturates; on network filesystems 16 workers is a good setting.
Other formats are read through their reader one slab at a time.

### Scheduling across jobs

Work items are independent and enumerated deterministically, so a scheduler can
split a conversion across processes or cluster jobs:

```python
converter = TIFFConverter(src, dst, version="0.5")
converter.init_store()          # once
n = len(converter.slabs)
# in each job k of K (fresh TIFFConverter per job):
TIFFConverter(src, dst, version="0.5").convert(slice(n * k // K, n * (k + 1) // K))
# once, after every job finished:
TIFFConverter(src, dst, version="0.5").finalize()
```

## Python

::: iohub.convert.TIFFConverter
    options:
      heading_level: 3
