# CLI Reference

## Selecting input positions

Commands that take `-i` / `--input-position-dirpaths` (`set-scale`,
`compute-pyramid`) accept a single position, a plate root (expanded into all of
its positions), or a shell glob. A single `-i` can take several space-separated
paths:

```bash
iohub compute-pyramid -i input.zarr/*/*/* --levels 4       # every position, via glob
iohub compute-pyramid -i input.zarr --levels 4             # whole plate (same result)
iohub set-scale -i input.zarr/A/1/0 input.zarr/B/2/0 -z 2  # specific positions
```

::: mkdocs-typer2
    :module: iohub.cli.cli
    :name: iohub
    :pretty: true
