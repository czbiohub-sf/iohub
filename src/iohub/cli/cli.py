import pathlib

import click

from iohub import __version__, open_ome_zarr
from iohub.cli.parsing import input_position_dirpaths
from iohub.convert import TIFFConverter
from iohub.core.ozx import pack_ozx, summarize_ozx
from iohub.reader import print_info
from iohub.rename_wells import rename_wells

VERSION = __version__

_DATASET_PATH = click.Path(exists=True, file_okay=False, resolve_path=True, path_type=pathlib.Path)
# Like _DATASET_PATH but also accepts files (e.g. ``.ozx`` archives).
_OME_ZARR_PATH = click.Path(exists=True, resolve_path=True, path_type=pathlib.Path)


@click.group()
@click.help_option("-h", "--help")
@click.version_option(version=VERSION)
def cli():
    """iohub: N-dimensional bioimaging I/O"""


@cli.command()
@click.help_option("-h", "--help")
@click.argument(
    "files",
    nargs=-1,
    required=True,
    type=_OME_ZARR_PATH,
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show usage guide to open dataset in Python and full tree for HCS Plates in OME-Zarr",
)
def info(files, verbose):
    """View metadata for one or more FILES.

    Supports Micro-Manager TIFF datasets (multi-page OME-TIFF, NDTIFF),
    OME-Zarr directory stores (v0.4 and v0.5), and RFC-9 zipped
    OME-Zarr archives (``.ozx``).
    """
    for file in files:
        click.echo(f"Reading file:\t {file}")
        print_info(file, verbose=verbose)


@cli.command()
@click.help_option("-h", "--help")
@click.option(
    "--input",
    "-i",
    required=True,
    type=_DATASET_PATH,
    help="Input Micro-Manager TIFF dataset directory",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(exists=False, resolve_path=True),
    help="Output zarr store (/**/converted.zarr)",
)
@click.option(
    "--grid-layout",
    "-g",
    required=False,
    is_flag=True,
    help="Arrange FOVs in a row/column grid layout for tiled acquisition",
)
@click.option(
    "--chunks",
    "-c",
    required=False,
    default="XYZ",
    help="Zarr chunk size given as 'XY', 'XYZ', or a tuple of chunk "
    "dimensions. If 'XYZ', chunk size will be limited to 500 MB.",
)
@click.option(
    "--ome-zarr-version",
    "-v",
    required=False,
    default="0.4",
    type=click.Choice(["0.4", "0.5"]),
    help="OME-NGFF version for the output Zarr store. '0.4' uses Zarr v2 format; '0.5' uses Zarr v3 format.",
)
def convert(input, output, grid_layout, chunks, ome_zarr_version):
    """Converts Micro-Manager TIFF datasets to OME-Zarr"""
    converter = TIFFConverter(
        input_dir=input,
        output_dir=output,
        grid_layout=grid_layout,
        chunks=chunks,
        version=ome_zarr_version,
    )
    converter()


@cli.command()
@click.help_option("-h", "--help")
@input_position_dirpaths()
@click.option(
    "--t-scale",
    "-t",
    required=False,
    type=float,
    help="New t scale",
)
@click.option(
    "--z-scale",
    "-z",
    required=False,
    type=float,
    help="New z scale",
)
@click.option(
    "--y-scale",
    "-y",
    required=False,
    type=float,
    help="New y scale",
)
@click.option(
    "--x-scale",
    "-x",
    required=False,
    type=float,
    help="New x scale",
)
@click.option(
    "--image",
    required=False,
    help="Image name to set scale for. Default is '0'",
)
def set_scale(
    input_position_dirpaths,
    t_scale=None,
    z_scale=None,
    y_scale=None,
    x_scale=None,
    image=None,
):
    """Update scale metadata in OME-Zarr datasets.

    ```
    iohub set-scale -i input.zarr/*/*/* -t 1.0 -z 1.0 -y 0.5 -x 0.5
    ```

    Supports setting a single axis at a time:

    ```
    iohub set-scale -i input.zarr/*/*/* -z 2.0
    ```
    """
    if image is None:
        image = "0"
    for input_position_dirpath in input_position_dirpaths:
        with open_ome_zarr(input_position_dirpath, layout="fov", mode="r+") as dataset:
            for name, value in zip(["t", "z", "y", "x"], [t_scale, z_scale, y_scale, x_scale], strict=False):
                if value is None:
                    continue
                dataset.set_scale(image, name, value)


@cli.command(name="rename-wells")
@click.help_option("-h", "--help")
@click.option(
    "-i",
    "--input",
    "zarrfile",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    required=True,
    help="Path to the input Zarr file.",
)
@click.option(
    "-c",
    "--csv",
    "csvfile",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to the CSV file containing old and new well names.",
)
def rename_wells_command(zarrfile, csvfile):
    """Rename wells in an plate.

    ```
    iohub rename-wells -i plate.zarr -c names.csv
    ```

    The CSV file must have two columns with old and new names in the form:
    ```
    A / 1, B / 2
    A / 2, B / 2
    ```
    """
    rename_wells(zarrfile, csvfile)


def _echo_ozx_summary(verb: str, path: pathlib.Path) -> None:
    """Print the one-liner ``ozx pack`` shows on success."""
    s = summarize_ozx(path)
    click.echo(f"{verb}: {path}  (ome version={s.version}, jsonFirst={s.json_first})")


@cli.group(name="ozx")
@click.help_option("-h", "--help")
def ozx_group():
    """RFC-9 zipped OME-Zarr (``.ozx``) operations.

    A ``.ozx`` is a single-file OME-Zarr archive (ZIP_STORED + ZIP64)
    designed for distribution - S3, AWS Open Data, HPC↔compute-cluster
    transfers - not as a live mutable store. One file to upload,
    download, and serve.

    Use ``ozx pack`` to bundle a directory store for export and
    ``ozx info`` to inspect an archive's RFC-9 properties. ``iohub
    info`` works on ``.ozx`` paths directly for the usual FOV summary.
    """


@ozx_group.command(name="pack")
@click.help_option("-h", "--help")
@click.option(
    "-i",
    "--input",
    "src",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True, path_type=pathlib.Path),
    help="Input OME-Zarr directory store.",
)
@click.option(
    "-o",
    "--output",
    "dst",
    required=True,
    type=click.Path(exists=False, resolve_path=True, path_type=pathlib.Path),
    help="Output .ozx path (must not exist).",
)
@click.option(
    "--version",
    "ome_version",
    default=None,
    help="OME-NGFF version to record. Sniffed from source if omitted.",
)
def ozx_pack(src, dst, ome_version):
    """Pack an OME-Zarr directory into an RFC-9 ``.ozx`` archive.

    Writes a ZIP_STORED + ZIP64 archive in one pass with entries in BFS
    order - root ``zarr.json`` first, other ``zarr.json`` next, chunks
    last - and ``jsonFirst:true`` in the archive comment. HTTP-range
    readers can then fetch all metadata in one contiguous Range request
    on cold opens, the win that matters for S3 and AWS Open Data.
    """
    out = pack_ozx(src, dst, version=ome_version)
    _echo_ozx_summary("packed", out)


@ozx_group.command(name="info")
@click.help_option("-h", "--help")
@click.argument(
    "path",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True, path_type=pathlib.Path),
)
def ozx_info(path):
    """Print RFC-9 archive metadata: version, ``jsonFirst``, entry count, size.

    Use it to sanity-check an archive before publishing (``jsonFirst``
    should be ``True`` after ``iohub ozx pack``) or to inspect archives
    received from collaborators or downloaded from public datasets.
    """
    import zipfile

    # One zip open for the namelist, one for the comment.
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    summary = summarize_ozx(path)
    n_entries = len(names)
    n_meta = sum(1 for n in names if n.rsplit("/", 1)[-1] == "zarr.json")
    n_dupes = n_entries - len(set(names))
    click.echo(f"path:        {path}")
    click.echo(f"size:        {path.stat().st_size:,} bytes")
    click.echo(f"entries:     {n_entries:,}  ({n_meta} zarr.json, {n_dupes} duplicate names)")
    click.echo(f"ome version: {summary.version}")
    click.echo(f"jsonFirst:   {summary.json_first}")
