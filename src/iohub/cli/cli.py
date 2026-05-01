import logging
import pathlib

import click

from iohub import __version__, open_ome_zarr
from iohub.cli.parsing import input_position_dirpaths
from iohub.convert import TIFFConverter
from iohub.reader import print_info
from iohub.rename_wells import rename_wells

_logger = logging.getLogger(__name__)

VERSION = __version__

_DATASET_PATH = click.Path(exists=True, file_okay=False, resolve_path=True, path_type=pathlib.Path)


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
    type=_DATASET_PATH,
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show usage guide to open dataset in Python and full tree for HCS Plates in OME-Zarr",
)
def info(files, verbose):
    """View basic metadata of a list of FILES.

    Supported formats are Micro-Manager-acquired TIFF datasets
    (single-page TIFF, multi-page OME-TIFF, NDTIFF)
    and OME-Zarr (v0.1 linear HCS layout and all v0.4 layouts).
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


_PYRAMID_METHODS = ["mean", "median", "mode", "min", "max", "stride"]
_PYRAMID_DIM_CHOICES = ["t", "z", "y", "x"]


def _parse_dims(ctx, param, value):
    if value is None:
        return None
    tokens = [t.strip().lower() for t in value.split(",") if t.strip()]
    if not tokens:
        return None
    invalid = [t for t in tokens if t not in _PYRAMID_DIM_CHOICES]
    if invalid:
        raise click.BadParameter(f"Unknown dim(s) {invalid}. Valid choices: {_PYRAMID_DIM_CHOICES}.")
    return set(tokens)


@cli.command(name="compute-pyramid")
@click.help_option("-h", "--help")
@input_position_dirpaths()
@click.option(
    "--levels",
    "-l",
    required=True,
    type=click.IntRange(min=2),
    help="Total number of pyramid levels including level 0 (e.g. 4 = level 0 + 3 extra).",
)
@click.option(
    "--method",
    "-m",
    required=False,
    default="mean",
    show_default=True,
    type=click.Choice(_PYRAMID_METHODS),
    help="Downsampling method.",
)
@click.option(
    "--dims",
    "-d",
    required=False,
    default=None,
    callback=_parse_dims,
    help=("Comma-separated axes to downsample (e.g. 'y,x' for YX-only). Defaults to 'z,y,x' inside iohub."),
)
def compute_pyramid(input_position_dirpaths, levels, method, dims):
    """Compute multiscale pyramid levels in place for OME-Zarr positions.

    The level-0 array is preserved; new downsampled levels are appended.

    ```
    iohub compute-pyramid -i input.zarr/*/*/* --levels 4
    iohub compute-pyramid -i input.zarr/*/*/* -l 3 -m median --dims y,x
    ```
    """
    for input_position_dirpath in input_position_dirpaths:
        _logger.info(f"Computing pyramid for {input_position_dirpath}")
        with open_ome_zarr(input_position_dirpath, layout="fov", mode="r+") as dataset:
            dataset.compute_pyramid(levels=levels, method=method, dims=dims)


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
