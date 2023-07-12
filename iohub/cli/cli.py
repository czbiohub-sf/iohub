import click

from iohub._version import __version__
from iohub.convert import TIFFConverter
from iohub.reader import print_info

VERSION = __version__

_DATASET_PATH = click.Path(exists=True, file_okay=False, resolve_path=True)


@click.group()
@click.help_option("-h", "--help")
@click.version_option(version=VERSION)
def cli():
    """\u001b[34;1m iohub: N-dimensional bioimaging I/O \u001b[0m"""


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
    help="Show usage guide to open dataset in Python "
    "and full tree for HCS Plates in OME-Zarr",
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
    "--format",
    "-f",
    required=False,
    type=str,
    help="Data type, 'ometiff', 'ndtiff', 'singlepagetiff'",
)
@click.option(
    "--scale-voxels",
    "-s",
    required=False,
    type=bool,
    default=True,
    help="Write voxel size (XY pixel size and Z-step, in micrometers) "
    "as scale coordinate transformation in NGFF. By default true.",
)
@click.option(
    "--grid-layout",
    "-g",
    required=False,
    is_flag=True,
    help="Arrange FOVs in a row/column grid layout for tiled acquisition",
)
@click.option(
    "--label-positions",
    "-p",
    required=False,
    is_flag=True,
    help="Dump postion labels in MM metadata to Omero metadata",
)
def convert(input, output, format, scale_voxels, grid_layout, label_positions):
    """Converts Micro-Manager TIFF datasets to OME-Zarr"""
    converter = TIFFConverter(
        input_dir=input,
        output_dir=output,
        data_type=format,
        scale_voxels=scale_voxels,
        grid_layout=grid_layout,
        label_positions=label_positions,
    )
    converter.run()
