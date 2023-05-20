import os

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
    "datasets",
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
def info(datasets, verbose):
    """View basic metadata of a list of FILES.

    Supported formats are Micro-Manager-acquired TIFF datasets
    (single-page TIFF, multi-page OME-TIFF, NDTIFF)
    and OME-Zarr (v0.1 linear HCS layout and all v0.4 layouts).
    """
    for dataset in datasets:
        click.echo(f"Reading dataset:\t {dataset}")
        print_info(dataset, verbose=verbose)


@cli.command()
@click.help_option("-h", "--help")
@click.argument(
    "input_datasets",
    nargs=-1,
    required=True,
    type=_DATASET_PATH,
    # help="Input Micro-Manager TIFF dataset directory",
)
@click.option(
    "--output",
    "-o",
    required=False,
    type=click.Path(exists=False, resolve_path=True),
    default="./",
    help="""Path to output. Defaults to the current directory with the input 
    dataset's name.""",
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
    help="Arrange positions in a HCS grid layout",
)
@click.option(
    "--label-positions",
    "-p",
    required=False,
    is_flag=True,
    help="Dump postion labels in MM metadata to Omero metadata",
)
def convert(
    input_datasets, output, format, scale_voxels, grid_layout, label_positions
):
    """Converts Micro-Manager TIFF datasets to OME-Zarr

    Example:

    >> iohub convert /glob/of/ome-tiff/folders/*

    will convert all ome-tiff folders to .zarr datasets in the current
    directory.
    """

    for input_dataset in input_datasets:
        click.echo(f"Converting dataset:\t {input_dataset}")

        output_dir = os.path.join(
            output, os.path.basename(input_dataset) + ".zarr"
        )
        converter = TIFFConverter(
            input_dir=input_dataset,
            output_dir=output_dir,
            data_type=format,
            scale_voxels=scale_voxels,
            grid_layout=grid_layout,
            label_positions=label_positions,
        )
        converter.run()
