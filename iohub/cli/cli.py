import sys

import click

from iohub._version import __version__
from iohub import WaveorderReader
from iohub.zarr_converter import ZarrConverter

VERSION = __version__


@click.group()
@click.version_option(version=VERSION)
def cli():
    print("\033[92miohub: N-dimensional bioimaging I/O \033[0m\n")


@cli.command()
@click.help_option("-h", "--help")
@click.argument("files", nargs=-1)
def info(files):
    """View basic metadata from a list of FILES"""
    for file in files:
        print(f"Reading file:\t {file}")
        reader = WaveorderReader(file)
        print_reader_info(reader)


def print_reader_info(reader):
    print(f"Positions:\t {reader.get_num_positions()}")
    print(f"Time points:\t {reader.shape[0]}")
    print(f"Channels:\t {reader.shape[1]}")
    print(f"(Z, Y, X):\t {reader.shape[2:]}")
    print(f"Channel names:\t {reader.channel_names}")
    print(f"Z step (um):\t {reader.z_step_size}")
    print("")


@cli.command()
@click.help_option("-h", "--help")
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="path to the raw data folder containing ome.tifs",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=str,
    help="full path to save the zarr store (../../Experiment.zarr)",
)
@click.option(
    "--data_type",
    required=False,
    type=str,
    help='Data type, "ometiff", "upti", "zarr"',
)
@click.option(
    "--replace_pos_name",
    required=False,
    type=bool,
    help="whether or not to append position name to data",
)
@click.option(
    "--format_hcs",
    required=False,
    type=bool,
    help='whether or not to format the data as an HCS "well-plate"',
)
def convert2zarr(input, output, data_type, replace_pos_name, format_hcs):
    """Convert MicroManager ome-tiff to ome-zarr

    Example:

    >> iohub convert -i ./ome-tiff/folder/ -o ./output.zarr
    """
    converter = ZarrConverter(
        input, output, data_type, replace_pos_name, format_hcs
    )
    converter.run_conversion()


if __name__ == "__main__":
    cli()
