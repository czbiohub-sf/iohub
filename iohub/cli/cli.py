import click

from iohub._version import __version__
from iohub.reader import imread
from iohub.convert import TIFFConverter

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
        reader = imread(file)
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
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
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
def convert(input, output, format, grid_layout, label_positions):
    """Converts Micro-Manager TIFF datasets to OME-Zarr"""
    converter = TIFFConverter(
        input_dir=input,
        output_dir=output,
        data_type=format,
        grid_layout=grid_layout,
        label_positions=label_positions,
    )
    converter.run()
