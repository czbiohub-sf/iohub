import click

from iohub._version import __version__
from iohub.reader import ImageReader

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
        reader = ImageReader(file)
        print_reader_info(reader)


def print_reader_info(reader):
    print(f"Positions:\t {reader.get_num_positions()}")
    print(f"Time points:\t {reader.shape[0]}")
    print(f"Channels:\t {reader.shape[1]}")
    print(f"(Z, Y, X):\t {reader.shape[2:]}")
    print(f"Channel names:\t {reader.channel_names}")
    print(f"Z step (um):\t {reader.z_step_size}")
    print("")
