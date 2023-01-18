import sys

import click

from iohub._version import __version__

VERSION = __version__


@click.version_option(version=VERSION)
def cli():
    """cli

    Parameters
    ----------
    ctx : click.Context

    """
    sys._excepthook = sys.excepthook

    def exception_hook(exctype, value, traceback):
        print(exctype, value, traceback)
        sys._excepthook(exctype, value, traceback)
        sys.exit(1)

    sys.excepthook = exception_hook

    pass


@cli.command()
@click.argument('files', nargs=-1)
def info(files):
    """info command

    Parameters
    ----------
    files

    """
    raise NotImplementedError()


@cli.command()
@click.argument('files', nargs=-1)
@click.option('-tf', '--target-format', default='', type=str)
def convert(files, **kwargs):
    """convert command

    Parameters
    ----------
    files
    kwargs

    """
    raise NotImplementedError()


if __name__ == '__main__':
    cli()
