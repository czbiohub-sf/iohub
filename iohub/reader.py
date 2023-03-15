from __future__ import annotations

import glob
import logging
import os
import sys
import warnings
from typing import TYPE_CHECKING, Literal

import natsort
import tifffile as tiff
import zarr

from iohub.multipagetiff import MicromanagerOmeTiffReader
from iohub.ndtiff import NDTiffReader
from iohub.ngff import NGFFNode, Plate, Position, open_ome_zarr
from iohub.reader_base import ReaderBase
from iohub.singlepagetiff import MicromanagerSequenceReader
from iohub.upti import UPTIReader
from iohub.zarrfile import ZarrReader

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath

# replicate from aicsimageio logging mechanism
###############################################################################

# modify the logging.ERROR level lower for more info
# CRITICAL
# ERROR
# WARNING
# INFO
# DEBUG
# NOTSET
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s"
# )
# log = logging.getLogger(__name__)

###############################################################################


# todo: add dim_order to all reader objects


def _check_zarr_data_type(src: str):
    try:
        root = zarr.open(src, "r")
        for key in ["omero", "plate"]:
            if key in root.attrs:
                return root.attrs[key].get("version")
    except Exception:
        return False
    return "unknown"


def _check_single_page_tiff(src: str):
    # pick parent directory in case a .tif file is selected
    if src.endswith(".tif"):
        src = os.path.dirname(src)

    files = glob.glob(os.path.join(src, "*.tif"))
    if len(files) == 0:
        sub_dirs = _get_sub_dirs(src)
        if sub_dirs:
            path = os.path.join(src, sub_dirs[0])
            files = glob.glob(os.path.join(path, "*.tif"))
            if len(files) > 0:
                with tiff.TiffFile(os.path.join(path, files[0])) as tf:
                    if (
                        len(tf.pages) == 1
                    ):  # and tf.pages[0].is_multipage is False:
                        return True
    return False


def _check_multipage_tiff(src: str):
    # pick parent directory in case a .tif file is selected
    if src.endswith(".tif"):
        src = os.path.dirname(src)

    files = glob.glob(os.path.join(src, "*.tif"))
    if len(files) > 0:
        with tiff.TiffFile(files[0]) as tf:
            if len(tf.pages) > 1:
                return True
            elif tf.is_multipage is False and tf.is_ome is True:
                return True
    return False


def _check_ndtiff(src: str):
    # go two levels up in case a .tif file is selected
    if src.endswith(".tif"):
        src = os.path.abspath(os.path.join(src, "../.."))

    # shortcut, may not be foolproof
    if os.path.exists(os.path.join(src, "Full resolution", "NDTiff.index")):
        return True
    elif os.path.exists(os.path.join(src, "NDTiff.index")):
        return True
    return False


def _get_sub_dirs(f: str):
    """
    subdir walk
    from https://github.com/mehta-lab/reconstruct-order

    Parameters
    ----------
    f:              (str)

    Returns
    -------
    sub_dir_name    (list) natsorted list of subdirectories
    """

    sub_dir_path = glob.glob(os.path.join(f, "*/"))
    sub_dir_name = [os.path.split(subdir[:-1])[1] for subdir in sub_dir_path]
    #    assert subDirName, 'No sub directories found'
    return natsort.natsorted(sub_dir_name)


def _infer_format(path: str):
    extra_info = None
    if ngff_version := _check_zarr_data_type(path):
        data_type = "omezarr"
        extra_info = ngff_version
    elif _check_ndtiff(path):
        data_type = "ndtiff"
    elif _check_multipage_tiff(path):
        data_type = "ometiff"
    elif _check_single_page_tiff(path):
        data_type = "singlepagetiff"
    else:
        raise FileNotFoundError(
            "Failed to infer data type: "
            f"No compatible data found under {path}."
        )
    return (data_type, extra_info)


def imread(
    path: str,
    data_type: Literal[
        "singlepagetiff", "ometiff", "ndtiff", "omezarr"
    ] = None,
    extract_data: bool = False,
    log_level: int = logging.WARNING,
):
    """Read image arrays and metadata from a bioimaging dataset.
    Supported formats are Micro-Manager TIFF formats
    (single-page TIFF, multi-page OME-TIFF, NDTIFF),
    and OME-Zarr (OME-NGFF v0.1 HCS and v0.4 FOV/HCS layouts).

    Parameters
    ----------
    path : str
        File path, directory path to ome-tiff series, or Zarr root path
    data_type :
    Literal["singlepagetiff", "ometiff", "ndtiff", "omezarr"], optional
        Dataset format, by default None
    extract_data : bool, optional
        True if ome_series should be extracted immediately for TIFF datasets,
        by default False
    log_level : int, optional
        One of 0, 10, 20, 30, 40, 50 for
        NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL, respectively,
        by default logging.WARNING

    Returns
    -------
    Reader
        A child instance of ReaderBase
    """

    logging.basicConfig(
        level=log_level,
        format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",  # noqa
    )
    logging.getLogger(__name__)

    # try to guess data type
    extra_info = None
    if data_type is None:
        fmt, extra_info = _infer_format(path)
    # identify data structure type
    if data_type == "ometiff":
        return MicromanagerOmeTiffReader(path, extract_data)
    elif data_type == "singlepagetiff":
        return MicromanagerSequenceReader(path, extract_data)
    elif data_type == "omezarr":
        if extra_info is None:
            _, extra_info = _infer_format(path)
        if extra_info == "0.1":
            return ZarrReader(path, version=extra_info)
        elif extra_info == "0.4":
            # `warnings` instead of `logging` since this can be avoided
            warnings.warn(
                UserWarning(
                    "For NGFF v0.4 datasets, `iohub.open_ome_zarr()` "
                    "is preferred over `iohub.imread()`. "
                    "These functions will return the same NGFF objects "
                    "that are different from other readers from `imread`."
                )
            )
            return open_ome_zarr(
                path, layout="auto", mode="r", version=extra_info
            )
        else:
            raise ValueError(f"NGFF version {extra_info} is not supported.")
    elif data_type == "ndtiff":
        return NDTiffReader(path)
    elif data_type == "upti":
        return UPTIReader(path, extract_data)
    else:
        raise ValueError(f"Reader of type {data_type} is not implemented")


def print_info(path: StrOrBytesPath, verbose=False):
    """Print summary information for a dataset.

    Parameters
    ----------
    path : StrOrBytesPath
        Path to the dataset
    verbose : bool, optional
        Show usage guide to open dataset in Python
        and full tree for HCS Plates in OME-Zarr,
        by default False
    """
    path = os.path.realpath(path)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="iohub"
            )
            fmt, extra_info = _infer_format(path)
            reader = imread(path, data_type=fmt)
    except FileNotFoundError:
        print("Error: No compatible dataset is found.", file=sys.stderr)
        return
    fmt_msg = f"Format:\t\t {fmt}"
    if extra_info:
        if extra_info.startswith("0."):
            fmt_msg += " v" + extra_info
    ch_msg = f"Channel names:\t {reader.channel_names}"
    code_msg = "\nThis datset can be opened with iohub in Python code:\n"
    msgs = []
    if isinstance(reader, ReaderBase):
        if verbose:
            msgs.extend(
                [
                    code_msg,
                    ">>> from iohub import imread",
                    f">>> reader = imread('{path}')",
                ]
            )
        msgs.extend(
            [
                "\nSummary:",
                fmt_msg,
                f"Positions:\t {reader.get_num_positions()}",
                f"Time points:\t {reader.shape[0]}",
                f"Channels:\t {reader.shape[1]}",
                ch_msg,
                f"(Z, Y, X):\t {reader.shape[2:]}",
                f"Z step (um):\t {reader.z_step_size}",
            ]
        )
        print(str.join("\n", msgs))
    elif isinstance(reader, NGFFNode):
        if verbose:
            msgs.extend(
                [
                    code_msg,
                    ">>> from iohub import open_ome_zarr",
                    f">>> dataset = open_ome_zarr('{path}', mode='r')",
                ]
            )
        msgs.extend(
            [
                "\nSummary:",
                fmt_msg,
                f"Axes:\t\t {[a.type for a in reader.axes]}",
                ch_msg,
            ]
        )
        if isinstance(reader, Plate):
            meta = reader.metadata
            msgs.extend(
                [
                    f"Row names:\t {[r.name for r in meta.rows]}",
                    f"Column names:\t {[c.name for c in meta.columns]}",
                    f"Wells:\t\t {len(meta.wells)}",
                ]
            )
        print(str.join("\n", msgs))
        if isinstance(reader, Position) or verbose:
            reader.print_tree()
