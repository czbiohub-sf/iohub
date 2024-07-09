import json
import re
import warnings
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Union

import blosc2
import numpy as np
import pandas as pd

from iohub.fov import BaseFOV

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath


ArrayIndex = Union[int, slice, list[int], np.ndarray]


def _array_to_blosc_buffer(
    in_array: np.ndarray,
    out_path: "StrOrBytesPath",
    overwrite: bool = False,
) -> None:
    """
    Compresses array and save into output path.
    This function does not use new functionality from `blosc2` to
    compress large arrays on purpose, to emulate Clear Control behavior.

    NOTE: function is not optimized and copies the data into bytes,
          this is used mainly for testing so this is not critical.

    Parameters
    ----------
    in_array : np.ndarray
        Input array.
    out_path : StrOrBytesPath
        Output blosc compressed buffer path.
    overwrite : bool
        When true it allows overwriting existing path.
    """
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        raise ValueError(f"{out_path} already exists use `ovewrite=True`.")

    kwargs = {"clevel": 3, "compressor": "lz4"}

    arr_bytes = in_array.tobytes()

    # make chunk size uniform
    num_chunks = len(arr_bytes) // blosc2.MAX_BUFFERSIZE + 1
    chunk_size = len(arr_bytes) // num_chunks

    with open(out_path, "wb") as f:
        while len(arr_bytes) > 0:
            compressed_chunk = blosc2.compress2(
                arr_bytes[:chunk_size], **kwargs
            )
            f.write(compressed_chunk)
            arr_bytes = arr_bytes[chunk_size:]


def blosc_buffer_to_array(
    buffer_path: "StrOrBytesPath",
    shape: tuple[int, ...],
    dtype: np.dtype,
    nthreads: int = 4,
) -> np.ndarray:
    """Loads compressed "blosc" file and converts into numpy array.

    Parameters
    ----------
    buffer_path : StrOrBytesPath
        Compressed blosc buffer path.
    shape : tuple[int, ...]
        Output array shape.
    dtype : np.dtype
        Output array data type.
    nthreads : int, optional
        Number of blosc decompression threads, by default 4

    Returns
    -------
    np.ndarray
        Output numpy array.
    """
    header_size = 32
    out_arr = np.empty(np.prod(shape), dtype=dtype)
    array_buffer = out_arr

    with open(buffer_path, "rb") as f:
        while True:
            # read header only
            blosc_header = bytes(f.read(header_size))
            if not blosc_header:
                break

            chunk_size, compress_chunk_size, _ = blosc2.get_cbuffer_sizes(
                blosc_header
            )

            # move to before the header and read chunk
            f.seek(f.tell() - header_size)
            chunk_buffer = f.read(compress_chunk_size)

            blosc2.decompress2(chunk_buffer, array_buffer, nthreads=nthreads)
            array_buffer = array_buffer[chunk_size // out_arr.itemsize :]

    return out_arr.reshape(shape)


def _cached(f: Callable) -> Callable:
    """Decorator that caches the array data using its key."""

    @wraps(f)
    def _key_cache_wrapper(
        self: "ClearControlFOV",
        key: Union[ArrayIndex, tuple[ArrayIndex, ArrayIndex]],
    ) -> np.ndarray:
        if not self._cache:
            return f(self, key)

        elif key != self._cache_key:
            self._cache_array = f(self, key)
            self._cache_key = key

        return self._cache_array

    return _key_cache_wrapper


class ClearControlFOV(BaseFOV):
    """
    Reader class for Clear Control dataset
    https://github.com/royerlab/opensimview.

    It provides an array-like API for the Clear Control
    dataset thats loads the volumes lazily.

    It assumes the channels and volumes have the same shape,
    when that is not the case the minimum value from each dimension is used.

    Parameters
    ----------
    data_path : StrOrBytesPath
        Clear Control dataset path.
    missing_value : Optional[int], optional
        If provided this class won't raise an error when missing a volume and
        it will return an array with the provided value.
    cache : bool
        When true caches the last array using the first two indices as key.
    """

    def __init__(
        self,
        data_path: "StrOrBytesPath",
        missing_value: Optional[int] = None,
        cache: bool = False,
    ):
        super().__init__()
        self._root = Path(data_path)
        self._missing_value = missing_value
        self._dtype = np.uint16
        self._cache = cache
        self._cache_key = None
        self._cache_array = None

    @property
    def root(self) -> Path:
        return self._root

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        """
        Reads Clear Control index data of every data and returns
        the element-wise minimum shape.
        """

        # dummy maximum shape size
        shape = [65535] * 4
        # guess of minimum line length, it might be wrong
        minimum_size = 64
        numbers = re.compile(r"\d+\.\d+|\d+")

        for index_filepath in self._root.glob("*.index.txt"):
            with open(index_filepath, "rb") as f:
                if index_filepath.stat().st_size > minimum_size:
                    f.seek(
                        -minimum_size, 2
                    )  # goes to a little bit before the last line
                last_line = f.readlines()[-1].decode("utf-8")

                values = list(numbers.findall(last_line))
                values = [
                    int(values[0]),
                    int(values[4]),
                    int(values[3]),
                    int(values[2]),
                ]

                shape = [min(s, v) for s, v in zip(shape, values)]

        shape.insert(1, len(self.channel_names))
        shape[0] += 1  # time points starts counts on zero

        return tuple(shape)

    @property
    def axes_names(self) -> list[str]:
        return ["T", "C", "Z", "Y", "X"]

    @property
    def channel_names(self) -> list[str]:
        """Return sorted channels name."""
        suffix = ".index.txt"
        return sorted(
            [
                p.name.removesuffix(suffix)
                for p in self._root.glob(f"*{suffix}")
            ]
        )

    def _read_volume(
        self,
        volume_shape: tuple[int, int, int],
        channels: Union[Sequence[str], str],
        time_point: int,
    ) -> np.ndarray:
        """
        Reads a single or multiple channels of blosc compressed
        Clear Control volume.

        Parameters
        ----------
        volume_shape : tuple[int, int, int]
            3-dimensional volume shape (z, y, x).
        channels : Sequence[str] | str]
            Channels names.
        time_point : int
            Volume time point.

        Returns
        -------
        np.ndarray
            Volume as an array can be single or multiple channels.

        Raises
        ------
        ValueError
            When expected volume path not found.
        """
        # single channel
        if isinstance(channels, str):
            volume_name = f"{str(time_point).zfill(6)}.blc"
            volume_path = self._root / "stacks" / channels / volume_name
            if not volume_path.exists():
                if self._missing_value is None:
                    raise ValueError(f"{volume_path} not found.")
                else:
                    warnings.warn(
                        f"{volume_path} not found. "
                        f"Filled with {self._missing_value}"
                    )
                    return np.full(
                        volume_name, self._missing_value, dtype=self._dtype
                    )
            return blosc_buffer_to_array(
                volume_path, volume_shape, dtype=self._dtype
            )

        return np.stack(
            [
                self._read_volume(volume_shape, ch, time_point)
                for ch in channels
            ]
        )

    @staticmethod
    def _fix_indexing(indexing: ArrayIndex) -> ArrayIndex:
        """Converts numpy array to simple python type or list."""
        if np.isscalar(indexing):
            try:
                return indexing.item()
            except AttributeError:
                return indexing

        elif isinstance(indexing, np.ndarray):
            return indexing.tolist()

        return indexing

    def __getitem__(
        self, key: Union[ArrayIndex, tuple[ArrayIndex, ...]]
    ) -> np.ndarray:
        """Lazily load array as indexed.

        Parameters
        ----------
        key : ArrayIndex | tuple[ArrayIndex, ...]
            An indexing key as in numpy, but a bit more limited.

        Returns
        -------
        np.ndarray
            Output array.

        Raises
        ------
        NotImplementedError
            Not all numpy array of indexing are implemented.
        """
        # standardizing indexing
        volume_slicing = None
        if isinstance(key, tuple):
            key = tuple(self._fix_indexing(k) for k in key)
            if len(key) == 1:
                key = key[0]
            elif len(key) > 2:
                key, volume_slicing = key[:2], key[2:]
        else:
            key = self._fix_indexing(key)

        if volume_slicing is None:
            return self._load_array(key)

        return self._load_array(key)[volume_slicing]

    @_cached
    def _load_array(
        self,
        key: Union[ArrayIndex, tuple[ArrayIndex, ArrayIndex]],
    ) -> np.ndarray:
        # these are properties are loaded to avoid multiple reads per call
        shape = self.shape
        channels = np.asarray(self.channel_names)
        time_pts = list(range(shape[0]))
        volume_shape = shape[-3:]

        err_msg = NotImplementedError(
            "ClearControlFOV indexing not implemented for first "
            f"two indices {key}. Only int, list[int], slice, "
            "and np.ndarray indexing are available."
        )

        # querying time points and channels at once
        if isinstance(key, tuple):
            T, C = key
            # single time point
            if isinstance(T, int):
                return self._read_volume(
                    volume_shape, channels[C], time_pts[T]
                )

            # multiple time points
            elif isinstance(T, (list, slice, np.ndarray)):
                return np.stack(
                    [
                        self._read_volume(volume_shape, channels[C], t)
                        for t in time_pts[T]
                    ]
                )

            else:
                raise err_msg

        # querying a single time point
        elif isinstance(key, int):
            return self._read_volume(volume_shape, channels, key)

        # querying multiple time points
        elif isinstance(key, (list, slice, np.ndarray)):
            return np.stack([self.__getitem__(t) for t in time_pts[key]])

        else:
            raise err_msg

    def __setitem__(self, key: Any, value: Any) -> None:
        raise PermissionError("ClearControlFOV is read-only.")

    @property
    def ndim(self) -> int:
        return 5

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def cache(self) -> bool:
        return self._cache

    @cache.setter
    def cache(self, value: bool) -> None:
        """Free current key/array cache."""
        self._cache = value
        if not value:
            self._cache_array = None
            self._cache_key = None

    def metadata(self) -> dict[str, Any]:
        """Summarizes Clear Control metadata into a dictionary."""
        cc_metadata = []
        for path in self._root.glob("*.metadata.txt"):
            with open(path, mode="r") as f:
                channel_metadata = pd.DataFrame(
                    [json.loads(s) for s in f.readlines()]
                )
            cc_metadata.append(channel_metadata)

        cc_metadata = pd.concat(cc_metadata)

        time_delta = cc_metadata.groupby("Channel")[
            "TimeStampInNanoSeconds"
        ].diff()
        acquisition_type = cc_metadata["AcquisitionType"].iat[0]

        metadata = {
            "voxel_size_z": cc_metadata["VoxelDimZ"].mean(),  # micrometers
            "voxel_size_y": cc_metadata["VoxelDimY"].mean(),  # micrometers
            "voxel_size_x": cc_metadata["VoxelDimX"].mean(),  # micrometers
            "time_delta": time_delta.mean().mean() / 1_000_000,  # seconds
            "acquisition_type": acquisition_type,
        }

        return metadata

    @property
    def scale(self) -> list[float]:
        """Dataset temporal, channel and spacial scales."""
        warnings.warn(
            ".scale will be deprecated use .zyx_scale or .t_scale.",
            category=DeprecationWarning,
        )
        metadata = self.metadata()
        return [
            metadata["time_delta"],
            1.0,
            metadata["voxel_size_z"],
            metadata["voxel_size_y"],
            metadata["voxel_size_x"],
        ]

    @property
    def zyx_scale(self) -> tuple[float, float, float]:
        """Helper function for FOV spatial scale (micrometer)."""
        metadata = self.metadata()
        return (
            metadata["voxel_size_z"],
            metadata["voxel_size_y"],
            metadata["voxel_size_x"],
        )

    @property
    def t_scale(self) -> float:
        """Helper function for FOV time scale (seconds)."""
        metadata = self.metadata()
        return metadata["time_delta"]


def create_mock_clear_control_dataset(path: "StrOrBytesPath") -> None:
    """
    Creates a (2, 4, 64, 64, 64) Clear Control dataset of random integers.

    Parameters
    ----------
    path : StrOrBytesPath
        Dataset output path.
    """
    path = Path(path)

    rng = np.random.default_rng(0)
    array = rng.integers(0, 1_500, size=(2, 4, 64, 64, 64), dtype=np.uint16)
    channels = ["C0L0", "C0L1", "C1L0", "C1L1"]
    shape_str = f"{array.shape[2]}, {array.shape[3]}, {array.shape[4]}"

    metadata = {
        "VoxelDimY": 0.25,
        "VoxelDimX": 0.25,
        "VoxelDimZ": 1.0,
        "AcquisitionType": "NA",
    }

    assert len(channels) == array.shape[1]

    for c, ch in enumerate(channels):
        channel_dir = path / "stacks" / ch
        channel_dir.mkdir(parents=True, exist_ok=True)

        index_path = path / f"{ch}.index.txt"
        metadata_path = path / f"{ch}.metadata.txt"

        with open(index_path, "w") as idx_f, open(metadata_path, "w") as mt_f:
            for t in range(array.shape[0]):
                out_path = channel_dir / f"{str(t).zfill(6)}.blc"
                time_stamp = 45_000_000 * t
                _array_to_blosc_buffer(array[t, c], out_path, overwrite=True)
                volume_metadata = dict(
                    Channel=ch, TimeStampInNanoSeconds=time_stamp, **metadata
                )
                mt_f.write(f"{json.dumps(volume_metadata)}\n")
                idx_f.write(f"{t} {time_stamp / 1_000_000:.4f} {shape_str}\n")
