import re
import json
import warnings
from typing import Any, Tuple, TYPE_CHECKING, List, Sequence, Dict, Optional
from pathlib import Path

import blosc2
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath


def blosc_buffer_to_array(
    buffer_path: "StrOrBytesPath",
    shape: Tuple[int, ...],
    dtype: np.dtype,
    nthreads: int = 4,
) -> np.ndarray:
    """Loads compressed "blosc" file and converts into numpy array.

    Parameters
    ----------
    buffer_path : StrOrBytesPath
        Compressed blosc buffer path.
    shape : Tuple[int, ...]
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

            chunk_size, compress_chunk_size, _ = blosc2.get_cbuffer_sizes(blosc_header)

            # move to before the header and read chunk
            f.seek(f.tell() - header_size)
            chunk_buffer = f.read(compress_chunk_size)

            blosc2.decompress2(chunk_buffer, array_buffer, nthreads=nthreads)
            array_buffer = array_buffer[chunk_size // out_arr.itemsize:]

    return out_arr.reshape(shape)


class ClearControlFOV:
    """
    Reader class for Clear Control dataset (https://github.com/royerlab/opensimview).
    It provides a array-like API for the Clear Control dataset while loading the volumes lazily.

    It assumes the channels and volumes have the same shape, the minimum from each channel is used.

    Parameters
    ----------
    data_path : StrOrBytesPath
        Clear Control dataset path.
    missing_value : Optional[int], optional
        If provided this class won't raise an error when missing a volume and
        it will return an array with the provided value.
    """
    def __init__(self, data_path: "StrOrBytesPath", missing_value: Optional[int] = None):
        super().__init__()
        self._data_path = Path(data_path)
        self._missing_value = missing_value
        self._dtype = np.uint16
        self._prev_key = None
        self._prev_array = None
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Reads Clear Control index data of every data and returns the element-wise minimum shape."""

        # dummy maximum shape size
        shape = [65535,] * 4
        # guess of minimum line length, it might be wrong
        minimum_size = 64
        numbers = re.compile(r"\d+\.\d+|\d+")

        for index_filepath in self._data_path.glob("*.index.txt"):
            with open(index_filepath, "rb") as f:
                if index_filepath.stat().st_size > minimum_size:
                    f.seek(-minimum_size, 2)  # goes to a little bit before the last line
                last_line = f.readlines()[-1].decode("utf-8")

                values = list(numbers.findall(last_line))
                values = [int(values[0]), int(values[4]), int(values[3]), int(values[2])]

                shape = [min(s, v) for s, v in zip(shape, values)]
        
        shape.insert(1, len(self.channels))

        return tuple(shape)
    
    @property
    def channels(self) -> List[str]:
        """Return sorted channels name."""
        suffix = ".index.txt"
        return sorted([
            p.name.removesuffix(suffix)
            for p in self._data_path.glob(f"*{suffix}")
        ])
    
    def _read_volume(
        self,
        volume_shape: Tuple[int, int, int],
        channels: Sequence[str] | str,
        time_point: int,
    ) -> np.ndarray:
        """Reads a single or multiple channels of blosc compressed Clear Control volume.

        Parameters
        ----------
        volume_shape : Tuple[int, int, int]
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
            volume_path = self._data_path / "stacks" / channels / volume_name
            if not volume_path.exists():
                if self._missing_value is None:
                    raise ValueError(f"{volume_path} not found.")
                else:
                    warnings.warn(f"{volume_path} not found. Filled with {self._missing_value}")
                    return np.full(volume_name, self._missing_value, dtype=self._dtype)
            return blosc_buffer_to_array(volume_path, volume_shape, dtype=self._dtype)
        
        return np.stack(
            [self._read_volume(volume_shape, ch, time_point) for ch in channels]
        )
    
    @staticmethod
    def _fix_indexing(indexing: int | slice | List | np.ndarray) -> int | slice | List | np.ndarray:
        """Converts numpy array to simple python type or list."""
        if isinstance(indexing, np.ScalarType):
            return indexing.item()
        return indexing

    def __getitem__(
        self, key: (
            int |
            slice |
            List |
            np.ndarray |
            Tuple[int, ...] |
            Tuple[slice, ...]
        ),
    ) -> np.ndarray:
        """Lazily load array as indexed.

        Parameters
        ----------
        key : int  |  slice  |  List  |  Tuple[int, ...]  |  Tuple[slice, ...]
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
        if key == self._prev_key:
            return self._prev_array

        # these are properties are loaded to avoid multiple reads per call
        shape = self.shape
        channels = self.channels
        time_pts = list(range(shape[0]))
        volume_shape = shape[-3:]

        err_msg = NotImplementedError(f"ClearControlFOV indexing not implemented for {key}."
                                       "Only Integer, List and slice indexing are available.")

        # querying time points and channels at once
        if isinstance(key, Tuple):
            key = tuple(self._fix_indexing(k) for k in key)

            if len(key) == 1:
                return self.__getitem__(key[0])

            if len(key) == 2:
                T, C = key
                arr_keys = ...

            else:
                T, C = key[:2]
                arr_keys = key[2:]
            
            # new query, checking only first two positions
            if self._prev_key != key[:2]:
                # single time point
                if isinstance(T, int):
                    self._prev_array = self._read_volume(volume_shape, channels[C], T) 
                
                # multiple time points
                elif isinstance(T, (List, slice, np.ndarray)):
                    self._prev_array = np.stack([
                        self._read_volume(volume_shape, channels[C], t)
                        for t in time_pts[T]
                    ])
                
                else:
                    raise err_msg

                # only saved the two first keys because the others belong to a single chunk (volume)
                self._prev_key = key[:2]
            
            return self._prev_array[arr_keys]

        # querying a single time point
        elif isinstance(key, int):
            key = self._fix_indexing(key)
            self._prev_array = self._read_volume(self._data_path, volume_shape, channels, key)

        # querying multiple time points
        elif isinstance(key, (List, slice, np.ndarray)):
            self._prev_array = np.stack([
                self.__getitem__(t) for t in time_pts[key]
            ])

        else:
            raise err_msg

        self._prev_key = key
        return self._prev_array

    def __setitem__(self, key: Any, value: Any) -> None:
        raise PermissionError("ClearControlFOV is read-only.")
    
    @property
    def ndim(self) -> int:
        return 5
    
    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def metadata(self) -> Dict[str, Any]:
        """Summarizes Clear Control metadata into a dictionary."""
        cc_metadata = []
        for path in self._data_path.glob("*.metadata.txt"):
            with open(path, mode="r") as f:
                channel_metadata = pd.concat([
                    json.loads(s) for s in f.readlines()
                ])
            cc_metadata.append(channel_metadata)
        
        cc_metadata = pd.concat(cc_metadata)

        time_delta = cc_metadata.groupby("Channel")["TimeStampInNanoSeconds"].diff()
        acquisition_type = cc_metadata["AcquisitionType"].first()

        metadata = {
            "voxel_size_z": cc_metadata["VoxelDimZ"].mean(),     # micrometers
            "voxel_size_y": cc_metadata["VoxelDimY"].mean(),     # micrometers
            "voxel_size_x": cc_metadata["VoxelDimX"].mean(),     # micrometers
            "time_delta": time_delta.mean().mean() / 1_000_000,  # seconds
            "acquisition_type": acquisition_type,
        }

        return metadata
