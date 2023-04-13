from typing import Tuple, TYPE_CHECKING

import blosc2
import numpy as np

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath


def _compressed_to_array(
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
