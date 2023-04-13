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

    header_size = 32
    out_arr = np.empty(np.prod(shape), dtype=dtype)

    with open(buffer_path, "rb") as f:
        compressed_buffer = f.read()

    array_buffer = out_arr

    while len(compressed_buffer) > header_size:
        blosc_header = bytes(compressed_buffer[:header_size])

        buffer_size, compressed_buffer_size, _ = blosc2.get_cbuffer_sizes(blosc_header)
        
        blosc2.decompress2(compressed_buffer[:compressed_buffer_size], array_buffer, nthreads=nthreads)

        array_buffer = array_buffer[buffer_size // out_arr.itemsize:]
        compressed_buffer = compressed_buffer[compressed_buffer_size:]

    return out_arr.reshape(shape)
