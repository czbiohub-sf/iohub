import numpy as np
from pathlib import Path

from iohub.clearcontrol import array_to_blosc_buffer, blosc_buffer_to_array, ClearControlFOV


def test_blosc_buffer(tmp_path: Path) -> None:

    buffer_path = tmp_path / "buffer.blc"
    in_array = np.random.randint(0, 5_000, size=(32, 32))

    array_to_blosc_buffer(in_array, buffer_path)
    out_array = blosc_buffer_to_array(buffer_path, in_array.shape, in_array.dtype)

    assert np.allclose(in_array, out_array)
