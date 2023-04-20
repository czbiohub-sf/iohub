import math as m
from typing import Sequence, Tuple
from copy import deepcopy

from iohub.ngff import Position


def _scale_integers(values: Sequence[int], factor: int) -> Tuple[int, ...]:
    """Computes the ceiling of the input sequence divided by the factor."""
    return tuple(int(m.ceil(v / factor)) for v in values)


def initialize_pyramid(fov: Position, levels: int) -> None:
    """
    Initializes the pyramid arrays with a down scaling of 2 per level.
    Decimals shapes are rounded up to ceiling.
    Scales metadata are also updated.

    Parameters
    ----------
    fov : Position
        Input NGFF field of view to be updated.
    levels : int
        Number of down scaling levels, if levels is 1 nothing happens.
    """
    array = fov.data
    for l in range(1, levels):
        factor = 2 ** l
        shape = array.shape[:2] + _scale_integers(array.shape[2:], factor)
        chunks = (1, 1) + _scale_integers(array.shape[2:], factor)

        transforms = deepcopy(fov.metadata.multiscales[0].datasets[0].coordinate_transformations)
        for tr in transforms:
            if tr.type == "scale":
                for i in range(2, len(tr.scale)):
                    tr.scale[i] /= factor

        fov.create_zeros(
            name=str(l),
            shape=shape,
            dtype=array.dtype,
            chunks=chunks,
            transform=transforms,
        )
