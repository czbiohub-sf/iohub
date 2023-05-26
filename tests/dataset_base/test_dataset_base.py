from typing import Any

import pytest

from iohub.dataset_base import BaseFOV


class FOV(BaseFOV):
    def __init__(self, axes: list[str]) -> None:
        super().__init__()
        self._axes = axes

    @property
    def axes_names(self) -> list[str]:
        return self._axes

    @property
    def channels(self) -> list[str]:
        return []

    def __getitem__(self, key: Any) -> Any:
        pass

    @property
    def dtype(self) -> Any:
        pass

    @property
    def shape(self) -> Any:
        pass


@pytest.mark.parametrize(
    "axes,missing",
    [
        (["T", "C", "Z", "Y", "X"], []),
        (["time", "z", "y", "x"], [1]),
        (["channels", "Y", "X"], [0, 2]),
    ],
)
def test_missing_axes(axes: list[str], missing: list[int]) -> None:
    fov = FOV(axes)
    assert fov._missing_axes() == missing

    shape = (10,) * len(axes)
    padded_shape = fov._pad_missing_axes(shape, 1)
    assert len(padded_shape) == 5

    for i, s in enumerate(padded_shape):
        if i in missing:
            assert s == 1
        else:
            assert s == 10
