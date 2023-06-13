from pathlib import Path
from typing import Any

import pytest

from iohub.fov import BaseFOV, FOVDict


class FOV(BaseFOV):
    def __init__(self, axes: list[str]) -> None:
        super().__init__()
        self._axes = axes

    @property
    def root(self) -> Path:
        return Path()

    @property
    def axes_names(self) -> list[str]:
        return self._axes

    @property
    def channel_names(self) -> list[str]:
        return []

    def __getitem__(self, key: Any) -> Any:
        pass

    @property
    def dtype(self) -> Any:
        pass

    @property
    def shape(self) -> Any:
        pass

    @property
    def zyx_scale(self) -> tuple[float, float, float]:
        return (1.0,) * 3

    @property
    def t_scale(self) -> float:
        raise 1.0


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


def test_fov_dict() -> None:

    good_collection = FOVDict(
        {
            "488": FOV(["y", "x"]),
            "561": FOV(["y", "x"]),
        },
        mask=FOV(["c", "x", "y"]),
    )

    assert len(good_collection) == 3
    assert "488" in good_collection
    assert good_collection["mask"] is not None

    with pytest.raises(TypeError):
        del good_collection["561"]

    with pytest.raises(TypeError):
        good_collection["segmentation"] = FOV(["x", "y"])

    with pytest.raises(TypeError):
        FOVDict({488: FOV(["y", "x"])})

    with pytest.raises(TypeError):
        FOVDict(mask=[1, 2, 3])
