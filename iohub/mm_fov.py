from __future__ import annotations

from pathlib import Path

from iohub.fov import BaseFOV, BaseFOVMapping


class MicroManagerFOV(BaseFOV):
    def __init__(self, parent: MicroManagerFOVMapping, key: int) -> None:
        self._position = key
        self._parent = parent

    @property
    def parent(self) -> MicroManagerFOVMapping:
        return self._parent

    @property
    def root(self) -> Path:
        return self.parent.root

    @property
    def zyx_scale(self) -> tuple[float, float, float]:
        return self.parent.zyx_scales

    @property
    def channel_names(self) -> list[str]:
        return self.parent.channel_names


class MicroManagerFOVMapping(BaseFOVMapping):
    pass
