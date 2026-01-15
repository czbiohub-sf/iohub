from __future__ import annotations

import re
from pathlib import Path
from typing import overload

from xarray import DataArray

from iohub.fov import BaseFOV, BaseFOVMapping

# Compile regex pattern once at module level for efficiency
# See https://chatgpt.com/share/e/68364412-fb4c-8002-8dcf-28127cfee37a
_HCS_POSITION_PATTERN = re.compile(
    r"([A-Z])(\d+)-Site_(\d+)|"
    r"Pos-(\d+)-(\d+)_(\d+)|"
    r"(\d+)-Pos(\d+)_?(\d+)?"
)


class MicroManagerFOV(BaseFOV):
    def __init__(self, parent: MicroManagerFOVMapping, key: int) -> None:
        self._position = key
        self._parent = parent

    def __repr__(self) -> str:
        return (
            f"Type: {type(self)}\n"
            f"Parent: {self.parent}\n"
            f"FOV key: {self._position}\n"
            f"Data:\n"
        ) + self.xdata.__repr__()

    def __eq__(self, other: BaseFOV) -> bool:
        if not isinstance(other, type(self)):
            return False
        return (self._position == other._position) and (
            self.root.absolute() == other.root.absolute()
        )

    @property
    def parent(self) -> MicroManagerFOVMapping:
        return self._parent

    @property
    def root(self) -> Path:
        return self.parent.root

    @property
    def zyx_scale(self) -> tuple[float, float, float]:
        return self.parent.zyx_scale

    @property
    def t_scale(self) -> float:
        return self.parent.t_scale

    @property
    def channel_names(self) -> list[str]:
        return self.parent.channel_names

    @property
    def xdata(self) -> DataArray:
        raise NotImplementedError

    def frame_metadata(self, t: int, z: int, c: int) -> dict | None:
        """
        Return image plane metadata for a given camera frame.

        Parameters
        ----------
        t : int
            Time index.
        z : int
            Z slice index.
        c : int
            Channel index.

        Returns
        -------
        dict | None
            Image plane metadata. None if not available.
        """
        raise NotImplementedError


class MicroManagerFOVMapping(BaseFOVMapping):
    def __init__(self):
        self._root: Path = None
        self._mm_meta: dict = None
        self._stage_positions: list[dict[str, str | float]] = []
        self.channel_names: list[str] = None

    def __repr__(self) -> str:
        return (f"Type: {type(self)}\nData:\n") + self.xdata.__repr__()

    @property
    def root(self) -> Path:
        """Root directory of the dataset."""
        return self._root

    @property
    def micromanager_metadata(self) -> dict | None:
        return self._mm_meta

    @micromanager_metadata.setter
    def micromanager_metadata(self, value):
        if not isinstance(value, dict):
            raise TypeError(
                f"Type of `mm_meta` should be `dict`, got `{type(value)}`."
            )
        self._mm_meta = value

    @property
    def micromanager_summary(self) -> dict | None:
        """Micro-manager summary metadata."""
        return self._mm_meta.get("Summary", None)

    @property
    def stage_positions(self):
        return self._stage_positions

    @stage_positions.setter
    def stage_positions(self, value):
        if not isinstance(value, list):
            raise TypeError(
                "Type of `stage_position` should be `list`, "
                f"got `{type(value)}`."
            )
        self._stage_positions = value

    @overload
    @staticmethod
    def _parse_hcs_position_label(label: str) -> tuple[str, str, str]: ...

    @overload
    @staticmethod
    def _parse_hcs_position_label(
        label: list[str],
    ) -> list[tuple[str, str, str]]: ...

    @staticmethod
    def _parse_hcs_position_label(
        label: str | list[str],
    ) -> tuple[str, str, str] | list[tuple[str, str, str]]:
        """Parse HCS position label(s) into (row, column, fov) components.

        Supports both single labels and lists of labels for flexible usage.

        Parameters
        ----------
        label : str or list[str]
            Single HCS position label or list of labels to parse

        Returns
        -------
        tuple[str, str, str] or list[tuple[str, str, str]]
            Parsed (row, column, fov) components. Returns single tuple for
            string input, list of tuples for list input.

        Raises
        ------
        ValueError
            If any label doesn't match supported formats
        """
        if isinstance(label, list):
            return [
                MicroManagerFOVMapping._parse_hcs_position_label(lbl)
                for lbl in label
            ]

        if (match := re.match(_HCS_POSITION_PATTERN, label)) is not None:
            if match.group(1):  # "A1-Site_0" case
                return (match.group(1), match.group(2), match.group(3))
            elif match.group(4):  # "Pos-5-000_005" case
                return ("0", match.group(4), match.group(5) + match.group(6))
            elif match.group(7):  # "1-Pos000_000" or "1-Pos000" case
                optional_last_part = match.group(9) or "000"
                return (
                    "0",
                    match.group(7),
                    match.group(8) + optional_last_part,
                )
        else:
            raise ValueError(
                f"HCS position label {label} is not in the format of "
                "'A1-Site_0', 'H12-Site_1', '1-Pos000_000', '1-Pos000"
                f"or 'Pos-1-000_000'"
            )

    @property
    def hcs_position_labels(self):
        """Parse plate position labels generated by the HCS position generator
        and split them into (row, column, FOV) components.

        This method supports multiple label formats commonly produced by
        micromanager. The returned values are a 3-part tuple with the
        following interpretation:
        - `row`: usually a letter (e.g. 'A'), or '0' if not explicitly given
        - `column`: the well column number or site index
        - `fov`: the field-of-view index, typically zero-padded

        Supported label formats and their outputs:

        ================  ========================
        Format            Output
        ================  ========================
        "A1-Site_0"       ('A', '1', '0')
        "1-Pos000_000"    ('0', '1', '000000')
        "2-Pos000_001"    ('0', '2', '000001')
        "1-Pos001"        ('0', '1', '001000')
        "Pos-5-000_005"   ('0', '5', '000005')
        ================  ========================

        Returns
        -------
        list[tuple[str, str, str]]
            A list of (row, column, fov) tuples corresponding to parsed label
            components in a uniform format.

        Raises
        ------
        ValueError
            If stage position metadata is missing or labels are not found.
        """
        if not self.stage_positions:
            raise ValueError("Stage position metadata not available.")

        try:
            labels = [pos["Label"] for pos in self.stage_positions]
        except KeyError:
            raise ValueError("Stage positions do not have labels.")

        return self._parse_hcs_position_label(labels)

    @property
    def zyx_scale(self) -> tuple[float, float, float]:
        """ZXY pixel size in micrometers."""
        raise NotImplementedError

    @property
    def t_scale(self) -> float:
        """Time scale in seconds."""
        raise NotImplementedError
