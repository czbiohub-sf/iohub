from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Optional, Type, Union

import numpy as np
from numpy.typing import ArrayLike

_AXES_PREFIX = ["T", "C", "Z", "Y", "X"]


class BaseFOV(ABC):
    @property
    @abstractmethod
    def root(self) -> Path:
        raise NotImplementedError

    @property
    @abstractmethod
    def axes_names(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def channel_names(self) -> list[str]:
        raise NotImplementedError

    def channel_index(self, key: str) -> int:
        """Return index of given channel."""
        return self.channel_names.index(key)

    def _missing_axes(self) -> list[int]:
        """Return sorted indices of missing axes."""
        if len(self.axes_names) == 5:
            return []

        elif len(self.axes_names) > 5:
            raise ValueError(
                f"{self.__name__} does not support more than 5 axes. "
                f"Found {len(self.axes_names)}"
            )

        axes = set(ax[:1].upper() for ax in self.axes_names)

        missing = []
        for i, ax in enumerate(_AXES_PREFIX):
            if ax not in axes:
                missing.append(i)

        return missing

    def _pad_missing_axes(
        self,
        seq: Union[list[Any], tuple[Any]],
        value: Any,
    ) -> Union[list[Any], tuple[Any]]:
        """Pads ``seq`` with ``value`` in the missing axes positions."""

        if isinstance(seq, tuple):
            is_tuple = True
            seq = list(seq)
        else:
            is_tuple = False

        for i in self._missing_axes():
            seq.insert(i, value)

        if is_tuple:
            seq = tuple(seq)

        if len(seq) != len(_AXES_PREFIX):
            raise RuntimeError(
                f"Failed to pad raw axes {self.axes_names} to {_AXES_PREFIX}"
            )

        return seq

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int, int, int, int]:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(
        self,
        key: Union[int, slice, tuple[Union[int, slice], ...]],
    ) -> ArrayLike:
        """
        Returned object must support the ``__array__`` interface,
        so that ``np.asarray(...)`` will work.
        """
        raise NotImplementedError

    @property
    def ndim(self) -> int:
        return 5

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        raise NotImplementedError

    @property
    @abstractmethod
    def zyx_scale(self) -> tuple[float, float, float]:
        """Helper function for FOV spatial scale (micrometer)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def t_scale(self) -> float:
        """Helper function for FOV time scale (seconds)."""
        raise NotImplementedError

    def __eq__(self, other: BaseFOV) -> bool:
        if not isinstance(other, BaseFOV):
            return False
        return self.root.absolute() == other.root.absolute()


class BaseFOVMapping(Mapping):
    @abstractmethod
    def __enter__(self) -> BaseFOVMapping:
        """Open the underlying file and return self."""
        raise NotImplementedError

    @abstractmethod
    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        """Close the files."""
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, position_key: str) -> bool:
        """Check if a position is present in the collection."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, position_key: str) -> BaseFOV:
        """FOV key position to FOV object."""
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterable[tuple[str, BaseFOV]]:
        """Iterates over pairs of keys and FOVs."""
        raise NotImplementedError


class FOVDict(BaseFOVMapping):
    """
    Basic implementation of a mapping of strings to BaseFOVs.
    """

    def __init__(
        self,
        data_dict: Optional[dict[str, BaseFOV]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self._data = {}

        if data_dict is not None:
            for key, fov in data_dict.items():
                self._safe_insert(key, fov)

        for key, fov in kwargs.items():
            self._safe_insert(key, fov)

    def _safe_insert(self, key: str, value: BaseFOV) -> None:
        """Checks if types are correct and key is unique."""
        if not isinstance(key, str):
            raise TypeError(
                f"{self.__class__.__name__} key must be str. "
                f"Found {key} with type {type(key)}"
            )

        if not isinstance(value, BaseFOV):
            raise TypeError(
                f"{self.__class__.__name__} value must subclass BaseFOV. "
                f"Found {key} with value type {type(value)}"
            )

        if key in self:
            raise KeyError(f"{key} already exists.")

        self._data[key] = value

    def __contains__(self, position_key: str) -> bool:
        """Checks if position_key already exists."""
        return position_key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, position_key: str) -> BaseFOV:
        """FOV key position to FOV object."""
        return self._data[position_key]

    def __iter__(self) -> Iterable[tuple[str, BaseFOV]]:
        """Iterates over pairs of keys and FOVs."""
        return self._data.items()

    def __enter__(self) -> FOVDict:
        """Open the underlying file and return self."""
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        """Close the files."""
        return False
