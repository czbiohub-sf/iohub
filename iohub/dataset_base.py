from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, Generator, Optional, Type, Union

import numpy as np
from numpy.typing import ArrayLike

_AXES_PREFIX = ["T", "C", "Z", "Y", "X"]


class BaseFOV(ABC):
    # NOTE: not using the `data` method from ngff Position

    @property
    @abstractmethod
    def axes_names(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def channels(self) -> list[str]:
        raise NotImplementedError

    def channel_index(self, key: str) -> int:
        """Return index of given channel."""
        return self.channels.index(key)

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

        assert len(seq) == len(_AXES_PREFIX)

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
    def zyx_scale(self) -> tuple[float, float, float]:
        """Helper function for FOV spatial scale."""
        raise NotImplementedError


class BaseFOVCollection(ABC):
    @abstractmethod
    def __enter__(self) -> "BaseFOVCollection":
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
    def __iter__(self) -> Generator[tuple[str, BaseFOV], None, None]:
        """Iterates over pairs of keys and FOVs."""
        raise NotImplementedError
