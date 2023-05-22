from abc import ABC, abstractmethod
from types import TracebackType
from typing import Generator, Optional, Type, Union

import numpy as np
from numpy.typing import ArrayLike


class BaseFOV(ABC):
    # NOTE: not using the `data` method from ngff Position

    @property
    @abstractmethod
    def channels(self) -> list[str]:
        raise NotImplementedError

    def channel_index(self, key: str) -> int:
        """Return index of given channel."""
        return self.channels.index(key)

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int, int, int, int]:
        # NOTE: suggestion, fix dimension to 5?
        # We could me more restrictive than ome-zarr
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
    @abstractmethod
    def ndim(self) -> int:
        raise NotImplementedError

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
