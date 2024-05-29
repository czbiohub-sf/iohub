from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union, cast

import numpy as np
import yaml

if TYPE_CHECKING:
    from _typeshed import StrOrBytesPath

    from iohub.clearcontrol import ArrayIndex


def _cached(f: Callable) -> Callable:
    """Decorator that caches the array data using its key."""

    @wraps(f)
    def _key_cache_wrapper(
        self: "DaXiFOV", key: tuple[int | None, ...]
    ) -> np.ndarray:
        if not self._cache:
            return f(self, key)

        elif key != self._cache_key:
            self._cache_array = f(self, key)
            self._cache_key = key

        return cast(np.ndarray, self._cache_array)

    return _key_cache_wrapper


class DaXiFOV:
    """
    Reader class for DaXi
    https://www.nature.com/articles/s41592-022-01417-2

    It provides an array-like API for DaXi timelapses
    that loads the volumes lazily.

    It assumes the channels and volumes have the same shape,
    when that is not the case the minimum value from each dimension is used.

    Parameters
    ----------
    data_path : StrOrBytesPath
        DaXi dataset path.
    missing_value : Optional[int], optional
        If provided this class won't raise an error when missing a volume and
        it will return an array with the provided value.
    cache : bool
        When true caches the last array using the first two indices as key.
    """

    _CHANNELS_KEY = "Laser wavelengths"
    _SHAPE_KEY = "Dataset dimension"
    _SHAPE_IDX = ("T", "V", "Z", "Y", "X")

    def __init__(
        self,
        data_path: "StrOrBytesPath",
        missing_value: Optional[int] = None,
        cache: bool = False,
    ):
        super().__init__()

        self._data_path = Path(data_path)

        with open(self._data_path / "metadata.yaml") as f:
            self._metadata = yaml.safe_load(f)

        self._missing_value = missing_value
        self._dtype = np.uint16
        self._cache = cache
        self._cache_key: tuple[int | None, ...] | None = None
        self._cache_array: np.ndarray | None = None

        self._channels = self._metadata[self._CHANNELS_KEY]

        shape_dict = self._metadata[self._SHAPE_KEY]

        self._raw_shape = tuple(shape_dict[k] for k in self._SHAPE_IDX)

        shape_dict["Z"] //= len(self._channels)
        shape_dict["V"] *= len(self._channels)

        self._shape = tuple(shape_dict[k] for k in self._SHAPE_IDX)

    def _volume_path(self, t: int, v: int) -> Path:
        z, y, x = self._raw_shape[2:]
        volume_path = (
            self._data_path / f"T_{int(t)}.V_{int(v)}.({z}x{y}x{x}).raw"
        )
        if not volume_path.exists():
            raise ValueError(f"Volume not found: {volume_path}")
        return volume_path

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Timelapse shape
        """
        return self._shape

    @property
    def channels(self) -> list[str]:
        """Return sorted channels name."""
        return self._channels

    @_cached
    def _load_array(
        self, key: tuple[int, int, int | None, int | None]
    ) -> np.ndarray:
        """
        Loads a single or multiple channels from DaXi raw.

        Parameters
        ----------
        key : tuple[int, int, int | None, int | None]
            Time point, view, channel_index, and z-index.
            z-index and are optional

        Returns
        -------
        np.ndarray
            Volume as an array can be single or multiple channels.

        Raises
        ------
        ValueError
            When expected volume path not found.
        """
        shape = list(self._shape[-2:])

        time_point, view, channel, z_index = key

        len_ch = len(self._channels)
        view //= len_ch

        if channel is None:
            slicing = slice(None)
            shape.insert(0, len_ch)
        else:
            slicing = slice(channel, None, len_ch)
            len_ch = 1

        if z_index is None:
            shape.insert(0, self._shape[2])  # z-shape
        else:
            z = z_index * len(self._channels)
            if channel is not None:
                slicing = z + channel
            else:
                slicing = slice(z, z + len(self._channels))

        map_arr = np.memmap(
            self._volume_path(time_point, view),
            dtype=self._dtype,
            shape=self._raw_shape[2:],
            mode="r",
        )

        print("--------")
        print(key)
        print(slicing)

        # TODO: catch when volume is not complete
        arr = map_arr[slicing]
        print(arr.shape)
        arr = arr.reshape(shape)

        if len(shape) == 4:
            # ZCYX -> CZYX
            arr = arr.swapaxes(0, 1)

        return arr

    @staticmethod
    def _fix_indexing(indexing: "ArrayIndex", size: int) -> list[int] | int:
        """Converts numpy array to simple python type or list."""
        # TODO: check if necessary
        if isinstance(indexing, slice):
            return list(range(size)[indexing])

        elif np.isscalar(indexing):
            try:
                int_index: int = indexing.item()
            except AttributeError:
                int_index = indexing

            if int_index < 0:
                int_index += size

            return int_index

        elif isinstance(indexing, np.ndarray):
            return indexing.tolist()

        return indexing

    def _load_array_from_key(
        self, key: tuple[list[int] | int | None, ...]
    ) -> np.ndarray:
        """
        Load array from a key with multiple channel indices.
        This function is called recursively until int-only indices are found.
        """
        for i, k in enumerate(key):
            if isinstance(k, int):
                continue

            elif k is None:
                if i >= 2:
                    continue

                k = list(range(self._shape[i]))

            arrs = []
            for int_key in k:
                new_key = key[:i] + (int_key,) + key[i + 1 :]
                arrs.append(self._load_array_from_key(new_key))

            return np.stack(arrs)

        return self._load_array(key)

    def __getitem__(
        self, key: Union["ArrayIndex", tuple["ArrayIndex", ...]]
    ) -> np.ndarray:
        """Lazily load array as indexed.

        Parameters
        ----------
        key : ArrayIndex | tuple[ArrayIndex, ...]
            An indexing key as in numpy, but a bit more limited.

        Returns
        -------
        np.ndarray
            Output array.

        Raises
        ------
        NotImplementedError
            Not all numpy array of indexing are implemented.
        """
        # standardizing indexing
        yx_slicing = slice(None)
        min_size = 4

        if not isinstance(key, tuple):
            key = (key,)

        key = tuple(self._fix_indexing(k, s) for s, k in zip(self._shape, key))
        args_key = key + (None,) * (min_size - len(key))

        if len(args_key) > min_size:  # min_size + 1 (z)
            args_key, yx_slicing = args_key[:min_size], args_key[min_size:]

        return self._load_array_from_key(args_key)[yx_slicing]

    def __setitem__(self, key: Any, value: Any) -> None:
        raise PermissionError("DaXiFOV is read-only.")

    @property
    def ndim(self) -> int:
        return 5

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def cache(self) -> bool:
        return self._cache

    @cache.setter
    def cache(self, value: bool) -> None:
        """Free current key/array cache."""
        self._cache = value
        if not value:
            self._cache_array = None
            self._cache_key = None

    def metadata(self) -> dict[str, Any]:
        """Summarizes Clear Control metadata into a dictionary."""
        return self._metadata

    @property
    def scale(self) -> list[float]:
        """Dataset temporal, channel and spacial scales."""
        # TODO: use metadata
        return [
            1.0,
            1.0,
            1.24,
            0.439,
            0.439,
        ]


def create_mock_daxi_dataset(path: "StrOrBytesPath") -> None:
    """
    Creates a (2, 4, 64, 64, 64) Clear Control dataset of random integers.

    Parameters
    ----------
    path : StrOrBytesPath
        Dataset output path.
    """
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)

    rng = np.random.default_rng(0)
    array = rng.integers(
        0, 1_500, size=(2, 2, 2 * 64, 64, 64), dtype=np.uint16
    )

    metadata = {
        "Laser wavelenghts": ["488", "561"],
        "Dataset dimension": {
            "T": 2,
            "V": 2,
            "X": 64,
            "Y": 64,
            "Z": 2 * 64,
        },
    }
    with open(path / "metadata.yaml", "w+") as f:
        yaml.dump(metadata, f)

    for t in range(array.shape[0]):
        for v in range(array.shape[1]):
            out_path = path / f"T_{t}.V_{v}.(128x64x64).raw"
            raw_map = np.memmap(
                out_path, dtype=np.uint16, mode="w+", shape=array.shape[2:]
            )
            raw_map[...] = array[t, v]
            raw_map.flush()


if __name__ == "__main__":
    import napari

    path = Path("/mnt/royer.daxi2/Merlin/neurog1.h2afva_05_21_2024")
    ds = DaXiFOV(path, cache=True)

    napari.imshow(ds)
    napari.run()
