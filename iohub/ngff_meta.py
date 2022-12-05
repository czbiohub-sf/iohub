"""
Dataclasses for OME-NGFF metadata.
Developed against OME-NGFF v0.4 and ome-zarr v0.6
"""

from abc import ABC
from pydantic.dataclasses import dataclass
from pydantic import validator

from typing import Optional, Literal, List, ClassVar


@dataclass
class Axis:
    name: str
    type: Optional[Literal["space", "time", "channel"]] = None
    unit: Optional[str] = None

    SPACE_UNITS: ClassVar[set] = {
        "angstrom",
        "attometer",
        "centimeter",
        "decimeter",
        "exameter",
        "femtometer",
        "foot",
        "gigameter",
        "hectometer",
        "inch",
        "kilometer",
        "megameter",
        "meter",
        "micrometer",
        "mile",
        "millimeter",
        "nanometer",
        "parsec",
        "petameter",
        "picometer",
        "terameter",
        "yard",
        "yoctometer",
        "yottameter",
        "zeptometer",
        "zettameter",
    }

    TIME_UNITS: ClassVar[set] = {
        "attosecond",
        "centisecond",
        "day",
        "decisecond",
        "exasecond",
        "femtosecond",
        "gigasecond",
        "hectosecond",
        "hour",
        "kilosecond",
        "megasecond",
        "microsecond",
        "millisecond",
        "minute",
        "nanosecond",
        "petasecond",
        "picosecond",
        "second",
        "terasecond",
        "yoctosecond",
        "yottasecond",
        "zeptosecond",
        "zettasecond",
    }

    @validator("unit")
    def valid_unit(cls, v, values):
        if "type" in values.keys() and v is not None:
            if values["type"] == "channel":
                raise ValueError(
                    f"Channel axis must not have a unit! Got unit: {v}."
                )
            if values["type"] == "space" and v not in cls.SPACE_UNITS:
                raise ValueError(
                    f"Got invalid space unit: '{v}' not in {cls.SPACE_UNITS}"
                )
            if values["type"] == "time" and v not in cls.TIME_UNITS:
                raise ValueError(
                    f"Got invalid time unit: '{v}' not in {cls.TIME_UNITS}"
                )
        return v


@dataclass
class Node(ABC):
    axes: List[Axis]
