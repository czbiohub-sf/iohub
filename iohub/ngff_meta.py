"""
Dataclasses for OME-NGFF metadata.
Developed against OME-NGFF v0.4 and ome-zarr v0.6
"""

from abc import ABC
from pydantic.dataclasses import dataclass
from pydantic import validator, Field

from typing import Optional, Literal, List, ClassVar, TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import StrPath


@dataclass
class Axis:
    """https://ngff.openmicroscopy.org/0.4/index.html#axes-md"""

    # MUST
    name: str
    # SHOULD
    type: Optional[Literal["space", "time", "channel"]]
    # SHOULD
    unit: Optional[str]
    # store constants as class variables
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
    def valid_unit(cls, v, values: dict):
        if values.get("type") and v is not None:
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
class Transformation:
    """https://ngff.openmicroscopy.org/0.4/index.html#trafo-md"""

    # MUST
    type: Literal["identity", "translation", "scale"] = "identity"
    # MUST? (keyword not found in spec for the fields below)
    translation: Optional[List[float]]
    scale: Optional[List[float]]
    path: Optional[str]

    @validator("type")
    def unique_tranformation(cls, v, values: dict):
        count = bool(values.get(v)) + bool(values.get("path"))
        if v == "identity" and count != 0:
            raise ValueError(
                "The identity transformation should not be specified!"
            )
        elif count != 1:
            raise ValueError(
                f"Only one of the {v} list and the path should be specified!"
            )


@dataclass
class Dataset:
    """https://ngff.openmicroscopy.org/0.4/index.html#multiscale-md"""

    # MUST
    path: StrPath
    # MUST
    coordinate_transformations: List[Transformation] = Field(
        None, alias="coordinateTransformations"
    )


@dataclass
class MultiScales:
    """https://ngff.openmicroscopy.org/0.4/index.html#multiscale-md"""

    # SHOULD
    version: Optional[Literal["0.1", "0.2", "0.3", "0.4"]] = "0.4"
    # SHOULD
    name: Optional[str] = ""
    # MUST
    axes: List[Axis]
    # MUST
    datasets: List[Dataset]
    # MAY
    coordinate_transformations: Optional[List[Transformation]] = Field(
        None, alias="coordinateTransformations"
    )
    # SHOULD, describes the downscaling method (e.g. 'gaussian')
    type: Optional[str]
    # SHOULD, additional information about the downscaling method
    metadata: Optional[dict]
