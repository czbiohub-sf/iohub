"""
Dataclasses with validation for OME-NGFF metadata.
Developed against OME-NGFF v0.4 and ome-zarr v0.6

Attributes are 'snake_case' with aliases to match NGFF names.
See https://ngff.openmicroscopy.org/0.4/index.html#naming-style about 'camelCase' inconsistency.
"""

from abc import ABC
from pydantic import validator, Field
from pydantic.dataclasses import dataclass
from pydantic.color import Color

from typing import (
    Optional,
    Any,
    Literal,
    List,
    Dict,
    TypedDict,
    ClassVar,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from _typeshed import StrPath


@dataclass
class AxisMeta:
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
class TransformationMeta:
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
class DatasetMeta:
    """https://ngff.openmicroscopy.org/0.4/index.html#multiscale-md"""

    # MUST
    path: StrPath
    # MUST
    coordinate_transformations: List[TransformationMeta] = Field(
        alias="coordinateTransformations"
    )


@dataclass
class VersionMeta:
    """OME-NGFF spec version. Default is the current version (0.4)."""

    version: Optional[Literal["0.1", "0.2", "0.3", "0.4"]] = "0.4"


@dataclass
class MultiScalesMeta(VersionMeta):
    """https://ngff.openmicroscopy.org/0.4/index.html#multiscale-md"""

    # SHOULD
    name: Optional[str] = ""
    # MUST
    axes: List[AxisMeta]
    # MUST
    datasets: List[DatasetMeta]
    # MAY
    coordinate_transformations: Optional[List[TransformationMeta]] = Field(
        alias="coordinateTransformations"
    )
    # SHOULD, describes the downscaling method (e.g. 'gaussian')
    type: Optional[str]
    # SHOULD, additional information about the downscaling method
    metadata: Optional[dict]


class WindowDict(TypedDict):
    """Dictionary of contrast limit settings"""

    start: float
    end: float
    min: float
    max: float


@dataclass
class ChannelMeta:
    """Channel display settings without clear documentation from the NGFF spec.
    https://docs.openmicroscopy.org/omero/5.6.1/developers/Web/WebGateway.html#imgdata"""

    active: bool = False
    coefficient: float = 1.0
    color: Color = Color("FFFFFF")
    family: str = "linear"
    inverted: bool = False
    label: str
    window: WindowDict

    class Config:
        json_encoders = {Color: lambda c: c.as_hex()}


@dataclass
class OMEROMeta(VersionMeta):
    """https://ngff.openmicroscopy.org/0.4/index.html#omero-md"""

    id: int
    name: Optional[str]
    channels: List[ChannelMeta]


@dataclass
class LabelsMeta:
    """https://ngff.openmicroscopy.org/0.4/index.html#labels-md"""

    # SHOULD? (keyword not found in spec)
    labels: str
    # unlisted groups MAY be labels


@dataclass
class LabelColorMeta:
    """https://ngff.openmicroscopy.org/0.4/index.html#label-md"""

    # MUST
    label_value: int = Field(alias="label-value")
    # MAY
    rgba: Color

    class Config:
        # MUST
        json_encoders = {Color: lambda c: c.as_rgb_tuple(alpha=True)}


@dataclass
class ImageLabelMeta(VersionMeta):
    """https://ngff.openmicroscopy.org/0.4/index.html#label-md"""

    # SHOULD
    colors: List[LabelColorMeta]
    # MAY
    properties: List[Dict[str, Any]]
    # MAY
    source: Dict[str, Any]


@dataclass
class AcquisitionMeta:
    """https://ngff.openmicroscopy.org/0.4/index.html#plate-md"""

    # MUST
    id: int
    # SHOULD
    name: Optional[str]
    # SHOULD
    maximum_field_count: Optional[int] = Field(alias="maximumfieldcount")

    @validator("id", "maximum_field_count")
    def geq_zero(cls, v):
        if v < 0:
            raise ValueError(
                "Integer identifier must be equal or greater to zero!"
            )


@dataclass
class PlateMeta:
    """https://ngff.openmicroscopy.org/0.4/index.html#plate-md"""

    # MAY
    acquisitions: Optional[List[dict]]
    #
