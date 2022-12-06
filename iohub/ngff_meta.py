"""
Dataclasses with validation for OME-NGFF metadata.
Developed against OME-NGFF v0.4 and ome-zarr v0.6

Attributes are 'snake_case' with aliases to match NGFF names.
See https://ngff.openmicroscopy.org/0.4/index.html#naming-style about 'camelCase' inconsistency.
"""

import re
from pydantic import validator, Field
from pydantic.dataclasses import dataclass, Dataclass
from pydantic.color import Color, ColorTuple
import pandas as pd

from typing import (
    Optional,
    Union,
    Literal,
    Any,
    List,
    Dict,
    TypedDict,
    ClassVar,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from _typeshed import StrPath


def unique_validator(
    data: List[Union[Dataclass, TypedDict]], field: Union[str, List[str]]
):
    """Called by validators to ensure the uniqueness of certain fields.

    Parameters
    ----------
    data : List[Union[Dataclass, TypedDict]]
        list of dataclass instances or typed dictionaries
    field : Union[str, List[str]]
        field(s) of the dataclass that must be unique

    Raises
    ------
    ValueError
        raised if any value is not unique
    """
    fields = [field] if isinstance(field, str) else field
    df = pd.DataFrame(data)
    for key in fields:
        if not df[key].is_unique:
            raise ValueError(f"'{key}' must be unique!")


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

    # SHOULD
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

    @validator("axes")
    def unique_name(cls, v):
        unique_validator(v, "name")


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
class RDefsMeta:
    """Rendering settings without clear documentation from the NGFF spec.
    https://docs.openmicroscopy.org/omero/5.6.1/developers/Web/WebGateway.html#imgdata"""

    default_t: int = Field(alias="defaultT")
    default_z: int = Field(alias="defaultZ")
    model: str = "color"
    projection: Optional[str] = "normal"


@dataclass
class OMEROMeta(VersionMeta):
    """https://ngff.openmicroscopy.org/0.4/index.html#omero-md"""

    id: int
    name: Optional[str]
    channels: List[ChannelMeta]
    rdefs: RDefsMeta


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
    rgba: ColorTuple

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

    @validator("colors", "properties")
    def unique_label_value(cls, v):
        # MUST
        unique_validator(v, "label_value")


@dataclass
class AcquisitionMeta:
    """https://ngff.openmicroscopy.org/0.4/index.html#plate-md"""

    # MUST
    id: int
    # SHOULD
    name: Optional[str]
    # SHOULD
    maximum_field_count: Optional[int] = Field(alias="maximumfieldcount")
    # MAY
    description: Optional[str]
    # MAY
    start_time: Optional[int] = Field(alias="starttime")
    # MAY
    end_time: Optional[int] = Field(alias="endtime")

    @validator("id", "maximum_field_count", "start_time", "end_time")
    def geq_zero(cls, v):
        # MUST
        if v < 0:
            raise ValueError(
                "The integer value must be equal or greater to zero!"
            )

    @validator("end_time")
    def end_after_start(cls, v: int, values: dict):
        # CUSTOM
        st = values.get("start_time")
        if st:
            if st > v:
                raise ValueError(
                    f"The start timestamp {st} should not be larger than the end timestamp {v}"
                )


@dataclass
class PlateAxisMeta:
    """OME-NGFF metadata for a row or a column on a multi-well plate.
    https://ngff.openmicroscopy.org/0.4/index.html#plate-md"""

    # MUST
    name: str

    @validator("name")
    def alpha_numeric(cls, v: str):
        # MUST
        if not (v.isalnum() or v.isnumeric()):
            raise ValueError(
                f"The column name must be alphanumerical! Got invalid value: '{v}'."
            )


@dataclass
class WellMeta:
    """OME-NGFF metadata for a well on a multi-well plate.
    https://ngff.openmicroscopy.org/0.4/index.html#plate-md"""

    path: StrPath
    row_index: int = Field(alias="rowIndex")
    column_index: int = Field(alias="columnIndex")

    @validator("path")
    def row_slash_column(cls, v: str):
        # MUST
        # regex: one line that is exactly two words separated by one forward slash
        if len(re.findall(r"^\w\/\w$", v)) != 1:
            raise ValueError(
                f"The well path '{v}' is not in the form of 'row/column'!"
            )

    @validator("row_index", "column_index")
    def geq_zero(cls, v: int):
        # MUST
        if v < 0:
            raise ValueError("Well position indices must not be negative!")


@dataclass
class PlateMeta(VersionMeta):
    """OME-NGFF high-content screening plate metadata.
    https://ngff.openmicroscopy.org/0.4/index.html#plate-md"""

    # SHOULD
    name: Optional[str]
    # MAY
    acquisitions: Optional[List[AcquisitionMeta]]
    # MUST
    rows: List[PlateAxisMeta]
    # MUST
    columns: List[PlateAxisMeta]
    # MUST
    wells: List[WellMeta]
    # SHOULD
    field_count: Optional[int]

    @validator("acquisitions")
    def unique_id(cls, v):
        # MUST
        unique_validator(v, "id")

    @validator("rows", "columns")
    def unique_name(cls, v):
        # MUST
        unique_validator(v, "name")

    @validator("wells")
    def unique_well(cls, v):
        # CUSTOM
        unique_validator(v, ["path", "row_index", "column_index"])

    @validator("field_count")
    def positive(cls, v):
        # MUST
        if v <= 0:
            raise ValueError("Field count must be a positive integer!")
