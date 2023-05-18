"""
Data model classes with validation for OME-NGFF metadata.
Developed against OME-NGFF v0.4 and ome-zarr v0.6

Attributes are 'snake_case' with aliases to match NGFF names in JSON output.
See https://ngff.openmicroscopy.org/0.4/index.html#naming-style
about 'camelCase' inconsistency.
"""

import re
from typing import Any, ClassVar, Literal, Optional, TypedDict, Union

import pandas as pd
from pydantic import BaseModel, Field, root_validator, validator
from pydantic.color import Color, ColorType


def unique_validator(
    data: list[Union[BaseModel, TypedDict]], field: Union[str, list[str]]
):
    """Called by validators to ensure the uniqueness of certain fields.

    Parameters
    ----------
    data : list[Union[BaseModel, TypedDict]]
        list of pydantic models or typed dictionaries
    field : Union[str, list[str]]
        field(s) of the dataclass that must be unique

    Raises
    ------
    ValueError
        raised if any value is not unique
    """
    fields = [field] if isinstance(field, str) else field
    if not isinstance(data[0], dict):
        data = [d.dict() for d in data]
    df = pd.DataFrame(data)
    for key in fields:
        if not df[key].is_unique:
            raise ValueError(f"'{key}' must be unique!")


def alpha_numeric_validator(data: str):
    """Called by validators to ensure that strings are alpha-numeric.

    Parameters
    ----------
    data : str
        string to check

    Raises
    ------
    ValueError
        raised if the string contains characters other than [a-zA-z0-9]
    """
    if not (data.isalnum() or data.isnumeric()):
        raise ValueError(
            f"The path name must be alphanumerical! Got: '{data}'."
        )


TO_DICT_SETTINGS = dict(exclude_none=True, by_alias=True)


class MetaBase(BaseModel):
    class Config:
        allow_population_by_field_name = True


class AxisMeta(MetaBase):
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


class TransformationMeta(MetaBase):
    """https://ngff.openmicroscopy.org/0.4/index.html#trafo-md"""

    # MUST
    type: Literal["identity", "translation", "scale"]
    # MUST? (keyword not found in spec for the fields below)
    translation: Optional[list[float]] = None
    scale: Optional[list[float]] = None
    path: Optional[str] = None

    @root_validator
    def no_extra_method(cls, values: dict):
        count = sum([bool(v) for _, v in values.items()])
        if values["type"] == "identity" and count > 1:
            raise ValueError(
                "Method should not be specified for identity transformation!"
            )
        elif count > 2:
            raise ValueError(
                "Only one type of transformation method is allowed."
            )
        return values


class DatasetMeta(MetaBase):
    """https://ngff.openmicroscopy.org/0.4/index.html#multiscale-md"""

    # MUST
    path: str
    # MUST
    coordinate_transformations: list[TransformationMeta] = Field(
        alias="coordinateTransformations"
    )


class VersionMeta(MetaBase):
    """OME-NGFF spec version. Default is the current version (0.4)."""

    # SHOULD
    version: Optional[Literal["0.1", "0.2", "0.3", "0.4"]]


class MultiScaleMeta(VersionMeta):
    """https://ngff.openmicroscopy.org/0.4/index.html#multiscale-md"""

    # MUST
    axes: list[AxisMeta]
    # MUST
    datasets: list[DatasetMeta]
    # SHOULD
    name: Optional[str] = None
    # MAY
    coordinate_transformations: Optional[list[TransformationMeta]] = Field(
        alias="coordinateTransformations"
    )
    # SHOULD, describes the downscaling method (e.g. 'gaussian')
    type: Optional[str] = None
    # SHOULD, additional information about the downscaling method
    metadata: Optional[dict] = None

    @validator("axes")
    def unique_name(cls, v):
        unique_validator(v, "name")
        return v

    def get_dataset_paths(self):
        return [dataset.path for dataset in self.datasets]


class WindowDict(TypedDict):
    """Dictionary of contrast limit settings"""

    start: float
    end: float
    min: float
    max: float


class ChannelMeta(MetaBase):
    """Channel display settings without clear documentation from the NGFF spec.
    https://docs.openmicroscopy.org/omero/5.6.1/developers/Web/WebGateway.html#imgdata
    """

    active: bool = False
    coefficient: float = 1.0
    color: ColorType = "FFFFFF"
    family: str = "linear"
    inverted: bool = False
    label: str = None
    window: WindowDict = None


class RDefsMeta(MetaBase):
    """Rendering settings without clear documentation from the NGFF spec.
    https://docs.openmicroscopy.org/omero/5.6.1/developers/Web/WebGateway.html#imgdata
    """

    default_t: int = Field(alias="defaultT")
    default_z: int = Field(alias="defaultZ")
    model: str = "color"
    projection: Optional[str] = "normal"


class OMEROMeta(VersionMeta):
    """https://ngff.openmicroscopy.org/0.4/index.html#omero-md"""

    id: int
    name: Optional[str]
    channels: Optional[list[ChannelMeta]]
    rdefs: Optional[RDefsMeta]


class ImagesMeta(MetaBase):
    """Metadata needed for 'Images' (or positions/FOVs) in an OME-NGFF dataset.
    https://ngff.openmicroscopy.org/0.4/index.html#image-layout"""

    multiscales: list[MultiScaleMeta]
    omero: OMEROMeta


class LabelsMeta(MetaBase):
    """https://ngff.openmicroscopy.org/0.4/index.html#labels-md"""

    # SHOULD? (keyword not found in spec)
    labels: str
    # unlisted groups MAY be labels


class LabelColorMeta(MetaBase):
    """https://ngff.openmicroscopy.org/0.4/index.html#label-md"""

    # MUST
    label_value: int = Field(alias="label-value")
    # MAY
    rgba: Optional[ColorType] = None

    @validator("rgba")
    def rgba_color(cls, v):
        v = Color(v).as_rgb_tuple(alpha=True)
        return v


class ImageLabelMeta(VersionMeta):
    """https://ngff.openmicroscopy.org/0.4/index.html#label-md"""

    # SHOULD
    colors: list[LabelColorMeta]
    # MAY
    properties: list[dict[str, Any]]
    # MAY
    source: dict[str, Any]

    @validator("colors", "properties")
    def unique_label_value(cls, v):
        # MUST
        unique_validator(v, "label_value")
        return v


class AcquisitionMeta(MetaBase):
    """https://ngff.openmicroscopy.org/0.4/index.html#plate-md"""

    # MUST
    id: int
    # SHOULD
    name: Optional[str] = None
    # SHOULD
    maximum_field_count: Optional[int] = Field(alias="maximumfieldcount")
    # MAY
    description: Optional[str] = None
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
        return v

    @validator("end_time")
    def end_after_start(cls, v: int, values: dict):
        # CUSTOM
        if st := values.get("start_time"):
            if st > v:
                raise ValueError(
                    f"Start timestamp {st} is larger than end timestamp {v}."
                )
        return v


class PlateAxisMeta(MetaBase):
    """OME-NGFF metadata for a row or a column on a multi-well plate.
    https://ngff.openmicroscopy.org/0.4/index.html#plate-md"""

    # MUST
    name: str

    @validator("name")
    def alpha_numeric(cls, v: str):
        # MUST
        alpha_numeric_validator(v)
        return v


class WellIndexMeta(MetaBase):
    """OME-NGFF metadata for a well on a multi-well plate.
    https://ngff.openmicroscopy.org/0.4/index.html#plate-md"""

    path: str
    row_index: int = Field(alias="rowIndex")
    column_index: int = Field(alias="columnIndex")

    @validator("path")
    def row_slash_column(cls, v: str):
        # MUST
        # regex: one line that is exactly two words separated by one '/'
        if len(re.findall(r"^\w+\/\w+$", v)) != 1:
            raise ValueError(
                f"The well path '{v}' is not in the form of 'row/column'!"
            )
        return v

    @validator("row_index", "column_index")
    def geq_zero(cls, v: int):
        # MUST
        if v < 0:
            raise ValueError("Well position indices must not be negative!")
        return v


class PlateMeta(VersionMeta):
    """OME-NGFF high-content screening plate metadata.
    https://ngff.openmicroscopy.org/0.4/index.html#plate-md"""

    # SHOULD
    name: Optional[str]
    # MAY
    acquisitions: Optional[list[AcquisitionMeta]]
    # MUST
    rows: list[PlateAxisMeta]
    # MUST
    columns: list[PlateAxisMeta]
    # MUST
    wells: list[WellIndexMeta]
    # SHOULD
    field_count: Optional[int]

    @validator("acquisitions")
    def unique_id(cls, v):
        # MUST
        unique_validator(v, "id")
        return v

    @validator("rows", "columns")
    def unique_name(cls, v):
        # MUST
        unique_validator(v, "name")
        return v

    @validator("wells")
    def unique_well(cls, v):
        # CUSTOM
        unique_validator(v, "path")
        return v

    @validator("field_count")
    def positive(cls, v):
        # MUST
        if v <= 0:
            raise ValueError("Field count must be a positive integer!")
        return v


class ImageMeta(MetaBase):
    """Image metadata field under an HCS well group.
    https://ngff.openmicroscopy.org/0.4/index.html#well-md"""

    # MUST if `PlateMeta.acquisitions` contains multiple acquisitions
    acquisition: Optional[int]
    # MUST
    path: str

    @validator("path")
    def alpha_numeric(cls, v):
        # MUST
        alpha_numeric_validator(v)
        return v


class WellGroupMeta(VersionMeta):
    """OME-NGFF high-content screening well group metadata.
    https://ngff.openmicroscopy.org/0.4/index.html#well-md"""

    # MUST
    images: list[ImageMeta]
