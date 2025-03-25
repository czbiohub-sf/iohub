from __future__ import annotations

"""
Data model classes with validation for OME-NGFF metadata.
Developed against OME-NGFF v0.4 and ome-zarr v0.9

Attributes are 'snake_case' with aliases to match NGFF names in JSON output.
See https://ngff.openmicroscopy.org/0.4/index.html#naming-style
about 'camelCase' inconsistency.
"""

import re
from typing import Annotated, Any, Literal, Optional

import pandas as pd
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)
from pydantic_extra_types.color import Color, ColorType

# TODO: remove when drop Python < 3.12
from typing_extensions import Self, TypedDict


def unique_validator(
    data: list[BaseModel], field: str | list[str]
) -> list[BaseModel]:
    """Called by validators to ensure the uniqueness of certain fields.

    Parameters
    ----------
    data : list[BaseModel]
        list of pydantic models or typed dictionaries
    field : str | list[str]
        field(s) of the dataclass that must be unique

    Returns
    -------
    list[BaseModel]
        valid input data

    Raises
    ------
    ValueError
        raised if any value is not unique
    """
    fields = [field] if isinstance(field, str) else field
    if not isinstance(data[0], dict):
        params = [d.model_dump() for d in data]
    df = pd.DataFrame(params)
    for key in fields:
        if not df[key].is_unique:
            raise ValueError(f"'{key}' must be unique!")
    return data


def alpha_numeric_validator(data: str) -> str:
    """Called by validators to ensure that strings are alpha-numeric.

    Parameters
    ----------
    data : str
        string to check

    Returns
    -------
    str
        valid input data

    Raises
    ------
    ValueError
        raised if the string contains characters other than [a-zA-z0-9]
    """
    if not data.isalnum():
        raise ValueError(
            f"The path name must be alphanumerical! Got: '{data}'."
        )
    return data


TO_DICT_SETTINGS = dict(exclude_none=True, by_alias=True)


class MetaBase(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class NamedAxisMeta(MetaBase):
    name: str


class ChannelAxisMeta(NamedAxisMeta):
    type: Literal["channel"] = "channel"


class SpaceAxisMeta(NamedAxisMeta):
    type: Literal["space"] = "space"
    unit: (
        Literal[
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
        ]
        | None
    ) = None


class TimeAxisMeta(NamedAxisMeta):
    type: Literal["time"] = "time"
    unit: (
        Literal[
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
        ]
        | None
    ) = None


"""https://ngff.openmicroscopy.org/0.4/index.html#axes-md"""
AxisMeta = Annotated[
    TimeAxisMeta | ChannelAxisMeta | SpaceAxisMeta,
    Discriminator("type"),
]


class TransformationMeta(MetaBase):
    """https://ngff.openmicroscopy.org/0.4/index.html#trafo-md"""

    # MUST
    type: Literal["identity", "translation", "scale"]
    # MUST? (keyword not found in spec for the fields below)
    translation: list[float] | None = None
    scale: list[float] | None = None
    path: Annotated[str, Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def no_extra_method(self) -> Self:
        methods = sum(
            bool(m is not None)
            for m in [self.translation, self.scale, self.path]
        )
        if self.type == "identity" and methods > 0:
            raise ValueError(
                "Method should not be specified for identity transformation!"
            )
        elif self.translation and self.scale:
            raise ValueError(
                "'translation' and 'scale' cannot be provided "
                f"in the same `{type(self).__name__}`!"
            )
        return self


class DatasetMeta(MetaBase):
    """https://ngff.openmicroscopy.org/0.4/index.html#multiscale-md"""

    # MUST
    path: str
    # MUST
    coordinate_transformations: list[TransformationMeta] = Field(
        alias=str("coordinateTransformations")
    )


class VersionMeta(MetaBase):
    """OME-NGFF spec version. Default is the current version (0.4)."""

    # SHOULD
    version: Literal["0.1", "0.2", "0.3", "0.4"] = "0.4"


class MultiScaleMeta(VersionMeta):
    """https://ngff.openmicroscopy.org/0.4/index.html#multiscale-md"""

    # MUST
    axes: list[AxisMeta]
    # MUST
    datasets: list[DatasetMeta]
    # SHOULD
    name: str | None = None
    # MAY
    coordinate_transformations: list[TransformationMeta] | None = Field(
        alias=str("coordinateTransformations"), default=None
    )
    # SHOULD, describes the downscaling method (e.g. 'gaussian')
    type: str | None = None
    # SHOULD, additional information about the downscaling method
    metadata: dict | None = None

    @field_validator("axes")
    @classmethod
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
    label: str | None = None
    window: WindowDict | None = None


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

    id: int | None = None
    name: str | None = None
    channels: list[ChannelMeta] | None = None
    rdefs: RDefsMeta | None = None


class ImagesMeta(MetaBase):
    """Metadata needed for 'Images' (or positions/FOVs) in an OME-NGFF dataset.
    https://ngff.openmicroscopy.org/0.4/index.html#image-layout"""

    multiscales: list[MultiScaleMeta]
    # transitional, optional
    omero: OMEROMeta | None = None


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
    rgba: ColorType | None = None

    @field_validator("rgba")
    @classmethod
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

    @field_validator("colors", "properties")
    @classmethod
    def unique_label_value(cls, v):
        # MUST
        unique_validator(v, "label_value")
        return v


class AcquisitionMeta(MetaBase):
    """https://ngff.openmicroscopy.org/0.4/index.html#plate-md"""

    # MUST
    id: NonNegativeInt
    # SHOULD
    name: str | None = None
    # SHOULD
    maximum_field_count: PositiveInt | None = Field(
        alias="maximumfieldcount", default=None
    )
    # MAY
    description: str | None = None
    # MAY
    start_time: NonNegativeInt | None = Field(
        alias=str("starttime"), default=None
    )
    # MAY
    end_time: NonNegativeInt | None = Field(alias=str("endtime"), default=None)

    @model_validator(mode="after")
    def end_after_start(self) -> Self:
        if self.start_time is not None and self.end_time is not None:
            if self.start_time > self.end_time:
                raise ValueError(
                    "The acquisition end time must be after the start time!"
                )
        return self


class PlateAxisMeta(MetaBase):
    """OME-NGFF metadata for a row or a column on a multi-well plate.
    https://ngff.openmicroscopy.org/0.4/index.html#plate-md"""

    # MUST
    name: Annotated[str, AfterValidator(alpha_numeric_validator)]


class WellIndexMeta(MetaBase):
    """OME-NGFF metadata for a well on a multi-well plate.
    https://ngff.openmicroscopy.org/0.4/index.html#plate-md"""

    path: str
    row_index: NonNegativeInt = Field(alias="rowIndex")
    column_index: NonNegativeInt = Field(alias="columnIndex")

    @field_validator("path")
    @classmethod
    def row_slash_column(cls, v: str):
        # MUST
        # regex: one line that is exactly two words separated by one '/'
        if len(re.findall(r"^\w+\/\w+$", v)) != 1:
            raise ValueError(
                f"The well path '{v}' is not in the form of 'row/column'!"
            )
        return v


class PlateMeta(VersionMeta):
    """OME-NGFF high-content screening plate metadata.
    https://ngff.openmicroscopy.org/0.4/index.html#plate-md"""

    # SHOULD
    name: str | None = None
    # MAY
    acquisitions: list[AcquisitionMeta] | None = None
    # MUST
    rows: list[PlateAxisMeta]
    # MUST
    columns: list[PlateAxisMeta]
    # MUST
    wells: list[WellIndexMeta]
    # SHOULD
    field_count: PositiveInt | None = None

    @field_validator("acquisitions")
    @classmethod
    def unique_id(cls, v):
        # MUST
        unique_validator(v, "id")
        return v

    @field_validator("rows", "columns")
    @classmethod
    def unique_name(cls, v):
        # MUST
        unique_validator(v, "name")
        return v

    @field_validator("wells")
    @classmethod
    def unique_well(cls, v):
        # CUSTOM
        unique_validator(v, "path")
        return v


class ImageMeta(MetaBase):
    """Image metadata field under an HCS well group.
    https://ngff.openmicroscopy.org/0.4/index.html#well-md"""

    # MUST if `PlateMeta.acquisitions` contains multiple acquisitions
    acquisition: int | None = None
    # MUST
    path: Annotated[str, AfterValidator(alpha_numeric_validator)]


class WellGroupMeta(VersionMeta):
    """OME-NGFF high-content screening well group metadata.
    https://ngff.openmicroscopy.org/0.4/index.html#well-md"""

    # MUST
    images: list[ImageMeta]
