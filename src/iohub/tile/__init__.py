"""iohub.tile — Tile, process, and reassemble large image volumes."""

from iohub._experimental import ExperimentalWarning
from iohub.tile._blenders import (
    BlendContext,
    Blender,
    DistanceBlender,
    GaussianBlender,
    UniformBlender,
    get_blender,
)
from iohub.tile._compositors import (
    CompositeContext,
    Compositor,
    FirstCompositor,
    MaxCompositor,
    MeanCompositor,
    get_compositor,
)
from iohub.tile._registry import register_strategy
from iohub.tile._resolvers import (
    LayoutResolver,
    StitchingYAMLResolver,
    TransformResolver,
)
from iohub.tile._tiler import SamplingMode, Tile, Tiler
from iohub.tile.tile import (
    CacheMode,
    apply_func_tiled,
    create_tile_store,
    process_tiles,
    stitch_from_store,
)

__all__ = [
    "BlendContext",
    "Blender",
    "CacheMode",
    "CompositeContext",
    "Compositor",
    "DistanceBlender",
    "ExperimentalWarning",
    "FirstCompositor",
    "GaussianBlender",
    "LayoutResolver",
    "MaxCompositor",
    "MeanCompositor",
    "SamplingMode",
    "Tile",
    "Tiler",
    "StitchingYAMLResolver",
    "TransformResolver",
    "UniformBlender",
    "apply_func_tiled",
    "create_tile_store",
    "get_blender",
    "get_compositor",
    "process_tiles",
    "register_strategy",
    "stitch_from_store",
]
