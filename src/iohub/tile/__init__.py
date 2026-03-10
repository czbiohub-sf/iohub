"""iohub.tile — Tile, process, and reassemble large image volumes."""

from iohub._experimental import ExperimentalWarning
from iohub.tile._assembler import Assembler
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
from iohub.tile._slicer import SamplingMode, Slicer, TileSpec
from iohub.tile.tile import CacheMode, apply_func_tiled, tile_and_assemble

__all__ = [
    "Assembler",
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
    "Slicer",
    "StitchingYAMLResolver",
    "TileSpec",
    "TransformResolver",
    "UniformBlender",
    "get_blender",
    "get_compositor",
    "apply_func_tiled",
    "register_strategy",
    "tile_and_assemble",
]
