from .context_anchored_tile_refine.node import (
    ContextAnchoredTileRefine,
    ContextAnchoredTileRefineVL,
    ContextAnchoredTileUpscaleVL,
)
from .context_anchored_tile_refine.testing import (
    ContextAnchoredTileTestCaptions,
    ContextAnchoredTileTestLayout,
    ContextAnchoredTileTestRender,
    ContextAnchoredTileTestSettings,
    ContextAnchoredTileTestUpscale,
)

NODE_CLASS_MAPPINGS = {
    "ContextAnchoredTileRefine": ContextAnchoredTileRefine,
    "ContextAnchoredTileRefineVL": ContextAnchoredTileRefineVL,
    "ContextAnchoredTileUpscaleVL": ContextAnchoredTileUpscaleVL,
    "ContextAnchoredTileTestSettings": ContextAnchoredTileTestSettings,
    "ContextAnchoredTileTestLayout": ContextAnchoredTileTestLayout,
    "ContextAnchoredTileTestUpscale": ContextAnchoredTileTestUpscale,
    "ContextAnchoredTileTestCaptions": ContextAnchoredTileTestCaptions,
    "ContextAnchoredTileTestRender": ContextAnchoredTileTestRender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ContextAnchoredTileRefine": "Context-Anchored Tile Refine",
    "ContextAnchoredTileRefineVL": "Context-Anchored Tile Refine (VL)",
    "ContextAnchoredTileUpscaleVL": "Context-Anchored Tile Upscale (VL)",
    "ContextAnchoredTileTestSettings": "Tile Test: Settings",
    "ContextAnchoredTileTestLayout": "Tile Test: Layout",
    "ContextAnchoredTileTestUpscale": "Tile Test: Upscale",
    "ContextAnchoredTileTestCaptions": "Tile Test: Captions",
    "ContextAnchoredTileTestRender": "Tile Test: Render",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
