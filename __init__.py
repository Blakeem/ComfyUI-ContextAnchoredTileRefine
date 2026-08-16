from .context_anchored_tile_refine.node import (
    ContextAnchoredTileRefine,
    ContextAnchoredTileRefineVL,
    ContextAnchoredTileUpscaleVL,
)

NODE_CLASS_MAPPINGS = {
    "ContextAnchoredTileRefine": ContextAnchoredTileRefine,
    "ContextAnchoredTileRefineVL": ContextAnchoredTileRefineVL,
    "ContextAnchoredTileUpscaleVL": ContextAnchoredTileUpscaleVL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ContextAnchoredTileRefine": "Context-Anchored Tile Refine",
    "ContextAnchoredTileRefineVL": "Context-Anchored Tile Refine (VL)",
    "ContextAnchoredTileUpscaleVL": "Context-Anchored Tile Upscale (VL)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
