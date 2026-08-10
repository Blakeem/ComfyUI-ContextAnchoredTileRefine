from .context_anchored_tile_refine.node import ContextAnchoredTileRefine, ContextAnchoredTileRefineVL, ContextAnchoredTileUpscaleVL, ContextAnchoredTileUpscaleVLVideo

NODE_CLASS_MAPPINGS = {
    "ContextAnchoredTileRefine": ContextAnchoredTileRefine,
    "ContextAnchoredTileRefineVL": ContextAnchoredTileRefineVL,
    "ContextAnchoredTileUpscaleVL": ContextAnchoredTileUpscaleVL,
    "ContextAnchoredTileUpscaleVLVideo": ContextAnchoredTileUpscaleVLVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ContextAnchoredTileRefine": "Context-Anchored Tile Refine",
    "ContextAnchoredTileRefineVL": "Context-Anchored Tile Refine (VL)",
    "ContextAnchoredTileUpscaleVL": "Context-Anchored Tile Upscale (VL)",
    "ContextAnchoredTileUpscaleVLVideo": "Context-Anchored Tile Upscale (VL Video)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
