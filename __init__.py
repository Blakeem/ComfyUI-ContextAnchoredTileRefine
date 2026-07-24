from .context_anchored_tile_refine.node import ContextAnchoredTileRefine

NODE_CLASS_MAPPINGS = {
    "ContextAnchoredTileRefine": ContextAnchoredTileRefine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ContextAnchoredTileRefine": "Context-Anchored Tile Refine",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
