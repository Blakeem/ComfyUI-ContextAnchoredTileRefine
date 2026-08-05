import torch

# Mirrors core nodes.MAX_RESOLUTION (importing it would pull comfy in at module scope).
MAX_RESOLUTION = 16384


def _validate_image(image):
    if image.ndim != 4:
        raise ValueError("image must be a [B,H,W,C] IMAGE tensor, got {} dimensions".format(image.ndim))
    if image.shape[1] < 8 or image.shape[2] < 8:
        raise ValueError("image must be at least 8x8 pixels, got {}x{}".format(image.shape[1], image.shape[2]))


class ContextAnchoredTileRefine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The upscaled image to refine tile by tile."}),
                "guider": ("GUIDER", {"tooltip": "The guider (e.g. from CFGGuider) that denoises each tile."}),
                "sampler": ("SAMPLER", {"tooltip": "The sampler used to denoise each tile."}),
                "sigmas": ("SIGMAS", {"tooltip": "The sigma schedule used when sampling each tile."}),
                "vae": ("VAE", {"tooltip": "The VAE used to encode each tile for sampling and decode the result."}),
                "noise": ("NOISE", {"tooltip": "The noise source (e.g. from RandomNoise) used when sampling each tile."}),
                "max_tile_width": ("INT", {"default": 1024, "min": 256, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Hard cap on the width the model ever sees per sampled crop, including the context_overlap and context_anchor rings. Set to the largest width the model supports."}),
                "max_tile_height": ("INT", {"default": 1024, "min": 256, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Hard cap on the height the model ever sees per sampled crop, including the context_overlap and context_anchor rings. Set to the largest height the model supports."}),
                "context_anchor": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": "Width of the fully-frozen, visible-only context ring sampled beyond the overlap on every edge (including up against a mask) then cropped away. With a mask it also sets the frozen-background halo the masked region conditions against — keep it > 0. Always additive: the ring outside a tile core is context_overlap + context_anchor. 32-256 is the useful range; 32 suits most scenes."}),
                "context_overlap": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": "Inter-tile directional feather width (multi-tile runs only; never applied at a mask boundary). Each tile is sampled oversized; on sides bordering an already-processed neighbor (top/left) this band is fully diffused and feathered into that neighbor (100% at the seam → 0% over the band); elsewhere it's context, then cropped. 32-256 is the useful range: DETAILED scenes need LESS (32 is invisible even when you know where the seam is) because the seam has texture to hide in and to route through, while a large smooth gradient (an open night sky) is the hard case and wants 128+. 0 = hard seams."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional region mask: only the masked region is refined (hardened at 0.5), cropped to the mask plus context_anchor, with a 1px anti-aliased edge; the unmasked background is left untouched. Feed an inverted mask for a second pass."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "refine"
    CATEGORY = "sampling/custom_sampling"

    @classmethod
    def VALIDATE_INPUTS(s, max_tile_width=None, max_tile_height=None, context_anchor=None, context_overlap=None):
        # Naming widgets here disables ComfyUI's default min/max validation for them,
        # so ranges are re-checked alongside the /8 constraint (widget step is UI-only;
        # API-submitted workflows can send arbitrary INTs, and all derived geometry
        # must stay on the /8 grid the VAE requires).
        checks = (
            ("max_tile_width", max_tile_width, 256, MAX_RESOLUTION),
            ("max_tile_height", max_tile_height, 256, MAX_RESOLUTION),
            ("context_anchor", context_anchor, 0, 512),
            ("context_overlap", context_overlap, 0, 512),
        )
        for name, value, minimum, maximum in checks:
            if value is None:
                continue
            if value % 8 != 0:
                return "{} must be a multiple of 8, got {}".format(name, value)
            if value < minimum or value > maximum:
                return "{} must be between {} and {}, got {}".format(name, minimum, maximum, value)
        return True

    def refine(self, image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, mask=None):
        _validate_image(image)
        if mask is not None:
            # Normalize a 2D [H,W] mask to [1,H,W]; reject any other rank.
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim != 3:
                raise ValueError("mask must be a [H,W] or [B,H,W] MASK tensor, got {} dimensions".format(mask.ndim))
            # Strict spatial match (no resample — a resized mask would misalign the region).
            if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
                raise ValueError("mask size {}x{} must match image size {}x{}".format(mask.shape[1], mask.shape[2], image.shape[1], image.shape[2]))
            # Batch must be 1 (broadcast to every image) or exactly the image batch.
            if mask.shape[0] not in (1, image.shape[0]):
                raise ValueError("mask batch {} must be 1 or match image batch {}".format(mask.shape[0], image.shape[0]))
            if mask.shape[0] == 1 and image.shape[0] != 1:
                mask = mask.expand(image.shape[0], -1, -1)
        from . import sampling
        return (sampling.refine_image(image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, mask=mask),)


class ContextAnchoredTileRefineVL(ContextAnchoredTileRefine):
    """The vision-conditioned variant for VLM-encoder models (Krea 2 family).

    Same tiling engine as the base node, but every tile's positive conditioning is
    replaced by its slice of ONE whole-image vision encode from the required CLIP:
    positionally exact, free of text demands, and globally informed, so tiles neither
    re-instantiate prompt objects they don't contain nor drift apart in story
    (gaze, tone, palette). The guider's positive text is ignored by construction;
    its negative still applies. No prompt input exists because none is needed.
    """

    @classmethod
    def INPUT_TYPES(s):
        input_types = ContextAnchoredTileRefine.INPUT_TYPES()
        input_types["required"]["clip"] = ("CLIP", {"tooltip": "The workflow's CLIP — must be a vision-language text encoder (Krea 2 family). The whole image is encoded once through its vision path and each tile's positive conditioning becomes its slice of that encode; the guider's positive prompt is ignored, the negative still applies."})
        # No mask input: the region/vision-grid coordinate semantics for a masked VL
        # refine are unresolved, and a half-right mapping would misplace every slice.
        del input_types["optional"]
        return input_types

    def refine(self, image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, clip):
        _validate_image(image)
        from . import sampling
        return (sampling.refine_image(image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, mask=None, vl_clip=clip),)
