"""Per-tile conditioning transform: cut every ControlNet hint down to the tile being sampled.

Core rebuilds a control's working hint from `cond_hint_original` whenever the latent shape
disagrees with the cached one, by center-crop-to-aspect plus rescale onto the current latent
(`comfy/controlnet.py:269` and `:279`). Handed a FULL-image hint, every tile therefore gets the
wrong content at the wrong scale. Handed a hint already cut to the tile's `crop_rect` — whose
dims are /8 by construction, matching `compression_ratio` — that rescale is an identity and the
hint lands aligned. That is the whole feature: no new node inputs, no resampling anywhere.

Control objects are DUCK-TYPED here (`copy()`, `set_previous_controlnet()`,
`cond_hint_original`, `extra_concat_orig`, `previous_controlnet`) so this module — like
`sampling.py` — imports torch and stdlib only at module scope, never `comfy`.

Only the `control` key is transformed. The other spatial cond keys (`gligen`, `area`, `mask`,
`reference_latents`) pass through untouched: their coordinate-space semantics on the node's mask
path are unresolved, and cropping `reference_latents` would regress Flux-Kontext-style
workflows that work today.
"""
import torch


def has_spatial_conds(original_conds):
    # The guard. True only when some cond dict carries a `control` object, i.e. only when the
    # transform below would change anything; every other run stays byte-identical. Accepts
    # None, which is what a guider without an `original_conds` attribute yields.
    if not original_conds:
        return False
    for cond_list in original_conds.values():
        for cond in cond_list:
            if cond.get("control") is not None:
                return True
    return False


def prepare_hint_canvas(original_conds, image_size, bbox=None):
    # One-time prep, in the caller's FULL input pixel space: validate every distinct hint
    # against `image_size` (H, W) and cut it to `bbox` (y0, y1, x0, x1) on the mask path, the
    # same slice the image itself gets. Returns {id(original tensor): prepared tensor}.
    #
    # An id that is ABSENT from the map means "not canvas-sized — pass through untouched".
    # That is how core's auto-inserted [1,1,1,1] `concat_mask` placeholder
    # (`comfy/controlnet.py:112-113`) survives: slicing it at a non-origin `crop_rect` would
    # yield a 0-dim tensor and crash core's `common_upscale`.
    #
    # Hints are CHANNELS-FIRST [B,C,H,W] (`nodes.py:903/941` build them with
    # `image.movedim(-1,1)`), so every size read and every slice here is on dims (-2,-1).
    height, width = image_size
    canvas = {}

    for control in _iter_controls(original_conds):
        hint = getattr(control, "cond_hint_original", None)
        if hint is not None and id(hint) not in canvas:
            hint_h, hint_w = int(hint.shape[-2]), int(hint.shape[-1])
            if (hint_h, hint_w) != (height, width):
                # Fail fast: a hint that is not the image being refined is a wiring mistake,
                # and core's fallback (proportional stretch) would silently misalign it.
                raise ValueError(
                    f"ContextAnchoredTileRefine: the ControlNet hint is {hint_h}x{hint_w} (HxW) but the "
                    f"image being refined is {height}x{width}. Feed the control preprocessor the same "
                    "upscaled image this node refines."
                )
            canvas[id(hint)] = _slice_bbox(hint, bbox)
        for extra in getattr(control, "extra_concat_orig", None) or ():
            # An extra_concat is spatial and core rescales it onto the hint
            # (`comfy/controlnet.py:287-296`), so a canvas-sized one must follow the hint's
            # crop; anything else is left out of the map on purpose (see above).
            if id(extra) in canvas:
                continue
            if (int(extra.shape[-2]), int(extra.shape[-1])) == (height, width):
                canvas[id(extra)] = _slice_bbox(extra, bbox)
    return canvas


def pad_hint_canvas(hint_canvas, size, padded_size):
    # Put the prepared hints on the SAME padded canvas as the pixels so a tile's `crop_rect`
    # indexes both identically: same reflect mode and the same right/bottom amounts as
    # `sampling.pad_image_to_multiple`. The hints are already channels-first, so F.pad hits
    # dims (-2,-1) with no movedim. Returns the input map unchanged when nothing needs padding.
    pad_h = padded_size[0] - size[0]
    pad_w = padded_size[1] - size[1]
    if pad_h == 0 and pad_w == 0:
        return hint_canvas
    return {
        key: torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
        for key, tensor in hint_canvas.items()
    }


def crop_tile_conds(original_conds, hint_canvas, crop_rect):
    # Per-tile transform: a NEW conds map whose every `control` chain is a fresh copy carrying
    # the `crop_rect` slice of the prepared hint. Nothing the caller owns is mutated — the cond
    # dicts are shallow copies and the control objects are copies, so the user's hint tensors
    # and control objects come out of a run exactly as they went in.
    #
    # FRESH copies every tile: `get_control` caches the processed hint and only invalidates it
    # on a SHAPE change (`comfy/controlnet.py:269`), and interior tiles share one crop shape,
    # so a reused object would serve tile k's hint to tile k+1.
    memo = {}
    cropped_conds = {}

    for key, cond_list in original_conds.items():
        cropped_list = []
        for cond in cond_list:
            control = cond.get("control")
            if control is None:
                cropped_list.append(cond)
                continue
            cropped = cond.copy()
            cropped["control"] = _copy_control(control, hint_canvas, crop_rect, memo)
            cropped_list.append(cropped)
        cropped_conds[key] = cropped_list
    return cropped_conds


def strip_control(original_conds):
    # The VL path's transform: a NEW conds map with the `control` key dropped from every cond
    # dict, so no control chain reaches the sampler on either branch. Clearing `control_active`
    # only skips the CROP — core still reads `control` off each cond (`comfy/samplers.py:96`)
    # and `get_control` rebuilds the working hint from `cond_hint_original` whenever it
    # disagrees with the latent (`comfy/controlnet.py:234-244`), so a control left on the
    # negative would steer the uncond branch alone against a center-cropped, squashed copy of
    # the caller's FULL-image hint. Nothing the caller owns is mutated: cond dicts are rebuilt,
    # never edited in place. Returns the input map unchanged when nothing carries control.
    if not has_spatial_conds(original_conds):
        return original_conds
    stripped_conds = {}

    for key, cond_list in original_conds.items():
        stripped_conds[key] = [
            {name: value for name, value in cond.items() if name != "control"}
            if "control" in cond else cond
            for cond in cond_list
        ]
    return stripped_conds


def _iter_controls(original_conds):
    # Every control object reachable from the conds, chain links included. Objects repeat (the
    # positive and negative dicts share one `c_net`, `nodes.py:948-956`); callers memoize.
    for cond_list in original_conds.values():
        for cond in cond_list:
            control = cond.get("control")
            while control is not None:
                yield control
                control = getattr(control, "previous_controlnet", None)


def _slice_bbox(tensor, bbox):
    # The mask path's bbox cut, channels-first. bbox is (y0, y1, x0, x1) or None.
    if bbox is None:
        return tensor
    y0, y1, x0, x1 = bbox
    return tensor[..., y0:y1, x0:x1]


def _crop_hint(tensor, crop_rect):
    # One tile's slice of a padded-canvas hint. Channels-first, so (-2,-1) are (H, W). A view,
    # never a copy: core only ever reads the hint.
    return tensor[..., crop_rect.y0:crop_rect.y1, crop_rect.x0:crop_rect.x1]


def _copy_control(control, hint_canvas, crop_rect, memo):
    # One chain link, memoized by id(original) so the positive and negative cond dicts that
    # share one control object (`nodes.py:948-956`) share one copy — mirroring core, and
    # halving the hint caches a doubled copy would build.
    copied = memo.get(id(control))
    if copied is not None:
        return copied

    copied = control.copy()
    previous = getattr(control, "previous_controlnet", None)
    if previous is not None:
        # `ControlBase.copy_to` (`comfy/controlnet.py:168-183`) carries everything EXCEPT
        # `previous_controlnet`, so a multi-ControlNet chain has to be re-linked link by link
        # or every net but the last silently drops out.
        copied.set_previous_controlnet(_copy_control(previous, hint_canvas, crop_rect, memo))

    hint = getattr(control, "cond_hint_original", None)
    prepared = None if hint is None else hint_canvas.get(id(hint))
    if prepared is not None:
        copied.cond_hint_original = _crop_hint(prepared, crop_rect)
    extra_concat = getattr(control, "extra_concat_orig", None)
    if extra_concat:
        copied.extra_concat_orig = [
            _crop_hint(hint_canvas[id(extra)], crop_rect) if id(extra) in hint_canvas else extra
            for extra in extra_concat
        ]

    memo[id(control)] = copied
    return copied
