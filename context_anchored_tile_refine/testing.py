"""The tile testing chain: the Layout node, the Upscale node, the Captions node and the
Render node.

`ContextAnchoredTileTestLayout` solves the production grid for the size the image will have
AFTER upscaling and draws that grid over a preview, so the tile count and every tile's number
are visible before the upscale is spent. The solve is `sync._prepare_run`'s own: the target
size rounded up to a multiple of 8 on each axis, then the two `grid.solve_axis` calls and the
one `grid.build_layout` call, so what the overlay shows is what the engine will sample.

`ContextAnchoredTileTestUpscale` runs the production upscale stage as a node of its own. That
is what buys the chain its caching: ComfyUI holds the upscaled canvas, and a downstream re-run
at a new seed never pays for it again.

`ContextAnchoredTileTestCaptions` writes one VLM caption per chosen tile through the same
`captions.generate_caption_set` the engine runs, from a settings file preset or from its own
instruction widgets, under one progress bar. The style caption is kept apart from the tile
captions and listed once, labelled, at the top. Its inputs carry no seed, so the captions survive
a seed re-roll further down the chain, and its IS_CHANGED re-runs it when the settings file
changes. Its tiles output feeds the Render node's tiles input, so one tile is captioned and
rendered together.

`ContextAnchoredTileTestRender` runs the production VL engine over that canvas. Named tiles
are rendered one at a time as region runs over the block the tile and its bordering tiles
occupy, at the parent grid's own rects, captions and noise slice, so one tile can be judged
without paying for the entire canvas.

The `TestLayout` object carries the widgets the grid was solved from, so every node further
down the chain reads its geometry from the LAYOUT rather than from its own copy of the
widgets, and no two nodes in one chain can disagree about the grid.

Module scope is torch and stdlib only. PIL is imported inside the drawing function, and every
comfy-touching path of upscale.py is reached only from inside a node method (the same contract
as sampling.py / vl.py, pinned by a subprocess test).
"""
import math
from dataclasses import dataclass, replace

import torch

from . import captions, grid, node, progress, sampling, upscale, vl

# The Captions node's preset option that reads the instruction widgets instead of a settings
# file preset. A named preset ignores those widgets, so a trial wording can stay in them.
CUSTOM_PRESET = "custom instructions"

# The overlay preview's pixel budget. Large enough to read a label on an 8K canvas, small
# enough that the picture stays a cheap preview.
OVERLAY_MEGAPIXELS = 2.0

# One fixed colour per band, outward from the core, so a tile reads the same way in every
# render. The bands are drawn band by band rather than tile by tile, so a later tile's rings
# never cover an earlier tile's core.
CROP_COLOR = (255, 96, 96)
OVERLAP_COLOR = (255, 208, 64)
CORE_COLOR = (96, 224, 255)
LABEL_COLOR = (255, 255, 255)
LINE_WIDTH = 2
# Past the core's own line, so a label never hides the corner it names.
LABEL_INSET = LINE_WIDTH + 1


@dataclass(frozen=True)
class TestLayout:
    """One solved grid plus the widgets it came from, the CATR_LAYOUT the chain passes down.

    `source_size` and `target_size` are (width, height) in pixels, the input's own size and
    the size after `upscale_by`. `layout` is solved on the target size padded up to a multiple
    of 8 on each axis, so its rects are the canvas rects the engine samples.
    """

    upscale_by: float
    max_tile_width: int
    max_tile_height: int
    context_anchor: int
    context_overlap: int
    source_size: tuple
    target_size: tuple
    layout: grid.Layout

    @property
    def tile_count(self):
        return len(self.layout.tiles)


@dataclass(frozen=True)
class TestCaptions:
    """The captions written for a layout plus the geometry they belong to, the CATR_CAPTIONS
    object.

    `captions` holds one entry per tile in layout order: a tuple of one caption per batch row,
    the tile's OWN caption as `captions.generate_caption_set` returns it, or None for a tile
    the node was not asked to caption. `style` is the whole-image style caption as a tuple of
    one per batch row, or None when the run asked for none. The two are joined by
    `captions.join_style_captions` only where the engine reads them (the Render node), so the
    listing shows the style once instead of on top of every tile. `tiles` is the tile numbers
    the node was asked for, in the order they were written, and empty when every tile was
    captioned. A caption belongs to a tile rect,
    so `target_size`, `grid` (columns, rows) and `rects` (every tile's `crop_rect` as (x0, y0,
    x1, y1), in order) let a consumer reject captions written for another grid at the same
    size. `preset` is the label they were written from.

    `str()` is the readable listing, which is what core's Preview as Text node shows for a
    value it cannot serialize as JSON, so the socket reads the same as the text output.
    """

    captions: tuple
    style: tuple | None
    tiles: tuple
    target_size: tuple
    grid: tuple
    rects: tuple
    preset: str

    def __str__(self):
        columns = self.grid[0]
        captioned = tuple(index for index, rows in enumerate(self.captions) if rows is not None)
        named = self.tiles or captioned
        bordering = tuple(index for index in captioned if index not in named)
        count = f"{len(captioned)} tiles" if len(captioned) == len(self.captions) else (
            f"{len(captioned)} of {len(self.captions)} tiles")
        blocks = [f"preset {self.preset}, {count} captioned"]
        if self.style is not None:
            blocks.append(f"style caption\n{self.style[0]}")
        blocks.extend(self._block(index, columns) for index in named)
        if bordering:
            blocks.append("bordering tiles")
            blocks.extend(self._block(index, columns) for index in bordering)
        return "\n\n".join(blocks)

    def _block(self, index, columns):
        # The overlay's own label, so a caption is found by the number drawn on the tile.
        return f"tile {index} r{index // columns}c{index % columns}\n{self.captions[index][0]}"


def _preview_size(target_width, target_height):
    # The target size shrunk to fit OVERLAY_MEGAPIXELS, keeping the target's aspect. Floored
    # rather than rounded, so the product cannot land above the budget.
    scale = math.sqrt(OVERLAY_MEGAPIXELS * 1_000_000 / (target_width * target_height))
    if scale >= 1.0:
        return target_width, target_height
    return max(1, int(target_width * scale)), max(1, int(target_height * scale))


def _preview_box(rect, x_scale, y_scale, preview_width, preview_height):
    # A canvas rect as the inclusive box PIL draws, clipped to the preview. The two scales are
    # separate because the layout lives on the padded canvas and the preview on the target
    # size, and the two axes are padded by different amounts.
    x0 = min(max(round(rect.x0 * x_scale), 0), preview_width - 1)
    y0 = min(max(round(rect.y0 * y_scale), 0), preview_height - 1)
    x1 = min(max(round(rect.x1 * x_scale) - 1, x0), preview_width - 1)
    y1 = min(max(round(rect.y1 * y_scale) - 1, y0), preview_height - 1)
    return [x0, y0, x1, y1]


def draw_overlay(picture, test_layout):
    """The grid drawn over a preview of `picture` ([H,W,C] in 0 to 1), as an IMAGE tensor."""
    # PIL here and nowhere higher: this module's scope is torch and stdlib.
    from PIL import Image, ImageDraw, ImageFont

    # ---- inputs
    target_width, target_height = test_layout.target_size
    preview_width, preview_height = _preview_size(target_width, target_height)
    x_scale = preview_width / target_width
    y_scale = preview_height / target_height
    tiles = test_layout.layout.tiles

    # ---- process. The resample is legitimate where a tile's would not be (prime directive 1):
    # this picture is looked at and then discarded, and nothing downstream samples it.
    source = picture[..., :3].detach().movedim(-1, 0)[None].float()
    resized = torch.nn.functional.interpolate(source, size=(preview_height, preview_width), mode="area")
    pixels = resized[0].movedim(0, -1).clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().contiguous()

    preview = Image.frombytes("RGB", (preview_width, preview_height), pixels.numpy().tobytes())
    draw = ImageDraw.Draw(preview)
    bands = (
        (CROP_COLOR, tuple(tile.crop_rect for tile in tiles)),
        (OVERLAP_COLOR, tuple(tile.overlap_inner_rect for tile in tiles)),
        (CORE_COLOR, tuple(tile.core for tile in tiles)),
    )
    for color, rects in bands:
        for rect in rects:
            draw.rectangle(_preview_box(rect, x_scale, y_scale, preview_width, preview_height),
                           outline=color, width=LINE_WIDTH)
    font = ImageFont.load_default()
    for index, tile in enumerate(tiles):
        box = _preview_box(tile.core, x_scale, y_scale, preview_width, preview_height)
        draw.text((box[0] + LABEL_INSET, box[1] + LABEL_INSET), f"{index} r{tile.row}c{tile.col}",
                  fill=LABEL_COLOR, font=font)

    # ---- output. A bytearray rather than the read-only bytes, since torch.frombuffer warns
    # on a non-writable buffer.
    drawn = torch.frombuffer(bytearray(preview.tobytes()), dtype=torch.uint8)
    return drawn.reshape(1, preview_height, preview_width, 3).to(torch.float32).div(255.0)


class ContextAnchoredTileTestLayout:
    """Solve the tile grid for the upscaled size and draw it over the image.

    This is a testing node and not one of the production nodes. The grid is the production solve, on the size the image will have after upscale_by, so the
    tile count and the tile numbers shown here are the ones the refine will use. Feed the
    layout to Tile Test: Upscale and the rest of the chain.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The image the chain starts from, before any upscale."}),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 8.0, "step": 0.01, "tooltip": "The upscale multiplier. The optional upscale_model runs first when one is connected."}),
                "max_tile_width": ("INT", {"default": 1536, "min": 256, "max": node.MAX_RESOLUTION, "step": 8, "tooltip": "Hard cap on the width the model ever sees per sampled crop, including the context_overlap and context_anchor rings. Set to the largest width the model supports."}),
                "max_tile_height": ("INT", {"default": 2048, "min": 256, "max": node.MAX_RESOLUTION, "step": 8, "tooltip": "Hard cap on the height the model ever sees per sampled crop, including the context_overlap and context_anchor rings. Set to the largest height the model supports."}),
                "context_anchor": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": "Pixels around each tile that are frozen and shown to the model as context, then cropped away."}),
                "context_overlap": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": "Overlapped context that is diffused from both sides and then blended. It anchors the tiles to each other, like context_anchor anchors each tile to its surroundings."}),
            },
        }

    RETURN_TYPES = ("CATR_LAYOUT", "IMAGE", "INT")
    RETURN_NAMES = ("layout", "overlay", "tile_count")
    FUNCTION = "solve_layout"
    CATEGORY = "image/upscaling/tile testing"

    @classmethod
    def VALIDATE_INPUTS(s, max_tile_width=None, max_tile_height=None, context_anchor=None, context_overlap=None):
        # The base node's own rules, read from the one place they are written, so the testing
        # chain and the production nodes cannot drift apart on what geometry is legal.
        return node.check_geometry(max_tile_width, max_tile_height, context_anchor, context_overlap)

    def solve_layout(self, image, upscale_by, max_tile_width, max_tile_height, context_anchor, context_overlap):
        # ---- inputs
        if image.shape[0] != 1:
            raise ValueError(
                f"Tile Test: Layout lays out one picture at a time, got a batch of {image.shape[0]}.")
        source_width, source_height = int(image.shape[2]), int(image.shape[1])

        # ---- process. sync._prepare_run's own pad and solve, on the size the upscale lands
        # on, so the grid drawn here is the grid the engine will sample.
        target_width, target_height = upscale.scale_target(source_width, source_height, upscale_by)
        canvas_width, canvas_height = grid.round8_up(target_width), grid.round8_up(target_height)
        sx = grid.solve_axis(canvas_width, max_tile_width, context_anchor, context_overlap, axis="width")
        sy = grid.solve_axis(canvas_height, max_tile_height, context_anchor, context_overlap, axis="height")
        test_layout = TestLayout(
            upscale_by=upscale_by,
            max_tile_width=max_tile_width,
            max_tile_height=max_tile_height,
            context_anchor=context_anchor,
            context_overlap=context_overlap,
            source_size=(source_width, source_height),
            target_size=(target_width, target_height),
            layout=grid.build_layout(canvas_width, canvas_height, sx, sy, context_anchor, context_overlap),
        )

        # ---- output
        return (test_layout, draw_overlay(image[0], test_layout), test_layout.tile_count)


class ContextAnchoredTileTestUpscale:
    """Run the upscale stage the all-in-one node runs, as a node of its own.

    This is a testing node and not one of the production nodes. The multiplier comes from the layout, so the canvas this returns is the size the layout
    was solved for. It is kept separate from the refine so ComfyUI caches the upscaled canvas
    and a re-run at a new seed never pays for it twice.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The same image Tile Test: Layout was given."}),
                "layout": ("CATR_LAYOUT", {"tooltip": "The layout from Tile Test: Layout. It carries the upscale multiplier."}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Optional upscale model, run over the entire image before the resize to the exact target."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale_image"
    CATEGORY = "image/upscaling/tile testing"

    def upscale_image(self, image, layout, upscale_model=None):
        # ---- inputs
        if image.shape[0] != 1:
            raise ValueError(
                f"Tile Test: Upscale upscales one picture at a time, got a batch of {image.shape[0]}.")
        source_size = (int(image.shape[2]), int(image.shape[1]))
        if source_size != layout.source_size:
            raise ValueError(
                f"Tile Test: Upscale was given a {source_size[0]}x{source_size[1]} image, but the "
                f"layout was solved for {layout.source_size[0]}x{layout.source_size[1]}. Feed both "
                "nodes the same image.")

        # ---- process
        upscaled = upscale.prepare_upscaled(image, upscale_model, layout.upscale_by, progress=None)
        result_size = (int(upscaled.shape[2]), int(upscaled.shape[1]))
        if result_size != layout.target_size:
            raise RuntimeError(
                f"the upscale produced {result_size[0]}x{result_size[1]}, but the layout was solved "
                f"for {layout.target_size[0]}x{layout.target_size[1]}. Every node below reads its "
                "geometry from the layout, so the two must agree.")

        # ---- output
        return (upscaled,)


def _parse_tile_numbers(text, tile_count, node_name):
    # The csv widget, as the overlay labels the tiles. A duplicate is dropped rather than
    # rejected, so naming a tile twice costs one render, and the order the user wrote is kept
    # because it is the order the two output lists come back in.
    numbers = []
    for token in text.split(","):
        entry = token.strip()
        if not entry:
            continue
        try:
            number = int(entry)
        except ValueError:
            raise ValueError(
                f"{node_name} was given the tile {entry!r}, which is not a tile number. tiles "
                f"takes comma separated numbers from 0 to {tile_count - 1}, or nothing at all "
                "for the entire canvas.") from None
        if not 0 <= number < tile_count:
            raise ValueError(
                f"{node_name} was given tile {number}, and this layout has {tile_count} tiles "
                f"numbered 0 to {tile_count - 1}.")
        if number not in numbers:
            numbers.append(number)
    return tuple(numbers)


def _preset_options():
    labels = list(captions.preset_labels())
    if CUSTOM_PRESET in labels:
        raise ValueError(
            f"{captions.settings_path()} defines a preset named {CUSTOM_PRESET!r}, which is the "
            "Tile Test: Captions option for its own instruction widgets. Rename the preset.")
    return [*labels, CUSTOM_PRESET]


def _caption_preset(preset, tile_instruction, style_caption, style_instruction, max_tokens,
                    caption_megapixels, prompt):
    # The block the captions are written from. A named preset is read from the settings file
    # and the three instruction widgets are left alone, so a trial wording can stay in them
    # while the file's own wording runs. The custom option reads those widgets instead. The
    # prompt fills {PROMPT} in either, the same as on the production nodes.
    if preset == CUSTOM_PRESET:
        if not tile_instruction.strip():
            raise ValueError(
                f"Tile Test: Captions was set to {CUSTOM_PRESET!r} with an empty "
                "tile_instruction. Write the question to ask about each tile, or pick a preset.")
        if max_tokens <= 0:
            raise ValueError(
                f"Tile Test: Captions was set to {CUSTOM_PRESET!r} with max_tokens 0. The custom "
                "option has no preset budget to fall back on, so max_tokens must be above 0.")
        base = captions.Preset(
            surface=captions.VLM_METHOD_CAPTIONS,
            label=CUSTOM_PRESET,
            vision=captions.load_settings().vision,
            tile_instruction=tile_instruction,
            tile_max_tokens=max_tokens,
            style_instruction=style_instruction if style_instruction.strip() else "",
            style_max_tokens=max_tokens,
        )
    else:
        # The labeled option, which resolves for every preset including the first one, whose
        # options the selector offers unlabeled.
        base = captions.resolve_method(f"{captions.VLM_METHOD_CAPTIONS} ({preset})")
    chosen = replace(
        base,
        style_instruction=base.style_instruction if style_caption else "",
        vision=replace(base.vision, caption_megapixels=caption_megapixels),
    )
    return captions.with_prompt(chosen, prompt)


def _bordering_tiles(layout, index):
    tiles = layout.tiles
    col0, col1, row0, row1 = grid.neighborhood(layout, index)
    return tuple(other for other, tile in enumerate(tiles)
                 if col0 <= tile.col <= col1 and row0 <= tile.row <= row1 and other != index)


def _tiles_to_caption(layout, tiles, with_neighbors):
    # (named, captioned): the tile numbers the widget asked for, then every tile that gets a
    # caption, the named ones first in the order written and then their bordering tiles in
    # layout order, since a block run at Tile Test: Render needs a caption for every lane.
    named = _parse_tile_numbers(tiles, len(layout.tiles), "Tile Test: Captions")
    if not named:
        return (), tuple(range(len(layout.tiles)))
    captioned = list(named)
    if with_neighbors:
        bordering = sorted({other for index in named for other in _bordering_tiles(layout, index)})
        captioned.extend(other for other in bordering if other not in captioned)
    return named, tuple(captioned)


class ContextAnchoredTileTestCaptions:
    """Write VLM captions for tiles of the layout, through the engine's own caption pass.

    This is a testing node and not one of the production nodes. The preset comes from the
    settings file, or from the instruction widgets when preset is set to custom instructions,
    so a prompt can be tried without editing the file. An empty tiles list captions every tile.
    A tile list captions those tiles, plus their bordering tiles when with_neighbors is on,
    which is what Tile Test: Render needs to render one of them with its neighbours. There is
    no seed here, so ComfyUI serves the captions from its cache while a seed is re-rolled
    further down the chain. Feed the captions and the tiles to Tile Test: Render.
    """

    @classmethod
    def INPUT_TYPES(s):
        options = _preset_options()
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The canvas from Tile Test: Upscale, at the layout's target size."}),
                "layout": ("CATR_LAYOUT", {"tooltip": "The layout from Tile Test: Layout. The tiles are captioned from it."}),
                "clip": ("CLIP", {"tooltip": "Must be a vision-language text encoder with a text generator (Krea 2 family)."}),
                "preset": (options, {"default": options[0], "tooltip": f"Which settings file preset the captions are written from. '{CUSTOM_PRESET}' writes them from the tile_instruction, style_instruction and max_tokens widgets instead, which a named preset ignores. The list is built when ComfyUI starts, so a new or renamed preset needs a restart."}),
                "tile_instruction": ("STRING", {"default": "", "multiline": True, "tooltip": f"What the VL model is asked about each tile. Read only when preset is '{CUSTOM_PRESET}'."}),
                "style_caption": ("BOOLEAN", {"default": True, "tooltip": "Write one style caption of the entire image and place it on top of every tile caption. Off leaves every preset's style caption out."}),
                "style_instruction": ("STRING", {"default": "", "multiline": True, "tooltip": f"What the VL model is asked about the entire image. Read only when preset is '{CUSTOM_PRESET}'. Empty writes no style caption."}),
                "max_tokens": ("INT", {"default": 0, "min": 0, "max": captions.MAX_CAPTION_TOKENS, "tooltip": f"Generation budget for every caption. The budget also covers the model's hidden reasoning turn. Read only when preset is '{CUSTOM_PRESET}', where it must be above 0."}),
                "caption_megapixels": ("FLOAT", {"default": captions.load_settings().vision.caption_megapixels, "min": 0.0, "max": vl.PICTURE_CAP_MEGAPIXELS, "step": 0.01, "tooltip": f"How much of the tile the VL model reads. Use 0 for the crop's own size, capped at {vl.PICTURE_CAP_MEGAPIXELS} megapixels."}),
                "tiles": ("STRING", {"default": "", "tooltip": "Comma separated tile numbers, as Tile Test: Layout labels them. Empty captions every tile. The same list comes out of the tiles output for Tile Test: Render."}),
                "with_neighbors": ("BOOLEAN", {"default": True, "tooltip": "Caption the bordering tiles of every named tile as well, which Tile Test: Render needs when its with_neighbors is on. Off captions the named tiles only."}),
            },
            "optional": {
                "prompt": node._prompt(),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("CATR_CAPTIONS", "STRING", "STRING")
    RETURN_NAMES = ("captions", "text", "tiles")
    FUNCTION = "caption_tiles"
    CATEGORY = "image/upscaling/tile testing"

    @classmethod
    def VALIDATE_INPUTS(s, caption_megapixels=None):
        # Naming a widget here disables ComfyUI's own min and max check for it (node.py's
        # VALIDATE_INPUTS states the rule), so the settings file's entire rule is re-checked.
        # The widget's own range cannot express "0 or at least VL_INPUT_MIN_MEGAPIXELS".
        if caption_megapixels is None or caption_megapixels == 0:
            return True
        if not captions.VL_INPUT_MIN_MEGAPIXELS <= caption_megapixels <= vl.PICTURE_CAP_MEGAPIXELS:
            return (
                "caption_megapixels must be 0, which reads the picture's own size, or between "
                f"{captions.VL_INPUT_MIN_MEGAPIXELS} and {vl.PICTURE_CAP_MEGAPIXELS}. Got "
                f"{caption_megapixels}.")
        return True

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # ComfyUI folds this value into the node's cache key, and the preset is read at run
        # time, so without it an edited preset never reaches a workflow nobody retuned.
        return captions.settings_fingerprint()

    def caption_tiles(self, image, layout, clip, preset, tile_instruction, style_caption,
                      style_instruction, max_tokens, caption_megapixels, tiles, with_neighbors,
                      prompt=None, unique_id=None):
        # ---- inputs
        if image.shape[0] != 1:
            raise ValueError(
                f"Tile Test: Captions writes captions for one picture at a time, got a batch of "
                f"{image.shape[0]}.")
        size = (int(image.shape[2]), int(image.shape[1]))
        if size != layout.target_size:
            raise ValueError(
                f"Tile Test: Captions was given a {size[0]}x{size[1]} image, but the layout was "
                f"solved for {layout.target_size[0]}x{layout.target_size[1]}. Feed it the canvas "
                "Tile Test: Upscale returns.")
        run_preset = _caption_preset(preset, tile_instruction, style_caption, style_instruction,
                                     max_tokens, caption_megapixels, prompt)
        all_tiles = layout.layout.tiles
        named, captioned = _tiles_to_caption(layout.layout, tiles, with_neighbors)
        chosen = [all_tiles[index] for index in captioned]

        # ---- process. The engine captions the PADDED canvas, so this pass must too, or a
        # tile's caption would describe a crop the run never reads. The ledger is what makes
        # the run ONE bar: core builds a per-token bar inside every clip.generate, and the
        # ledger's shim routes it into the caption's own chunk instead of resetting the display.
        padded, _ = sampling.pad_image_to_multiple(image)
        n_captions = len(chosen) + (1 if run_preset.style_instruction else 0)
        ledger = progress.build_caption_ledger(n_captions, unique_id=unique_id)
        with ledger:
            style, written = captions.generate_caption_set(clip, padded, chosen, run_preset,
                                                           progress=ledger)
            ledger.finish()

        # ---- output
        per_tile = [None] * len(all_tiles)
        for index, rows in zip(captioned, written, strict=True):
            per_tile[index] = tuple(rows)
        result = TestCaptions(
            captions=tuple(per_tile),
            style=tuple(style) if style else None,
            tiles=named,
            target_size=layout.target_size,
            grid=(layout.layout.sol_x.n, layout.layout.sol_y.n),
            rects=tuple((tile.crop_rect.x0, tile.crop_rect.y0, tile.crop_rect.x1, tile.crop_rect.y1)
                        for tile in all_tiles),
            preset=run_preset.label,
        )
        return (result, str(result), ", ".join(str(index) for index in named))


def _run_preset(surface, canvas_tokens, crop_tokens):
    # The settings file's block with the two widgets written over its [vision] table, so a
    # token count can be tried without editing the file. Both counts at 0 is what the file
    # itself rejects at load, and it is rejected here for the same reason.
    if canvas_tokens == 0 and crop_tokens == 0:
        raise ValueError(
            "Tile Test: Render was given canvas_tokens 0 and crop_tokens 0. A tile needs vision "
            "rows from at least one of the two.")
    base = captions.resolve_method(surface)
    return replace(base, vision=replace(base.vision, canvas_tokens=canvas_tokens,
                                        crop_tokens=crop_tokens))


def _run_captions(surface, given, test_layout):
    # A caption belongs to a tile rect, so a set written for another grid would describe rects
    # this run never samples. The vision surface reads no captions at all. Returns the
    # captions object whole, since its style is joined onto the lanes only at _lane_captions.
    if surface == captions.VLM_METHOD_VISION:
        return None
    if given is None:
        raise ValueError(
            f"Tile Test: Render was asked for the {surface!r} surface with no captions connected. "
            "Feed it Tile Test: Captions, or pick the 'vision tokens' surface.")
    if given.target_size != test_layout.target_size:
        raise ValueError(
            f"Tile Test: Render was given captions written at {given.target_size[0]}x"
            f"{given.target_size[1]}, and the layout was solved for "
            f"{test_layout.target_size[0]}x{test_layout.target_size[1]}. Feed both nodes the same "
            "layout.")
    columns, rows = test_layout.layout.sol_x.n, test_layout.layout.sol_y.n
    if given.grid != (columns, rows):
        raise ValueError(
            f"Tile Test: Render was given captions for a {given.grid[0]}x{given.grid[1]} tile "
            f"grid, and this layout is {columns}x{rows}. Feed both nodes the same layout.")
    rects = tuple((tile.crop_rect.x0, tile.crop_rect.y0, tile.crop_rect.x1, tile.crop_rect.y1)
                  for tile in test_layout.layout.tiles)
    if given.rects != rects:
        raise ValueError(
            f"Tile Test: Render was given captions for a {given.grid[0]}x{given.grid[1]} tile grid "
            f"whose tile rects are not this {columns}x{rows} layout's. Feed both nodes the same "
            "layout.")
    return given


def _lane_captions(given, lanes):
    # The form the engine reads, joined HERE and nowhere earlier, so the captions object and
    # its listing keep the style apart from the tiles.
    if lanes is None:
        return None
    return tuple(tuple(rows) for rows in captions.join_style_captions(given.style, lanes))


def _captioned_tiles(tile_captions):
    return ", ".join(str(index) for index, rows in enumerate(tile_captions) if rows is not None)


def _full_captions(tile_captions):
    # The full run gives every tile a lane, so every tile needs its caption.
    if tile_captions is None or all(rows is not None for rows in tile_captions):
        return tile_captions
    raise ValueError(
        "Tile Test: Render was asked to render the entire canvas, and the captions cover tiles "
        f"{_captioned_tiles(tile_captions)} only. Clear tiles on Tile Test: Captions, or connect "
        "its tiles output to tiles here.")


def _tile_range(layout, index, with_neighbors):
    # The inclusive column and row bounds one tile is rendered over: its bordering tiles as
    # well, which is the lane set the full run gives it, or the tile on its own.
    if with_neighbors:
        return grid.neighborhood(layout, index)
    tile = layout.tiles[index]
    return tile.col, tile.col, tile.row, tile.row


def _clip_rect(rect, image):
    # A canvas rect as (x0, y0, x1, y1) inside the image. The layout is solved on the canvas
    # padded up to a multiple of 8, so an edge rect can reach past the picture itself.
    return (rect.x0, rect.y0, min(rect.x1, int(image.shape[2])), min(rect.y1, int(image.shape[1])))


def _cut(image, rect):
    x0, y0, x1, y1 = rect
    return image[:, y0:y1, x0:x1, :3].contiguous()


def _region_mask(sub, image):
    # The block is run through the engine's MASK path, so the region it denoises is handed over
    # as a mask. node._normalize_mask is what lands it on the image's device, which the region
    # composite in refine_sync multiplies against image tensors.
    height, width = int(image.shape[1]), int(image.shape[2])
    region = sub.region
    mask = torch.zeros((1, height, width), dtype=torch.float32)
    mask[:, region.y0:min(region.y1, height), region.x0:min(region.x1, width)] = 1.0
    return node._normalize_mask(mask, image)


def _check_region_crop(mask, sub, context_anchor, image):
    # The engine derives its own crop from the mask and places every lane rect inside it, so a
    # crop that is not the block would sample rects the full run never sampled.
    block = _clip_rect(sub.block, image)
    y0, y1, x0, x1 = sampling._expand_snap_clamp(
        sampling._mask_bbox(mask >= 0.5), context_anchor, int(image.shape[1]), int(image.shape[2]))
    if (x0, y0, x1, y1) != block:
        raise RuntimeError(
            f"Tile Test: Render built a mask the engine crops to {(x0, y0, x1, y1)}, and the block "
            f"is {block}. The two must be one rect, or the lanes sample what the full run never "
            "did.")


def _block_captions(tile_captions, sub, columns, index):
    # Every lane of the block carries the caption its tile was given in the PARENT grid, or a
    # lane would be conditioned on another tile's description.
    if tile_captions is None:
        return None
    parent_indices = tuple((sub.first_row + tile.row) * columns + sub.first_col + tile.col
                           for tile in sub.layout.tiles)
    missing = [parent for parent in parent_indices if tile_captions[parent] is None]
    if missing:
        raise ValueError(
            f"Tile Test: Render needs a caption for every lane of tile {index}'s block, and tiles "
            f"{', '.join(str(parent) for parent in missing)} have none. The captions cover tiles "
            f"{_captioned_tiles(tile_captions)}. Turn with_neighbors on at Tile Test: Captions, "
            "or off here.")
    return tuple(tile_captions[parent] for parent in parent_indices)


def _unsampled_lists(image, test_layout, requested):
    # denoise 0 samples nothing, and the two lists still carry one entry per requested tile, so
    # the number of pictures a workflow previews never depends on a widget that samples nothing.
    if not requested:
        untouched = image[..., :3].clone()
        return ([untouched], [untouched])
    tiles = [_cut(image, _clip_rect(test_layout.layout.tiles[index].crop_rect, image))
             for index, _sub in requested]
    blocks = [_cut(image, _clip_rect(sub.block, image)) for _index, sub in requested]
    return (tiles, blocks)


class ContextAnchoredTileTestRender:
    """Refine tiles of the layout through the production VL engine, for testing.

    This is a testing node and not one of the production nodes. With an empty tiles list it is
    a full run over the canvas and returns the refined picture. With a comma separated list of
    tile numbers it renders each named tile as one region run, over the block that tile and its
    bordering tiles occupy, and returns the tiles and the blocks as image lists.
    The geometry comes from the layout and the captions from Tile Test: Captions, so a tile can
    be re-rendered at a new seed or a new token count while every node above stays cached.
    """

    @classmethod
    def INPUT_TYPES(s):
        # Lazy import: this module's scope stays comfy-free (pinned by a subprocess test).
        import comfy.samplers

        vision = captions.load_settings().vision
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The canvas from Tile Test: Upscale, at the layout's target size."}),
                "layout": ("CATR_LAYOUT", {"tooltip": "The layout from Tile Test: Layout. Every rect this node samples is read from it."}),
                "model": ("MODEL", {"tooltip": "The diffusion model that denoises each tile."}),
                "clip": ("CLIP", {"tooltip": "Must be a vision-language text encoder (Krea 2 family). There is no positive prompt input, since each tile is conditioned on the image itself."}),
                "vae": ("VAE", {"tooltip": "The VAE that encodes and decodes each tile."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Noise is drawn once for the entire canvas and then sliced per tile, so a tile keeps the noise the full run would have given it."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "sgm_uniform"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "anchor_source": node._anchor_source(),
                "surface": (list(captions.VLM_SURFACES), {"default": captions.VLM_METHOD_VISION, "tooltip": "What fills every tile's positive. The two caption surfaces need Tile Test: Captions connected."}),
                "canvas_tokens": ("INT", {"default": vision.canvas_tokens, "min": 0, "max": vl.MAX_VISION_TOKENS, "tooltip": "Vision rows a tile takes from one encode of the entire image. 0 turns that source off."}),
                "crop_tokens": ("INT", {"default": vision.crop_tokens, "min": 0, "max": vl.MAX_VISION_TOKENS, "tooltip": "Vision rows a tile takes from the encode of its own crop. 0 turns that source off."}),
                "tiles": ("STRING", {"default": "", "tooltip": "Comma separated tile numbers, as Tile Test: Layout labels them. Empty renders the entire canvas as one run."}),
                "with_neighbors": ("BOOLEAN", {"default": True, "tooltip": "On renders the tile with its bordering tiles as lanes of one run, which is what the full run does. Off renders the tile alone, and its anchor ring then stays unrefined source, since no neighbour lane refines it."}),
            },
            "optional": {
                "captions": ("CATR_CAPTIONS", {"tooltip": "The captions from Tile Test: Captions. The two caption surfaces need them and the vision tokens surface ignores them."}),
                "negative": ("CONDITIONING", {"tooltip": "Optional negative conditioning. Unconnected it is an empty encode of this node's CLIP."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("tiles", "blocks")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "render_tiles"
    CATEGORY = "image/upscaling/tile testing"

    def render_tiles(self, image, layout, model, clip, vae, seed, sampler_name, scheduler, steps,
                     cfg, denoise, anchor_source, surface, canvas_tokens, crop_tokens, tiles,
                     with_neighbors, captions=None, negative=None):
        # Lazy import: this module's scope stays comfy-free (pinned by a subprocess test).
        import comfy.samplers

        # ---- inputs. Every rejection here runs before the first model call, since the text
        # encoder and the diffusion model each cost minutes to reach.
        node._validate_image(image)
        if image.shape[0] != 1:
            raise ValueError(
                f"Tile Test: Render renders one picture at a time, got a batch of {image.shape[0]}.")
        size = (int(image.shape[2]), int(image.shape[1]))
        if size != layout.target_size:
            raise ValueError(
                f"Tile Test: Render was given a {size[0]}x{size[1]} image, but the layout was "
                f"solved for {layout.target_size[0]}x{layout.target_size[1]}. Feed it the canvas "
                "Tile Test: Upscale returns.")
        preset = _run_preset(surface, canvas_tokens, crop_tokens)
        given = _run_captions(surface, captions, layout)
        tile_captions = None if given is None else given.captions
        # The sub layouts are solved HERE so grid.sub_layout's ring reach error reaches the user
        # before the text encoder loads.
        requested = tuple(
            (index, grid.sub_layout(layout.layout, *_tile_range(layout.layout, index, with_neighbors)))
            for index in _parse_tile_numbers(tiles, layout.tile_count, "Tile Test: Render"))
        # Which tiles need a caption is known once the blocks are, and the two checks run here
        # so a caption set that covers too few tiles is named before any model loads.
        if not requested:
            tile_captions = _lane_captions(given, _full_captions(tile_captions))
        columns = layout.layout.sol_x.n
        block_captions = {index: _lane_captions(given, _block_captions(tile_captions, sub, columns, index))
                          for index, sub in requested}

        # ---- process
        sigmas = upscale.build_sigmas(model, scheduler, steps, denoise)
        if sigmas.numel() < 2:
            return _unsampled_lists(image, layout, requested)
        empty = upscale.encode_empty(clip)
        guider = upscale.build_guider(model, empty, negative or empty, cfg)
        sampler = comfy.samplers.sampler_object(sampler_name)
        if not requested:
            refined = sampling.refine_image(
                image, guider, sampler, sigmas, vae, upscale.Noise_RandomNoise(seed),
                layout.max_tile_width, layout.max_tile_height, layout.context_anchor,
                layout.context_overlap, mask=None, vl_clip=clip, vlm_method=surface,
                anchor_source=anchor_source, sampler_name=sampler_name, preset=preset,
                tile_captions=tile_captions, layout=layout.layout)
            return ([refined], [refined])

        rendered_tiles = []
        rendered_blocks = []
        for index, sub in requested:
            mask = _region_mask(sub, image)
            _check_region_crop(mask, sub, layout.context_anchor, image)
            # The entire canvas' own draw sliced at the block, so a tile starts from the cells
            # the full run would have given it rather than from a draw of its own.
            noise = upscale.SlicedCanvasNoise(vae, seed, layout.layout.h, layout.layout.w, sub.block)
            refined = sampling.refine_image(
                image, guider, sampler, sigmas, vae, noise,
                layout.max_tile_width, layout.max_tile_height, layout.context_anchor,
                layout.context_overlap, mask=mask, vl_clip=clip, vlm_method=surface,
                anchor_source=anchor_source, sampler_name=sampler_name, preset=preset,
                tile_captions=block_captions[index], layout=sub,
                noise_fields=noise.noise_fields(sampler, sigmas))
            rendered_tiles.append(_cut(refined, _clip_rect(layout.layout.tiles[index].crop_rect, image)))
            rendered_blocks.append(_cut(refined, _clip_rect(sub.block, image)))

        # ---- output
        return (rendered_tiles, rendered_blocks)
