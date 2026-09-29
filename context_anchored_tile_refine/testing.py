"""The tile testing chain: the Settings node, the Layout node, the Upscale node, the Captions
node and the Render node.

`ContextAnchoredTileTestSettings` outputs every value of one settings file preset and the
[vision] table, one socket per settings key, read when it runs. Wired into the Captions and
Render nodes, any one value can be swapped for a text node without editing the file.

`ContextAnchoredTileTestLayout` solves the production grid for the size the image will have
AFTER upscaling and draws that grid over a preview, so the tile count and every tile's number
are visible before the upscale is spent. The solve is `sync._prepare_run`'s own: the target
size rounded up to a multiple of 8 on each axis, then the one `grid.solve_layout` call, so
what the overlay shows is what the engine will sample.

`ContextAnchoredTileTestUpscale` runs the production upscale stage as a node of its own. That
is what buys the chain its caching: ComfyUI holds the upscaled canvas, and a downstream re-run
at a new seed never pays for it again.

`ContextAnchoredTileTestCaptions` writes one text per chosen tile under one progress bar, from
its optional text sockets. A socket is on when it is connected and holds more than whitespace,
and off otherwise, so a stage is turned off by disconnecting it. tile_caption_instruction runs
the caption kind through the engine's own `captions.generate_caption_set`, and
tile_tags_instruction runs the tags kind through `tags.generate_tag_trace`, whose texts are
the ones the engine writes. Its tile_texts output lists the style caption once above every
tile's own text, and four debug outputs print the prompt tags, the listed tags, the
verification scores and the final tags per tile. Every text output is Markdown, for the Markdown
mode of Preview as Text. Its inputs carry no seed, so the texts
survive a seed re-roll further down the chain. Its tiles output feeds the Render node's tiles
input, so one tile is captioned and rendered together.

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
import re
from dataclasses import dataclass, replace
from typing import ClassVar

import torch

from . import captions, grid, node, progress, sampling, tags, upscale, vl

CAPTIONS_NODE = "Tile Test: Captions"

# The Captions node's instruction sockets in input order, each with the tile text kind that
# reads it, or None for a socket both kinds read.
INSTRUCTION_SOCKETS = (
    ("global_style_instruction", None),
    ("tile_caption_instruction", captions.TILE_TEXT_CAPTION),
    ("tile_tags_instruction", captions.TILE_TEXT_TAGS),
    ("prompt_tags_instruction", captions.TILE_TEXT_TAGS),
    ("tile_tags_verification_statement", captions.TILE_TEXT_TAGS),
)

# The overlay preview's pixel budget. Large enough to read a label on an 8K canvas, small
# enough that the picture stays a cheap preview.
OVERLAY_MEGAPIXELS = 2.0

# Every Tile Test: Settings output in order, as (settings key, socket type). The first seven
# are preset keys, the next three the [vision] table's, and the last three the tags thresholds.
SETTINGS_PRESET_OUTPUTS = (
    ("global_style_instruction", "STRING"),
    ("global_style_max_tokens", "INT"),
    ("tile_caption_instruction", "STRING"),
    ("tile_caption_max_tokens", "INT"),
    ("tile_tags_instruction", "STRING"),
    ("prompt_tags_instruction", "STRING"),
    ("tile_tags_verification_statement", "STRING"),
)
SETTINGS_VISION_OUTPUTS = (
    ("caption_megapixels", "FLOAT"),
    ("canvas_tokens", "INT"),
    ("crop_tokens", "INT"),
)
# Preset keys added after the node was first wired. They come after the vision outputs because
# a saved workflow links an output by its slot number, so an insert would move every later
# link onto another value of the same type.
SETTINGS_TAGS_THRESHOLD_OUTPUTS = (
    ("tile_tags_verification_threshold", "FLOAT"),
    ("tile_tags_position_threshold", "FLOAT"),
    ("prompt_tags_verification_threshold", "FLOAT"),
)
SETTINGS_OUTPUTS = (*SETTINGS_PRESET_OUTPUTS, *SETTINGS_VISION_OUTPUTS, *SETTINGS_TAGS_THRESHOLD_OUTPUTS)
# What a preset key outputs when the preset's kind does not carry it, so an unused socket
# holds an empty value rather than failing the run.
SETTINGS_ABSENT_VALUES = {"STRING": "", "INT": 0, "FLOAT": 0.0}
# Keyed by settings key rather than listed by slot, so an output added without its tooltip
# fails at import instead of shifting every later tooltip onto the wrong socket.
SETTINGS_OUTPUT_TOOLTIPS = {
    "global_style_instruction": f"The question the VL model is asked about the entire image for the style caption, with {captions.PROMPT_PLACEHOLDER} left in.",
    "global_style_max_tokens": "The generation budget for the style caption.",
    "tile_caption_instruction": f"The question the VL model is asked about each tile by the caption kind, with {captions.PROMPT_PLACEHOLDER} left in. A tags preset outputs an empty text.",
    "tile_caption_max_tokens": "The generation budget for each tile caption. A tags preset outputs 0.",
    "tile_tags_instruction": "The question that asks the VL model to list the things in each tile as tags. A caption preset outputs an empty text.",
    "prompt_tags_instruction": f"The question that asks the VL model to list the things the prompt names, with {captions.PROMPT_PLACEHOLDER} left in. A caption preset outputs an empty text.",
    "tile_tags_verification_statement": f"The statement each candidate tag is scored against on its tile, with {captions.TAG_PLACEHOLDER} left in. A caption preset outputs an empty text.",
    "caption_megapixels": "The picture size in megapixels the VL model reads for each caption, from the [vision] table.",
    "canvas_tokens": "The vision rows a tile takes from one encode of the entire image, from the [vision] table.",
    "crop_tokens": "The vision rows a tile takes from the encode of its own crop, from the [vision] table.",
    "tile_tags_verification_threshold": "The score a tag the tile's own list names must reach on the entire tile to be kept. A caption preset outputs 0.",
    "tile_tags_position_threshold": "The score a tag must reach on one of the six strips of a tile for that strip to hold it. A caption preset outputs 0.",
    "prompt_tags_verification_threshold": "The score a thing from the prompt that the tile's own list lacks must reach on the entire tile to be kept. A caption preset outputs 0.",
}

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
    size. `preset` is the label they were written from and `kind` that preset's tile text
    kind, so a tags set holds each tile's tag text where a caption set holds its caption.

    `str()` is the Markdown listing, which is what core's Preview as Text node shows for a
    value it cannot serialize as JSON, so the socket reads the same as the tile_texts output.
    """

    captions: tuple
    style: tuple | None
    tiles: tuple
    target_size: tuple
    grid: tuple
    rects: tuple
    preset: str
    kind: str = captions.TILE_TEXT_CAPTION

    def __str__(self):
        columns = self.grid[0]
        captioned = tuple(index for index, rows in enumerate(self.captions) if rows is not None)
        named = self.tiles or captioned
        bordering = tuple(index for index in captioned if index not in named)
        count = f"{len(captioned)} tiles" if len(captioned) == len(self.captions) else (
            f"{len(captioned)} of {len(self.captions)} tiles")
        title = f"## tile_texts\n{self.kind.capitalize()} kind, {count}"
        if bordering:
            title += f", named tiles {', '.join(str(index) for index in named)} first and then their bordering tiles"
        style = "Off, `global_style_instruction` is not connected."
        if self.style is not None:
            style = _quote(self.style[0])
        sections = [_section("### Style caption", ["Placed on top of every tile's text.", style])]
        sections.extend(_section(_tile_header(index, columns), [_quote(self.captions[index][0])])
                        for index in (*named, *bordering))
        return _debug_text(f"{title}.", "Tile Test: Render conditions each tile on its own text "
                           "below, with the style caption placed on top of it.", sections)


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


class ContextAnchoredTileTestSettings:
    """Output every value of one settings file preset and the [vision] table.

    This is a testing node and not one of the production nodes. Each output is named as its
    settings key and carries the file's value as written, with {PROMPT} and {TAG} left for
    the node that reads it. Wire the outputs into Tile Test: Captions and Tile Test: Render,
    and swap any one wire for a text node to try a wording without editing the file. A key
    the preset's kind does not carry outputs an empty text or 0.
    """

    @classmethod
    def INPUT_TYPES(s):
        labels = list(captions.preset_labels())
        return {
            "required": {
                "preset": (labels, {"default": labels[0], "tooltip": "Which settings file preset the outputs are read from, of the caption or the tags kind. The list is built when ComfyUI starts, so a new or renamed preset needs a restart. An edited wording reaches the outputs on the next queued run."}),
            },
        }

    RETURN_TYPES = tuple(kind for _, kind in SETTINGS_OUTPUTS)
    RETURN_NAMES = tuple(key for key, _ in SETTINGS_OUTPUTS)
    OUTPUT_TOOLTIPS = tuple(SETTINGS_OUTPUT_TOOLTIPS[key] for key, _ in SETTINGS_OUTPUTS)
    FUNCTION = "read_settings"
    CATEGORY = "image/upscaling/tile testing"
    SEARCH_ALIASES: ClassVar[list[str]] = ["tile settings", "caption presets", "settings file", "tile testing"]

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # ComfyUI folds this value into the node's cache key, and the file is read at run
        # time, so without it an edited wording never reaches the outputs.
        return captions.settings_fingerprint()

    def read_settings(self, preset):
        # ---- inputs. load_settings runs the file's own validation before any value leaves.
        settings = captions.load_settings()
        if preset not in settings.presets:
            raise RuntimeError(
                f"Tile Test: Settings was set to preset {preset!r}, which "
                f"{captions.settings_path()} does not define. It offers "
                f"{list(settings.presets)}. Restart ComfyUI to rebuild the preset list, or pick "
                "another preset.")
        block = settings.presets[preset]
        vision = settings.vision

        # ---- process
        preset_values = tuple(block.get(key, SETTINGS_ABSENT_VALUES[kind])
                              for key, kind in SETTINGS_PRESET_OUTPUTS)
        vision_values = tuple(getattr(vision, key) for key, _ in SETTINGS_VISION_OUTPUTS)
        threshold_values = tuple(float(block.get(key, SETTINGS_ABSENT_VALUES[kind]))
                                 for key, kind in SETTINGS_TAGS_THRESHOLD_OUTPUTS)

        # ---- output
        return (*preset_values, *vision_values, *threshold_values)


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
                **node._upscale_geometry("The upscale multiplier. The grid is solved for the size the image has after this upscale."),
            },
        }

    RETURN_TYPES = ("CATR_LAYOUT", "IMAGE", "INT")
    RETURN_NAMES = ("layout", "overlay", "tile_count")
    OUTPUT_TOOLTIPS = (
        "The solved tile grid and the widgets it was solved from. Feed it to Tile Test: Upscale, Tile Test: Captions and Tile Test: Render.",
        f"A preview of the image at its upscaled shape, with every tile's crop, overlap and core outlined and its number labeled. The preview is at most {OVERLAY_MEGAPIXELS:g} megapixels.",
        "The number of tiles in the grid.",
    )
    FUNCTION = "solve_layout"
    CATEGORY = "image/upscaling/tile testing"
    SEARCH_ALIASES: ClassVar[list[str]] = ["tile grid", "tile layout", "grid preview", "tile count", "tile testing"]

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
        test_layout = TestLayout(
            upscale_by=upscale_by,
            max_tile_width=max_tile_width,
            max_tile_height=max_tile_height,
            context_anchor=context_anchor,
            context_overlap=context_overlap,
            source_size=(source_width, source_height),
            target_size=(target_width, target_height),
            layout=grid.solve_layout(canvas_width, canvas_height, max_tile_width, max_tile_height,
                                     context_anchor, context_overlap),
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
    OUTPUT_TOOLTIPS = (
        "The upscaled canvas at the layout's target size. Feed it to Tile Test: Captions and Tile Test: Render.",
    )
    FUNCTION = "upscale_image"
    CATEGORY = "image/upscaling/tile testing"
    SEARCH_ALIASES: ClassVar[list[str]] = ["upscale", "upscale canvas", "tile testing"]

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


def _is_on(text):
    return text is not None and bool(text.strip())


def _text_kind(texts):
    # The kind is read off which tile socket is on. A socket the other kind reads that is on
    # is refused rather than ignored, so nothing connected changes nothing silently.
    caption_on = bool(texts["tile_caption_instruction"])
    tags_on = bool(texts["tile_tags_instruction"])
    if caption_on and tags_on:
        raise ValueError(
            f"{CAPTIONS_NODE} has tile_caption_instruction and tile_tags_instruction both "
            "connected. Disconnect one of them: tile_caption_instruction writes a caption per "
            "tile and tile_tags_instruction writes tags per tile.")
    if not caption_on and not tags_on:
        raise ValueError(
            f"{CAPTIONS_NODE} has neither tile_caption_instruction nor tile_tags_instruction "
            "connected with text. Connect one of them, for example from the matching output of "
            "Tile Test: Settings.")
    kind = captions.TILE_TEXT_CAPTION if caption_on else captions.TILE_TEXT_TAGS
    for name, reader in INSTRUCTION_SOCKETS:
        if texts[name] and reader not in (None, kind):
            raise ValueError(
                f"{CAPTIONS_NODE} runs the {kind} kind, and {name} is connected, which only the "
                f"{reader} kind reads. Disconnect {name}, or connect tile_tags_instruction in "
                "place of tile_caption_instruction.")
    return kind


def _check_tags_texts(texts):
    # The settings file's placeholder rules for a tags block, applied to the sockets, since a
    # socket can carry any text node's wording.
    for name in ("tile_tags_instruction", "tile_tags_verification_statement"):
        if captions.PROMPT_PLACEHOLDER in texts[name]:
            raise ValueError(
                f"{CAPTIONS_NODE} was given {name} holding {captions.PROMPT_PLACEHOLDER}. Only "
                "prompt_tags_instruction carries the prompt, so move "
                f"{captions.PROMPT_PLACEHOLDER} there.")
    question = texts["prompt_tags_instruction"]
    if question and captions.PROMPT_PLACEHOLDER not in question:
        raise ValueError(
            f"{CAPTIONS_NODE} was given prompt_tags_instruction without "
            f"{captions.PROMPT_PLACEHOLDER}. Write {captions.PROMPT_PLACEHOLDER} where the prompt "
            "goes, or disconnect it to list no things from the prompt.")
    statement = texts["tile_tags_verification_statement"]
    if statement and captions.TAG_PLACEHOLDER not in statement:
        raise ValueError(
            f"{CAPTIONS_NODE} was given tile_tags_verification_statement without "
            f"{captions.TAG_PLACEHOLDER}. Write {captions.TAG_PLACEHOLDER} where each candidate "
            "tag goes, or disconnect it to keep every candidate unchecked.")


def _check_budget(name, value):
    # A linked value bypasses the widget's min, and a tags Tile Test: Settings node outputs 0
    # for the caption budget it does not carry.
    if value < 1:
        raise ValueError(
            f"{CAPTIONS_NODE} was given {name} {value}, and a caption needs a budget of 1 token "
            f"or more. Set {name} to 1 or more, or link it from a Tile Test: Settings node whose "
            "preset carries it.")


def _check_score(name, value, above_zero=False):
    # A linked value bypasses the widget's min and max. The tagging stages take a verification
    # threshold above 0 only.
    floor_ok = value > 0 if above_zero else value >= 0
    span = "above 0 and at most 1" if above_zero else "between 0 and 1"
    if not (floor_ok and value <= 1):
        raise ValueError(
            f"{CAPTIONS_NODE} was given {name} {value}, and it is a score {span}. Set it {span}, "
            "or link it from a Tile Test: Settings node whose preset carries it.")


def _caption_megapixels_error(value):
    # The widget's own range cannot express "0 or at least VL_INPUT_MIN_MEGAPIXELS".
    if value == 0 or captions.VL_INPUT_MIN_MEGAPIXELS <= value <= vl.PICTURE_CAP_MEGAPIXELS:
        return None
    return (
        "caption_megapixels must be 0, which reads the picture's own size, or between "
        f"{captions.VL_INPUT_MIN_MEGAPIXELS} and {vl.PICTURE_CAP_MEGAPIXELS}. Got {value}.")


def _check_caption_megapixels(value):
    # A linked value reaches VALIDATE_INPUTS as None, so the rule runs again here.
    error = _caption_megapixels_error(value)
    if error is not None:
        raise ValueError(
            f"{CAPTIONS_NODE}: {error} Set it in that range, or link it from a Tile Test: "
            "Settings node.")


def _socket_preset(texts, prompt, tile_caption_max_tokens, global_style_max_tokens,
                   caption_megapixels, verification_threshold, position_threshold,
                   prompt_verification_threshold):
    # One preset built from the sockets, checked in full before any model call, since the
    # text encoder costs minutes to reach the same rejection.
    kind = _text_kind(texts)
    style = texts["global_style_instruction"]
    _check_caption_megapixels(caption_megapixels)
    vision = replace(captions.load_settings().vision, caption_megapixels=caption_megapixels)
    if style:
        _check_budget("global_style_max_tokens", global_style_max_tokens)
    if kind == captions.TILE_TEXT_TAGS:
        _check_tags_texts(texts)
        _check_score("tile_tags_verification_threshold", verification_threshold, above_zero=True)
        _check_score("tile_tags_position_threshold", position_threshold)
        _check_score("prompt_tags_verification_threshold", prompt_verification_threshold, above_zero=True)
        preset = captions.Preset(
            surface=captions.VLM_METHOD_CAPTIONS,
            label=CAPTIONS_NODE,
            vision=vision,
            style_instruction=style,
            style_max_tokens=global_style_max_tokens,
            kind=captions.TILE_TEXT_TAGS,
            tile_tags_instruction=texts["tile_tags_instruction"],
            prompt_tags_instruction=texts["prompt_tags_instruction"],
            tile_tags_verification_statement=texts["tile_tags_verification_statement"],
            tile_tags_verification_threshold=verification_threshold,
            prompt_tags_verification_threshold=prompt_verification_threshold,
            tile_tags_position_threshold=position_threshold,
        )
    else:
        _check_budget("tile_caption_max_tokens", tile_caption_max_tokens)
        preset = captions.Preset(
            surface=captions.VLM_METHOD_CAPTIONS,
            label=CAPTIONS_NODE,
            vision=vision,
            tile_instruction=texts["tile_caption_instruction"],
            tile_max_tokens=tile_caption_max_tokens,
            style_instruction=style,
            style_max_tokens=global_style_max_tokens,
        )
    return captions.with_prompt(preset, prompt)


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


# --- the Captions node's text outputs, written as Markdown for Preview as Text's Markdown
# mode. Pure functions over the tags traces, one picture row.

PROMPT_TAGS_TITLE = ("## prompt_tags\nThe things the VL model listed from the connected prompt, "
                     "which every tile checks.")
LISTED_TITLE = "## tags_listed\nThe tags the VL model listed for each tile."
VERIFIED_TITLE = "## tags_verified\nEach candidate tag scored on its tile."
FINAL_TITLE = "## tags_final\nThe kept tags of each tile, their positions and the tile text."
NO_TAG_STAGES = ("The caption kind has no tag stages, and connecting `tile_tags_instruction` in "
                 "place of `tile_caption_instruction` runs them.")
VERIFICATION_OFF = "off, `tile_tags_verification_statement` is not connected"
ORIGIN_NAMES = {"prompt": "prompt", "both": "prompt and VL model", "model": "VL model"}
# The library rounds every noul to 6 digits, so 6 digits show every score exactly. A threshold with more digits acts
# as its 6 digit ceiling.
SCORE_DIGITS = 6

# Model text is escaped so that it shows as written. A tag such as <think> would otherwise be
# removed by the frontend's HTML sanitizer, and a line starting with "-" would become a list.
_MARKDOWN_CHARACTERS = re.compile(r"([\\`*_\[\]<>|~])")
_LINE_START_MARKER = re.compile(r"^(\s*)(?:([#+=-])|(\d+)([.)]))")


def _p(value, digits=2):
    # Truncated, not rounded, so a shown score never reaches a threshold its row fell below.
    # The inner round absorbs float error such as 0.29 * 100 == 28.999999999999996.
    scale = 10 ** digits
    return f"{math.floor(round(value * scale, 6)) / scale:.{digits}f}"


def _escaped(line):
    line = _MARKDOWN_CHARACTERS.sub(r"\\\1", line)
    return _LINE_START_MARKER.sub(lambda m: f"{m[1]}\\{m[2]}" if m[2] else f"{m[1]}{m[3]}\\{m[4]}", line)


def _quote(text):
    # A blockquote wraps long lines where a code block scrolls. A trailing backslash is a hard
    # line break, so the model's own line breaks survive.
    if not text:
        return "> *no text was written*"
    lines = text.split("\n")
    quoted = []
    for position, line in enumerate(lines):
        next_has_text = position + 1 < len(lines) and lines[position + 1].strip()
        hard_break = "\\" if line.strip() and next_has_text else ""
        quoted.append(f"> {_escaped(line)}{hard_break}".rstrip())
    return "\n".join(quoted)


def _table(header, rows):
    if not rows:
        return "None."
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _tag_list(items):
    return ", ".join(_escaped(item) for item in items) or "none"


def _section(header, blocks):
    return "\n\n".join([header, *blocks])


def _debug_text(title, sentence, sections):
    return "\n\n".join([f"{title}\n{sentence}", *sections])


def _tile_header(index, columns):
    # The overlay's own number, so a text is found by the label drawn on the tile.
    return f"### Tile {index}, row {index // columns}, column {index % columns}"


def _prompt_tags_debug(preset, run):
    settings = tags.tag_settings(preset)
    sentence = (f"A prompt the VL model scores below {_p(settings.english_threshold)} on "
                f'"{settings.language_question}" is translated to English first. The VL model lists the '
                "things the English text names, a listed tag with a word the English text lacks is "
                f"dropped, and the thing check drops a tag at p(other) {_p(settings.thing_threshold)} or "
                "above. The kept tags join every tile's candidates.")
    translation = []
    rows = []

    if run.prompt is None:
        reason = "No prompt is connected." if not preset.prompt else "`prompt_tags_instruction` is not connected."
        return _debug_text(PROMPT_TAGS_TITLE, sentence, [_section("### Prompt", [reason])])
    if run.prompt.text != preset.prompt:
        translation = [_section("### English translation", [_quote(run.prompt.text)])]
    rows = [("kept" if tag in run.prompt.tags else "dropped", _p(p), _escaped(tag))
            for tag, p in zip(run.prompt.listed, run.prompt.p_other, strict=True)]
    return _debug_text(PROMPT_TAGS_TITLE, sentence, [
        *translation,
        _section("### Question sent to the VL model once per picture",
                 [_quote(tags.prompt_tags_question(preset, run.prompt.text))]),
        _section("### VL model reply", [_quote(run.prompt.reply)]),
        _section("### Listed tags", [_table(("Result", "p(other)", "Tag"), rows)])])


def _listed_block(header, trace, fallback_question):
    replies = ["**VL model reply**", _quote(trace.reply)]
    parsed_from = "the reply"

    if trace.echo_reply is not None:
        replies = ["**VL model reply**", _quote(trace.echo_reply), "**Fallback question**", _quote(fallback_question),
                   "**VL model reply to the fallback question**", _quote(trace.reply)]
        parsed_from = "the fallback reply"
    return _section(header, [*replies, f"**Tags parsed from {parsed_from} ({len(trace.proposed)}):** "
                                       f"{_tag_list(trace.proposed)}"])


def _listed_debug(preset, headers, traces):
    fallback_question = tags.tag_settings(preset).echo_fallback_instruction
    sections = [_section("### Question sent to the VL model for every tile", [_quote(preset.tile_tags_instruction)])]
    sections.extend(_listed_block(header, trace, fallback_question)
                    for header, trace in zip(headers, traces, strict=True))
    return _debug_text(LISTED_TITLE, "The VL model is asked the question below about each tile, "
                       f"its list is stopped after {tags.MAX_PROPOSED_TAGS} tags, and its reply is "
                       "split into tags. A reply with no tag besides category nouns is replaced by "
                       "the reply to a fallback question.", sections)


def _verified_block(header, trace, threshold, prompt_threshold):
    rows = tuple(zip(trace.candidates, trace.scores, trace.origins, strict=True))
    # The prompt's rows come first, since a prompt tag is the one a wording change moves.
    ordered = [row for row in rows if row[2] != "model"] + [row for row in rows if row[2] == "model"]
    table = [(ORIGIN_NAMES[origin],
              "kept" if p >= (prompt_threshold if origin == "prompt" else threshold) else "dropped",
              _p(p, SCORE_DIGITS), _escaped(item)) for item, p, origin in ordered]
    left_out = ", ".join(f"{_escaped(name)} ({reason})" for name, reason in trace.dropped) or "none"
    return _section(header, [_table(("Source", "Result", "Score", "Tag"), table),
                             f"**Left out before verification:** {left_out}"])


def _unchecked_block(header, trace):
    return _section(header, [f"**Kept unchecked ({len(trace.candidates)}):** {_tag_list(trace.candidates)}"])


def _verified_debug(preset, headers, traces):
    if not preset.tile_tags_verification_statement:
        return _debug_text(VERIFIED_TITLE, "`tile_tags_verification_statement` is not connected, so "
                           "verification is off and no candidate was scored.", [_unchecked_block(header, trace)
                                           for header, trace in zip(headers, traces, strict=True)])
    threshold = preset.tile_tags_verification_threshold
    prompt_threshold = preset.prompt_tags_verification_threshold
    sentence = ("Each candidate is scored with `tile_tags_verification_statement` on its tile. A tag "
                f"the VL model listed is kept at {_p(threshold, SCORE_DIGITS)} or above "
                "(`tile_tags_verification_threshold`), and a prompt tag it did not list at "
                f"{_p(prompt_threshold, SCORE_DIGITS)} or above (`prompt_tags_verification_threshold`).")
    return _debug_text(VERIFIED_TITLE, sentence, [_verified_block(header, trace, threshold, prompt_threshold)
                                                  for header, trace in zip(headers, traces, strict=True)])


def _placement(item, term, unplaced):
    if item in unplaced:
        return "dropped, no strip holds it"
    if not term:
        return "kept with no term"
    return f"kept at {term}"


def _axis_cell(strips, threshold):
    # Bold marks a strip that holds the tag, the same rule tags.axis_word applies.
    return ", ".join(f"**{_p(p, SCORE_DIGITS)}**" if p >= threshold else _p(p, SCORE_DIGITS) for p in strips)


def _positions_table(trace, threshold):
    header = ("Tag", "Result", f"Rows ({', '.join(tags.ROW_WORDS)})",
              f"Columns ({', '.join(tags.COLUMN_WORDS)})")
    rows = [(_escaped(item), _placement(item, term, trace.unplaced),
             _axis_cell(strips[:3], threshold), _axis_cell(strips[3:], threshold))
            for item, strips, term in zip(trace.kept, trace.strips, trace.terms, strict=True)]
    return _table(header, rows)


def _final_block(header, trace, positions_on, threshold):
    subsets = [item for item in trace.verified if item not in trace.kept]
    table = [_positions_table(trace, threshold)] if positions_on else []
    return _section(header, [f"**Dropped as a subset of a longer kept tag:** {_tag_list(subsets)}",
                             *table, "**Tile text**", _quote(trace.text)])


def _final_debug(preset, headers, traces, locate):
    threshold = preset.tile_tags_position_threshold
    positions_off = ""
    if not preset.tile_tags_verification_statement:
        positions_off = VERIFICATION_OFF
    elif not locate:
        positions_off = "off, `position_terms` is off"
    positions = (f"Positions are {positions_off}. The remaining tags make the tile text."
                 if positions_off else
                 "Each remaining tag is scored on six strips of the tile, three rows and three "
                 f"columns. A strip holds a tag at {_p(threshold, SCORE_DIGITS)} or above "
                 "(`tile_tags_position_threshold`). A tag no strip holds is dropped. On each axis "
                 "where exactly one strip holds a tag, that strip names the tag's position term. "
                 "A center column adds no word beside a row word. "
                 "The tags with their terms make the tile text.")
    sentence = f"A kept tag that is part of a longer kept tag is dropped. {positions}"
    return _debug_text(FINAL_TITLE, sentence, [_final_block(header, trace, not positions_off, threshold)
                                               for header, trace in zip(headers, traces, strict=True)])


def _tags_debug(preset, headers, run, locate):
    # (prompt_tags, tags_listed, tags_verified, tags_final).
    traces = [rows[0] for rows in run.tiles]
    return (_prompt_tags_debug(preset, run), _listed_debug(preset, headers, traces),
            _verified_debug(preset, headers, traces), _final_debug(preset, headers, traces, locate))


def _caption_debug():
    # The caption kind has no tag stages, so each tags output says so in one sentence.
    prompt_tags = _debug_text(PROMPT_TAGS_TITLE, "The caption kind reads the prompt only through "
                              f"{captions.PROMPT_PLACEHOLDER} in its instructions, so no things are "
                              "listed from it.", [])
    return (prompt_tags, *(_debug_text(title, NO_TAG_STAGES, [])
                         for title in (LISTED_TITLE, VERIFIED_TITLE, FINAL_TITLE)))


class ContextAnchoredTileTestCaptions:
    """Write tile texts for tiles of the layout, through the engine's own caption or tags pass.

    This is a testing node and not one of the production nodes. Every instruction is an
    optional text socket, which is on when it is connected and holds more than whitespace, so
    a stage is turned off by disconnecting it and a wording is tried by wiring a text node in
    its place. tile_caption_instruction runs the caption kind and tile_tags_instruction the
    tags kind, and exactly one of the two must be on. Wire them from Tile Test: Settings.
    tile_texts lists the texts Tile Test: Render conditions on, and prompt_tags, tags_listed,
    tags_verified and tags_final print every tags stage per tile. An empty tiles
    list captions every tile. A tile list captions those tiles, plus their bordering tiles
    when with_neighbors is on, which is what Tile Test: Render needs to render one of them
    with its neighbours. There is no seed here, so ComfyUI serves the texts from its cache
    while a seed is re-rolled further down the chain. Feed the captions and the tiles to Tile
    Test: Render.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The canvas from Tile Test: Upscale, at the layout's target size."}),
                "layout": ("CATR_LAYOUT", {"tooltip": "The layout from Tile Test: Layout. The tiles are captioned from it."}),
                "clip": ("CLIP", {"tooltip": "Must be a vision-language text encoder with a text generator (Krea 2 family)."}),
                "tiles": ("STRING", {"default": "", "tooltip": "Comma separated tile numbers, as Tile Test: Layout labels them. Empty captions every tile. The same list comes out of the tiles output for Tile Test: Render."}),
                "with_neighbors": ("BOOLEAN", {"default": True, "tooltip": "Caption the bordering tiles of every named tile as well, which Tile Test: Render needs when its with_neighbors is on. Off captions the named tiles only."}),
                "position_terms": ("BOOLEAN", {"default": True, "tooltip": "Score each kept tag on six strips of its tile, drop a tag no strip holds, and write a position term such as top-left. Off writes each tag without a term, makes no strip requests and drops no tag for its strips. Read by the tags kind when tile_tags_verification_statement is connected, and ignored by the caption kind."}),
                "caption_megapixels": ("FLOAT", {"default": captions.load_settings().vision.caption_megapixels, "min": 0.0, "max": vl.PICTURE_CAP_MEGAPIXELS, "step": 0.01, "tooltip": f"How much of the picture the VL model reads for every caption this node writes, the tile captions and the style caption. Use 0 for the picture's own size, capped at {vl.PICTURE_CAP_MEGAPIXELS} megapixels. The tags kind reads it for the style caption only, since its tags questions read a fixed copy of about 1 megapixel. Can take the caption_megapixels output of Tile Test: Settings."}),
                "tile_caption_max_tokens": ("INT", {"default": 768, "min": 1, "max": captions.MAX_CAPTION_TOKENS, "tooltip": "Generation budget for each tile caption, which also covers the model's hidden reasoning turn. Read by the caption kind and ignored by the tags kind. Can take the tile_caption_max_tokens output of Tile Test: Settings."}),
                "global_style_max_tokens": ("INT", {"default": 768, "min": 1, "max": captions.MAX_CAPTION_TOKENS, "tooltip": "Generation budget for the style caption, which also covers the model's hidden reasoning turn. Read when global_style_instruction is connected. Can take the global_style_max_tokens output of Tile Test: Settings."}),
                "tile_tags_verification_threshold": ("FLOAT", {"default": captions.SHIPPED_TAGS_VERIFICATION_THRESHOLD, "min": 0.00001, "max": 1.0, "step": 0.00001, "tooltip": "The score tile_tags_verification_statement must reach on the entire tile to keep a tag the tile's own list names. Read by the tags kind when tile_tags_verification_statement is connected. Can take the tile_tags_verification_threshold output of Tile Test: Settings."}),
                "tile_tags_position_threshold": ("FLOAT", {"default": captions.SHIPPED_TAGS_POSITION_THRESHOLD, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The score tile_tags_verification_statement must reach on one of the six strips of a tile for that strip to hold a tag. A tag no strip holds is dropped, and a strip that is the only one holding a tag on its axis names the tag's position term. A center column adds no word beside a row word. Read by the tags kind when position_terms is on. Can take the tile_tags_position_threshold output of Tile Test: Settings."}),
                "prompt_tags_verification_threshold": ("FLOAT", {"default": captions.SHIPPED_PROMPT_TAGS_VERIFICATION_THRESHOLD, "min": 0.00001, "max": 1.0, "step": 0.00001, "tooltip": "The score tile_tags_verification_statement must reach on the entire tile to keep a thing from the prompt that the tile's own list lacks. Read by the tags kind when prompt_tags_instruction and tile_tags_verification_statement are connected. Can take the prompt_tags_verification_threshold output of Tile Test: Settings."}),
            },
            "optional": {
                "prompt": node._prompt(),
                "global_style_instruction": ("STRING", {"forceInput": True, "tooltip": f"What the VL model is asked about the entire image, for one style caption placed on top of every tile's text, in either kind. {captions.PROMPT_PLACEHOLDER} is filled from prompt. Unconnected, or holding only whitespace, writes no style caption."}),
                "tile_caption_instruction": ("STRING", {"forceInput": True, "tooltip": f"What the VL model is asked about each tile, which runs the caption kind. {captions.PROMPT_PLACEHOLDER} is filled from prompt. Connect this or tile_tags_instruction, never both. Unconnected leaves the caption kind off."}),
                "tile_tags_instruction": ("STRING", {"forceInput": True, "tooltip": f"The question that asks the VL model to list the things in each tile as comma separated tags, which runs the tags kind. It cannot hold {captions.PROMPT_PLACEHOLDER}. Connect this or tile_caption_instruction, never both. Unconnected leaves the tags kind off."}),
                "prompt_tags_instruction": ("STRING", {"forceInput": True, "tooltip": f"The question that asks the VL model to list the things the prompt names, once per picture. It must hold {captions.PROMPT_PLACEHOLDER}, where the prompt goes. Every tile checks the listed things. Read by the tags kind when prompt is connected. Unconnected lists no things from the prompt."}),
                "tile_tags_verification_statement": ("STRING", {"forceInput": True, "tooltip": f"The statement each candidate tag is scored true or false against on its tile. It must hold {captions.TAG_PLACEHOLDER}, where the tag goes, and a tag is kept at tile_tags_verification_threshold. Read by the tags kind only. Unconnected keeps every candidate unchecked and writes no position terms."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("CATR_CAPTIONS", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("captions", "tile_texts", "tiles", "prompt_tags", "tags_listed",
                    "tags_verified", "tags_final")
    OUTPUT_TOOLTIPS = (
        "The style caption and every captioned tile's text, for the captions input of Tile Test: Render.",
        "Markdown that lists the style caption once and then each captioned tile's own text, the texts Tile Test: Render conditions on.",
        "The named tile numbers, comma separated, for the tiles input of Tile Test: Render. It is empty when the tiles input was empty.",
        "Markdown of the tags kind's prompt stage: the question sent once per picture, the VL model's reply and whether each listed tag was kept.",
        "Markdown of the tags kind's listing stage: each tile's VL model reply and the tags parsed from it.",
        "Markdown of the tags kind's verification stage: each tile's candidate tags with their scores and whether each was kept.",
        "Markdown of the tags kind's last stage: each tile's tags after the subset and position checks, and the tile text they make.",
    )
    FUNCTION = "caption_tiles"
    CATEGORY = "image/upscaling/tile testing"
    SEARCH_ALIASES: ClassVar[list[str]] = ["tile captions", "tile tags", "caption debug", "tag debug", "tile testing"]

    @classmethod
    def VALIDATE_INPUTS(s, caption_megapixels=None):
        # Naming a widget here disables ComfyUI's own min and max check for it (node.py's
        # VALIDATE_INPUTS states the rule), so the settings file's entire rule is re-checked.
        if caption_megapixels is None:
            return True
        return _caption_megapixels_error(caption_megapixels) or True

    @classmethod
    def IS_CHANGED(s, **kwargs):
        # ComfyUI folds this value into the node's cache key, and the preset's [vision] table
        # is read from the settings file at run time, so the key follows the file.
        return captions.settings_fingerprint()

    def caption_tiles(self, image, layout, clip, tiles, with_neighbors, position_terms,
                      caption_megapixels, tile_caption_max_tokens, global_style_max_tokens,
                      tile_tags_verification_threshold, tile_tags_position_threshold,
                      prompt_tags_verification_threshold,
                      prompt=None, global_style_instruction=None, tile_caption_instruction=None,
                      tile_tags_instruction=None, prompt_tags_instruction=None,
                      tile_tags_verification_statement=None, unique_id=None):
        # ---- inputs. Every rejection here runs before the first model call.
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
        sockets = {
            "global_style_instruction": global_style_instruction,
            "tile_caption_instruction": tile_caption_instruction,
            "tile_tags_instruction": tile_tags_instruction,
            "prompt_tags_instruction": prompt_tags_instruction,
            "tile_tags_verification_statement": tile_tags_verification_statement,
        }
        texts = {name: value if _is_on(value) else "" for name, value in sockets.items()}
        run_preset = _socket_preset(texts, prompt, tile_caption_max_tokens,
                                    global_style_max_tokens, caption_megapixels,
                                    tile_tags_verification_threshold, tile_tags_position_threshold,
                                    prompt_tags_verification_threshold)
        is_tags = run_preset.kind == captions.TILE_TEXT_TAGS
        all_tiles = layout.layout.tiles
        columns = layout.layout.sol_x.n
        named, captioned = _tiles_to_caption(layout.layout, tiles, with_neighbors)
        chosen = [all_tiles[index] for index in captioned]
        headers = [_tile_header(index, columns) for index in captioned]

        # ---- process. The engine captions the PADDED canvas, so this pass must too, or a
        # tile's caption would describe a crop the run never reads. The ledger is what makes
        # the run ONE bar: core builds a per-token bar inside every clip.generate, and the
        # ledger's shim routes it into the caption's own chunk instead of resetting the display.
        padded, _ = sampling.pad_image_to_multiple(image)
        ledger = progress.build_caption_ledger(run_preset, len(chosen), unique_id=unique_id)
        with ledger:
            if is_tags:
                tag_run = tags.generate_tag_trace(
                    clip, padded, chosen, run_preset, progress=ledger, locate=position_terms)
                style, written = tags.tag_texts(tag_run)
                debug = _tags_debug(run_preset, headers, tag_run, position_terms)
            else:
                style, written = captions.generate_caption_set(clip, padded, chosen, run_preset,
                                                               progress=ledger)
                debug = _caption_debug()
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
            grid=(columns, layout.layout.sol_y.n),
            rects=tuple((tile.crop_rect.x0, tile.crop_rect.y0, tile.crop_rect.x1, tile.crop_rect.y1)
                        for tile in all_tiles),
            preset=run_preset.label,
            kind=run_preset.kind,
        )
        return (result, str(result), ", ".join(str(index) for index in named), *debug)


def _run_preset(surface, canvas_tokens, crop_tokens):
    # The settings file's block with the two widgets written over its [vision] table, so a
    # token count can be tried without editing the file. Both counts at 0 is what the file
    # itself rejects at load, and it is rejected here for the same reason.
    # A linked value bypasses the widget's min and max.
    for name, value in (("canvas_tokens", canvas_tokens), ("crop_tokens", crop_tokens)):
        if not 0 <= value <= vl.MAX_VISION_TOKENS:
            raise ValueError(
                f"Tile Test: Render was given {name} {value}, and it must be between 0 and "
                f"{vl.MAX_VISION_TOKENS}. Set {name} in that range.")
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
    return image[:, y0:y1, x0:x1, :3].clone()


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
        vision = captions.load_settings().vision
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The canvas from Tile Test: Upscale, at the layout's target size."}),
                "layout": ("CATR_LAYOUT", {"tooltip": "The layout from Tile Test: Layout. Every rect this node samples is read from it."}),
                "model": ("MODEL", {"tooltip": "The diffusion model that denoises each tile."}),
                "clip": ("CLIP", {"tooltip": "Must be a vision-language text encoder (Krea 2 family). There is no positive prompt input, since each tile is conditioned on the image itself."}),
                "vae": ("VAE", {"tooltip": "The VAE that encodes and decodes each tile."}),
                **node._sampling_widgets(),
                "anchor_source": node._anchor_source(),
                "surface": (list(captions.VLM_SURFACES), {"default": captions.VLM_METHOD_VISION_CAPTIONS, "tooltip": "What fills every tile's positive. The two caption surfaces need Tile Test: Captions connected. The VL nodes use the vision tokens and captions surface by default."}),
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
    OUTPUT_TOOLTIPS = (
        "One image per named tile, the refined canvas cut at the tile's crop rect. An empty tiles input returns the entire refined canvas.",
        "One image per named tile, the refined block rendered for it, which spans its bordering tiles when with_neighbors is on. An empty tiles input returns the entire refined canvas.",
    )
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "render_tiles"
    CATEGORY = "image/upscaling/tile testing"
    SEARCH_ALIASES: ClassVar[list[str]] = ["tile render", "render one tile", "tile preview", "tile testing"]

    @classmethod
    def VALIDATE_INPUTS(s, sampler_name=None):
        # The production nodes' queue-time sampler check, so a sampler the engine cannot time is
        # named before the text encoder and the diffusion model load.
        return node.check_sampler(sampler_name)

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
