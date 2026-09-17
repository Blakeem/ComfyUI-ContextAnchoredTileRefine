"""The tile testing chain: what the Layout node solves and draws, what the Upscale node
forwards, what the Captions node asks the VL model, and what the Render node hands the engine.

The grid math itself is covered by test_grid, the caption pipeline by test_captions and the
engine by test_sync. What is pinned here is that the Layout node solves the PRODUCTION grid on
the padded target size, that the overlay it draws stays a readable preview, that the Upscale
node runs the production upscale stage with the multiplier the layout carries, that the
Captions node captions the padded canvas from the preset the widgets ask for, and that the
Render node reaches sampling.refine_image with the layout's own rects, captions and noise
slice. The Render node's collaborators are all replaced by recorders, so what is pinned there
is which value reaches which parameter and not any pixel math."""
from types import SimpleNamespace

import pytest
import torch
from test_captions import FakeCaptionClip

from context_anchored_tile_refine import captions, grid, progress, sampling, testing, upscale, vl
from context_anchored_tile_refine.node import ContextAnchoredTileRefine
from context_anchored_tile_refine.testing import (
    ContextAnchoredTileTestCaptions,
    ContextAnchoredTileTestLayout,
    ContextAnchoredTileTestRender,
    ContextAnchoredTileTestUpscale,
)

WIDGETS = {
    "upscale_by": 2.0,
    "max_tile_width": 1536,
    "max_tile_height": 2048,
    "context_anchor": 32,
    "context_overlap": 32,
}


def _solve(image=None, **overrides):
    """Run the Layout node. Returns its three outputs named."""
    widgets = dict(WIDGETS)
    widgets.update(overrides)
    picture = torch.rand(1, 700, 1000, 3) if image is None else image
    layout, overlay, tile_count = ContextAnchoredTileTestLayout().solve_layout(image=picture, **widgets)
    return SimpleNamespace(layout=layout, overlay=overlay, tile_count=tile_count)


def test_the_layout_is_the_production_solve_on_the_padded_target():
    # The engine pads the canvas to /8 and then solves (sync._prepare_run). A layout solved on
    # anything else would draw a grid the refine never samples.
    result = _solve()

    sx = grid.solve_axis(2000, 1536, 32, 32, axis="width")
    sy = grid.solve_axis(1400, 2048, 32, 32, axis="height")
    assert result.layout.layout == grid.build_layout(2000, 1400, sx, sy, 32, 32)


def test_an_odd_target_size_is_padded_up_to_a_multiple_of_8():
    # 1001 * 2.0 = 2002, which the engine pads to 2008 before solving.
    result = _solve(image=torch.rand(1, 701, 1001, 3))

    assert result.layout.target_size == (2002, 1402)
    assert (result.layout.layout.w, result.layout.layout.h) == (2008, 1408)


def test_sizes_and_tile_count_for_a_1000x700_image_at_2x():
    result = _solve()

    assert result.layout.source_size == (1000, 700)
    assert result.layout.target_size == (2000, 1400)
    # 2000 px of width needs two 1000 px cores under the 1536 cap. 1400 px of height fits one.
    assert result.tile_count == 2
    assert result.layout.tile_count == 2


def test_the_widgets_ride_along_on_the_layout():
    # Every node below reads its geometry from the LAYOUT, so the widgets must reach it.
    result = _solve(context_anchor=64, context_overlap=16, max_tile_width=1024)

    assert result.layout.upscale_by == 2.0
    assert result.layout.max_tile_width == 1024
    assert result.layout.max_tile_height == 2048
    assert result.layout.context_anchor == 64
    assert result.layout.context_overlap == 16


def test_a_grid_config_error_propagates_naming_the_widget():
    # grid.GridConfigError already names the widget at fault, so the node must not swallow it.
    with pytest.raises(grid.GridConfigError, match="max_tile_width 256"):
        _solve(max_tile_width=256, context_anchor=512, context_overlap=512)


def test_the_overlay_is_an_image_tensor_under_the_preview_budget():
    result = _solve()
    overlay = result.overlay

    assert overlay.ndim == 4
    assert overlay.shape[0] == 1 and overlay.shape[3] == 3
    assert overlay.dtype == torch.float32
    height, width = int(overlay.shape[1]), int(overlay.shape[2])
    assert width * height <= testing.OVERLAY_MEGAPIXELS * 1_000_000
    # The TARGET's aspect, not the source's: the grid is solved on the upscaled size.
    assert width / height == pytest.approx(2000 / 1400, abs=0.01)
    assert float(overlay.min()) >= 0.0 and float(overlay.max()) <= 1.0


def test_a_small_target_keeps_the_target_size_exactly():
    # Under the budget nothing is resampled down, so the preview IS the target size.
    result = _solve(image=torch.rand(1, 100, 200, 3), upscale_by=2.0)

    assert result.layout.target_size == (400, 200)
    assert tuple(result.overlay.shape) == (1, 200, 400, 3)


def test_the_first_cores_top_left_pixel_carries_the_core_colour():
    # Tile 0's core starts at the canvas origin, and the core band is drawn LAST, so the
    # corner pixel pins both the scaling and the band order (a later band would hide a core).
    result = _solve()

    expected = torch.tensor(testing.CORE_COLOR, dtype=torch.float32) / 255.0
    assert torch.allclose(result.overlay[0, 0, 0], expected)


def test_the_second_tiles_core_starts_at_its_scaled_column():
    # The rects are scaled per axis from the layout's canvas onto the preview. A core drawn at
    # the unscaled column would sit hundreds of pixels away.
    result = _solve()

    preview_width = int(result.overlay.shape[2])
    column = round(result.layout.layout.tiles[1].core.x0 * preview_width / 2000)
    expected = torch.tensor(testing.CORE_COLOR, dtype=torch.float32) / 255.0
    assert torch.allclose(result.overlay[0, 0, column], expected)


def test_the_layout_node_rejects_a_batch_above_one():
    with pytest.raises(ValueError, match="got a batch of 2"):
        _solve(image=torch.rand(2, 700, 1000, 3))


def test_validate_inputs_returns_the_base_nodes_own_message():
    # One rule set for the production nodes and the testing chain, so a workflow that is legal
    # for one is legal for the other.
    geometry = {"max_tile_width": 1004, "max_tile_height": 2048, "context_anchor": 32, "context_overlap": 32}
    message = ContextAnchoredTileTestLayout.VALIDATE_INPUTS(**geometry)

    assert message == ContextAnchoredTileRefine.VALIDATE_INPUTS(**geometry)
    assert message == "max_tile_width must be a multiple of 8, got 1004"
    geometry["max_tile_width"] = 1024
    assert ContextAnchoredTileTestLayout.VALIDATE_INPUTS(**geometry) is True


def _record_prepare_upscaled(monkeypatch, result):
    recorded = {}

    def fake_prepare_upscaled(image, upscale_model, upscale_by, progress=None):
        recorded["call"] = (image, upscale_model, upscale_by, progress)
        return result

    monkeypatch.setattr(upscale, "prepare_upscaled", fake_prepare_upscaled)
    return recorded


def test_the_upscale_node_forwards_the_layouts_multiplier_and_the_model(monkeypatch):
    # The multiplier comes from the LAYOUT and never from a second widget, which is what keeps
    # the canvas at the size the grid was solved for.
    layout = _solve(upscale_by=1.5).layout
    image = torch.rand(1, 700, 1000, 3)
    upscaled = torch.rand(1, 1050, 1500, 3)
    upscale_model = object()
    recorded = _record_prepare_upscaled(monkeypatch, upscaled)

    result = ContextAnchoredTileTestUpscale().upscale_image(image=image, layout=layout,
                                                            upscale_model=upscale_model)

    assert recorded["call"] == (image, upscale_model, 1.5, None)
    assert result == (upscaled,)


def test_the_upscale_node_rejects_an_image_the_layout_was_not_solved_for(monkeypatch):
    layout = _solve().layout
    recorded = _record_prepare_upscaled(monkeypatch, torch.rand(1, 1400, 2000, 3))

    with pytest.raises(ValueError, match=r"800x600 image, but the layout was solved for 1000x700"):
        ContextAnchoredTileTestUpscale().upscale_image(image=torch.rand(1, 600, 800, 3), layout=layout)
    assert "call" not in recorded


def test_the_upscale_node_rejects_a_result_that_is_not_the_layouts_target(monkeypatch):
    layout = _solve().layout
    _record_prepare_upscaled(monkeypatch, torch.rand(1, 1408, 2008, 3))

    with pytest.raises(RuntimeError, match=r"produced 2008x1408, but the layout was solved for 2000x1400"):
        ContextAnchoredTileTestUpscale().upscale_image(image=torch.rand(1, 700, 1000, 3), layout=layout)


def test_the_upscale_node_rejects_a_batch_above_one(monkeypatch):
    layout = _solve().layout
    _record_prepare_upscaled(monkeypatch, torch.rand(1, 1400, 2000, 3))

    with pytest.raises(ValueError, match="got a batch of 2"):
        ContextAnchoredTileTestUpscale().upscale_image(image=torch.rand(2, 700, 1000, 3), layout=layout)


# --- the Captions node -----------------------------------------------------------------

# The artwork preset is the shipped file's second block, the only one that asks for a style
# caption, which is what makes the style half of every setting visible.
ARTWORK_TILE = ("succinct prose containing relative and absolute positions of specific things "
                "with object and character identifying demographics.")
ARTWORK_STYLE = ("succinct flowing prose of only the overall style and artistic medium and "
                 "physical medium. No objects or items in the scene.")

CAPTION_WIDGETS = {
    "preset": "standard",
    "tile_instruction": "",
    "style_caption": True,
    "style_instruction": "",
    "max_tokens": 0,
    "caption_megapixels": captions.SHIPPED_CAPTION_MEGAPIXELS,
    "tiles": "",
    "with_neighbors": True,
    "prompt": "",
}

# The custom option with every widget it reads filled in, so a test changes one at a time.
CUSTOM_WIDGETS = {
    "preset": testing.CUSTOM_PRESET,
    "tile_instruction": "name the objects",
    "style_instruction": "name the medium",
    "max_tokens": 64,
    "prompt": "",
}


def _caption_layout():
    # A two-tile grid whose 300 px height the node pads to 304, the way the engine does.
    return _solve(image=torch.rand(1, 300, 600, 3), upscale_by=1.0,
                  max_tile_width=512, max_tile_height=512).layout


def _grid_layout():
    # A 4x4 grid on a 600x600 canvas (the Render node's own fixture), where tile 5 has eight
    # bordering tiles and tile 0 has three.
    return _solve(image=torch.rand(1, 600, 600, 3), upscale_by=1.0, max_tile_width=288,
                  max_tile_height=288).layout


def _caption(clip, layout=None, image=None, **overrides):
    """Run the Captions node. Returns its three outputs named."""
    widgets = dict(CAPTION_WIDGETS)
    widgets.update(overrides)
    test_layout = _caption_layout() if layout is None else layout
    if image is None:
        width, height = test_layout.target_size
        image = torch.rand(1, height, width, 3)
    written, text, tiles = ContextAnchoredTileTestCaptions().caption_tiles(
        image=image, layout=test_layout, clip=clip, **widgets)
    return SimpleNamespace(written=written, text=text, tiles=tiles)


def _asking_clip():
    return FakeCaptionClip(answer=lambda image, instruction: f"asked {instruction}")


def _counting_clip():
    clip = FakeCaptionClip(answer=lambda image, instruction: f"caption {len(clip.generate_calls)}")
    return clip


def test_the_preset_widget_offers_the_settings_files_own_presets_then_the_custom_option():
    # preset_labels is read once per session for the reason vlm_methods is, so the widget the
    # frontend cached at startup cannot offer a preset the run then fails to resolve.
    preset = ContextAnchoredTileTestCaptions.INPUT_TYPES()["required"]["preset"]

    assert captions.preset_labels() == ("prompted", "standard", "artwork")
    assert preset[0] == ["prompted", "standard", "artwork", "custom instructions"]
    assert preset[1]["default"] == "prompted"


def test_a_settings_preset_named_like_the_custom_option_is_refused(monkeypatch):
    # The two would be one combo entry routed two ways, so the file is told to rename it. The
    # label list is read through the cached preset_labels, which the autouse fixture clears.
    shipped = captions.load_settings()
    presets = {**shipped.presets, testing.CUSTOM_PRESET: shipped.presets["standard"]}
    monkeypatch.setattr(captions, "load_settings",
                        lambda path=None: captions.Settings(vision=shipped.vision, presets=presets))

    with pytest.raises(ValueError, match="defines a preset named 'custom instructions'"):
        ContextAnchoredTileTestCaptions.INPUT_TYPES()


def test_the_caption_size_widget_defaults_to_the_settings_files_own_value():
    widget = ContextAnchoredTileTestCaptions.INPUT_TYPES()["required"]["caption_megapixels"]

    assert widget[1]["default"] == captions.SHIPPED_CAPTION_MEGAPIXELS
    assert (widget[1]["min"], widget[1]["max"]) == (0.0, 2.0)


def test_every_caption_input_has_a_tooltip():
    for name, definition in ContextAnchoredTileTestCaptions.INPUT_TYPES()["required"].items():
        assert "tooltip" in definition[1], name


def test_the_captions_node_asks_for_its_node_id():
    # The ledger writes its status line under the node, and core hands the id in as a hidden
    # input only.
    assert ContextAnchoredTileTestCaptions.INPUT_TYPES()["hidden"] == {"unique_id": "UNIQUE_ID"}


def test_the_captions_node_returns_the_captions_the_listing_and_the_tiles():
    assert ContextAnchoredTileTestCaptions.RETURN_TYPES == ("CATR_CAPTIONS", "STRING", "STRING")
    assert ContextAnchoredTileTestCaptions.RETURN_NAMES == ("captions", "text", "tiles")


def test_the_prompt_widget_is_the_vl_nodes_own(comfy_stubs):
    # One definition (node._prompt) on every node that fills {PROMPT}, so the wording and the
    # default cannot drift between the production nodes and the test chain.
    from context_anchored_tile_refine import node

    assert ContextAnchoredTileTestCaptions.INPUT_TYPES()["required"]["prompt"] == node._prompt()


def test_the_prompt_is_written_into_the_default_preset(comfy_stubs):
    # The shipped default asks for {PROMPT}, and the node fills it from its own widget, the
    # same captions.with_prompt the production nodes run.
    clip = _asking_clip()
    shipped = captions.load_settings().presets["prompted"]["tile_caption_instruction"]

    result = _caption(clip, preset="prompted", prompt="a fox in the centre\n")

    expected = shipped.replace("{PROMPT}", "a fox in the centre")
    assert "a fox in the centre" in expected
    assert [call["text"] for call in clip.generate_calls] == [expected] * 2
    assert result.written.preset == "prompted"


def test_the_prompt_is_written_into_a_custom_instruction(comfy_stubs):
    clip = _asking_clip()

    _caption(clip, **{**CUSTOM_WIDGETS, "tile_instruction": 'Full prompt: "{PROMPT}". Name it.',
                      "style_instruction": "Style of {PROMPT}.", "prompt": "a fox"})

    assert [call["text"] for call in clip.generate_calls] == [
        "Style of a fox.", 'Full prompt: "a fox". Name it.', 'Full prompt: "a fox". Name it.']


def test_a_blank_prompt_against_a_preset_that_asks_for_one_is_refused_before_any_caption(comfy_stubs):
    clip = _asking_clip()

    with pytest.raises(RuntimeError, match=r"preset 'prompted' asks for \{PROMPT\}.*prompt input is empty"):
        _caption(clip, preset="prompted", prompt="")

    assert clip.generate_calls == []


def test_every_caption_is_asked_with_the_reasoning_turn_on(comfy_stubs):
    # The test chain runs the engine's own caption pass, so every question carries
    # thinking=True and strip_thinking then cuts the reasoning off the answer.
    clip = _asking_clip()

    _caption(clip, preset="artwork")

    assert len(clip.generate_calls) == 3
    assert all(call["thinking"] is True for call in clip.tokenize_calls)


def test_the_preset_is_read_from_the_file_when_named(comfy_stubs):
    # The default preset is the FIRST block and the node asks for it by label, so the wording
    # and the budget both come from the file rather than from a constant here.
    clip = _asking_clip()

    result = _caption(clip, preset="standard")

    assert [call["text"] for call in clip.generate_calls] == [captions.RICH_GROUPED_INSTRUCTION] * 2
    assert [call["max_length"] for call in clip.generate_calls] == [768, 768]
    assert result.written.preset == "standard"


def test_a_named_preset_ignores_the_instruction_and_budget_widgets(comfy_stubs):
    # A trial wording stays in the widgets while a preset runs, so switching between the two
    # never means clearing the fields.
    clip = _asking_clip()

    result = _caption(clip, preset="artwork", tile_instruction="name the objects",
                      style_instruction="name the medium", max_tokens=64)

    assert [call["text"] for call in clip.generate_calls] == [
        ARTWORK_STYLE, ARTWORK_TILE, ARTWORK_TILE]
    assert [call["max_length"] for call in clip.generate_calls] == [768, 768, 768]
    assert result.written.preset == "artwork"


def test_the_custom_option_writes_from_the_three_widgets(comfy_stubs):
    clip = _asking_clip()

    result = _caption(clip, **CUSTOM_WIDGETS)

    assert [call["text"] for call in clip.generate_calls] == [
        "name the medium", "name the objects", "name the objects"]
    assert [call["max_length"] for call in clip.generate_calls] == [64, 64, 64]
    # The style caption is kept apart from the tile captions. The Render node joins them.
    assert result.written.style == ("asked name the medium",)
    assert result.written.captions == (("asked name the objects",),) * 2
    assert result.written.preset == "custom instructions"


def test_the_custom_option_still_reads_the_caption_size_widget(comfy_stubs, monkeypatch):
    seen = []
    monkeypatch.setattr(captions, "resample_for_vl",
                        lambda pixels, budget=None: seen.append(budget) or pixels)

    _caption(_asking_clip(), caption_megapixels=0.25, **CUSTOM_WIDGETS)

    assert seen == [250_000] * 3


def test_an_empty_custom_style_instruction_writes_no_style_caption(comfy_stubs):
    clip = _asking_clip()

    result = _caption(clip, **dict(CUSTOM_WIDGETS, style_instruction="  "))

    assert [call["text"] for call in clip.generate_calls] == ["name the objects"] * 2
    assert result.written.style is None
    assert result.written.captions == (("asked name the objects",),) * 2


def test_the_custom_option_rejects_an_empty_tile_instruction(comfy_stubs):
    with pytest.raises(ValueError, match="with an empty tile_instruction"):
        _caption(_asking_clip(), **dict(CUSTOM_WIDGETS, tile_instruction=" "))


def test_the_custom_option_rejects_a_zero_budget(comfy_stubs):
    with pytest.raises(ValueError, match="with max_tokens 0"):
        _caption(_asking_clip(), **dict(CUSTOM_WIDGETS, max_tokens=0))


def test_the_caption_size_widget_sets_the_picture_the_vl_model_reads(comfy_stubs):
    # resample_for_vl asks comfy.utils.common_upscale for the budget with the aspect kept, so
    # the recorded sizes are what the VL model was handed.
    _caption(_asking_clip(), preset="artwork", caption_megapixels=0.25)

    assert len(comfy_stubs["common_upscale_calls"]) == 3
    for _shape, width, height, method, _crop in comfy_stubs["common_upscale_calls"]:
        assert method == "area"
        assert width * height == pytest.approx(250_000, rel=0.02)


@pytest.mark.parametrize("widgets", [{"preset": "artwork"}, CUSTOM_WIDGETS])
def test_style_caption_off_removes_the_style_caption(comfy_stubs, widgets):
    # artwork asks for a style caption and the custom widgets carry one, so switching the
    # boolean off has to be what removes it on both.
    clip = _asking_clip()

    result = _caption(clip, style_caption=False, **widgets)

    assert len(clip.generate_calls) == 2
    assert result.written.style is None
    assert all("\n" not in rows[0] for rows in result.written.captions)


def test_every_tile_is_captioned_from_the_padded_canvas(comfy_stubs, monkeypatch):
    # The engine captions the canvas padded to /8, so a crop read off the unpadded image would
    # describe 300 rows where the run reads 304.
    monkeypatch.setattr(captions, "resample_for_vl", lambda pixels, budget=None: pixels)
    clip = FakeCaptionClip()

    _caption(clip, layout=_caption_layout())

    assert [tuple(call["image"].shape) for call in clip.tokenize_calls] == [
        (1, 304, 368, 3), (1, 304, 360, 3)]


def test_the_captions_object_carries_the_grid_it_was_written_for(comfy_stubs):
    # A caption belongs to a tile rect, so the Render node can reject captions written for
    # another grid at the same size.
    layout = _caption_layout()

    result = _caption(FakeCaptionClip(), layout=layout)

    assert result.written.target_size == layout.target_size == (600, 300)
    assert result.written.grid == (2, 1)
    assert result.written.rects == tuple(
        (tile.crop_rect.x0, tile.crop_rect.y0, tile.crop_rect.x1, tile.crop_rect.y1)
        for tile in layout.layout.tiles)
    assert result.written.captions == (("a plain caption",), ("a plain caption",))
    assert result.written.style is None
    assert result.written.tiles == ()


def test_the_text_output_lists_every_tile_by_its_overlay_number(comfy_stubs):
    result = _caption(_counting_clip())

    assert result.text == ("preset standard, 2 tiles captioned\n\n"
                           "tile 0 r0c0\ncaption 1\n\n"
                           "tile 1 r0c1\ncaption 2")
    # Core's Preview as Text node falls back to str() for a value it cannot serialize, so the
    # captions socket reads the same as the text output.
    assert str(result.written) == result.text
    assert result.tiles == ""


@pytest.mark.parametrize("widgets", [{"preset": "artwork"}, CUSTOM_WIDGETS])
def test_the_text_output_lists_the_style_caption_once_at_the_top(comfy_stubs, widgets):
    # The style caption is generated once and read once, so it is listed once, labelled, above
    # the tiles, and the listing reads the same whether a file preset or the custom option
    # wrote it.
    clip = _counting_clip()

    result = _caption(clip, **widgets)

    assert result.text == (f"preset {widgets['preset']}, 2 tiles captioned\n\n"
                           "style caption\ncaption 1\n\n"
                           "tile 0 r0c0\ncaption 2\n\n"
                           "tile 1 r0c1\ncaption 3")
    assert str(result.written) == result.text


def test_named_tiles_alone_are_captioned_with_neighbors_off(comfy_stubs):
    # The reason the widget exists: one caption instead of sixteen while a wording is tuned.
    clip = _counting_clip()

    result = _caption(clip, layout=_grid_layout(), tiles="5", with_neighbors=False)

    assert len(clip.generate_calls) == 1
    assert result.written.captions == tuple(("caption 1",) if index == 5 else None for index in range(16))
    assert result.written.tiles == (5,)
    assert result.tiles == "5"
    assert result.text == "preset standard, 1 of 16 tiles captioned\n\ntile 5 r1c1\ncaption 1"


def test_named_tiles_bring_their_bordering_tiles_with_neighbors_on(comfy_stubs):
    # Tile Test: Render runs a named tile with its bordering tiles as lanes, and every lane
    # needs a caption, so the block's tiles are written too. The named tiles come first, in
    # the order written, and the bordering tiles after in layout order, named ones excluded.
    clip = _counting_clip()

    result = _caption(clip, layout=_grid_layout(), tiles="5, 0")

    captioned = [index for index, rows in enumerate(result.written.captions) if rows is not None]
    assert captioned == [0, 1, 2, 4, 5, 6, 8, 9, 10]
    assert result.written.captions[5] == ("caption 1",)
    assert result.written.captions[0] == ("caption 2",)
    assert result.written.captions[1] == ("caption 3",)
    assert result.written.tiles == (5, 0)
    assert result.tiles == "5, 0"
    assert result.text.startswith("preset standard, 9 of 16 tiles captioned\n\n"
                                  "tile 5 r1c1\ncaption 1\n\ntile 0 r0c0\ncaption 2\n\n"
                                  "bordering tiles\n\ntile 1 r0c1\ncaption 3\n\n")


def test_the_style_caption_is_written_once_for_a_tile_list(comfy_stubs):
    clip = _asking_clip()

    result = _caption(clip, layout=_grid_layout(), preset="artwork", tiles="5", with_neighbors=False)

    assert [call["text"] for call in clip.generate_calls] == [ARTWORK_STYLE, ARTWORK_TILE]
    assert result.written.style == (f"asked {ARTWORK_STYLE}",)
    assert result.written.captions[5] == (f"asked {ARTWORK_TILE}",)


@pytest.mark.parametrize("text", ["two", "5,x"])
def test_the_captions_node_rejects_a_csv_entry_that_is_not_a_number(comfy_stubs, text):
    with pytest.raises(ValueError, match=r"Tile Test: Captions was given the tile .* which is not a tile number"):
        _caption(FakeCaptionClip(), layout=_grid_layout(), tiles=text)


def test_the_captions_node_rejects_a_tile_number_outside_the_grid(comfy_stubs):
    with pytest.raises(ValueError, match=r"Tile Test: Captions was given tile 16, and this layout has 16 tiles"):
        _caption(FakeCaptionClip(), layout=_grid_layout(), tiles="16")


def test_the_run_is_one_progress_bar_that_ends_full(comfy_stubs):
    # Core builds a per-token bar inside every clip.generate and the caption pass built one of
    # its own, so the display reset at every caption and stopped where the stop token fired.
    # The ledger's shim routes the inner bars into one bar over the whole run.
    clip = _asking_clip()

    _caption(clip, layout=_grid_layout(), preset="artwork", tiles="5")

    assert len(comfy_stubs["progress_bars"]) == 1
    bar = comfy_stubs["progress_bars"][0]
    total = round(10 * progress.K_CAPTION * progress.EMIT_SCALE)
    assert bar.total == total
    values = [value for value, _total, _preview in bar.updates]
    assert values == sorted(values)
    assert bar.updates[-1][:2] == (total, total)
    assert clip.generate_calls[0]["text"] == ARTWORK_STYLE


def test_the_progress_shim_is_restored_after_the_run(comfy_stubs):
    import comfy.utils

    before = comfy.utils.ProgressBar
    _caption(_asking_clip())

    assert comfy.utils.ProgressBar is before


def test_the_captions_node_rejects_a_batch_above_one():
    with pytest.raises(ValueError, match="got a batch of 2"):
        _caption(FakeCaptionClip(), image=torch.rand(2, 300, 600, 3))


def test_the_captions_node_rejects_an_image_the_layout_was_not_solved_for():
    with pytest.raises(ValueError, match=r"800x600 image, but the layout was solved for 600x300"):
        _caption(FakeCaptionClip(), image=torch.rand(1, 600, 800, 3))


@pytest.mark.parametrize("value", [-0.5, 0.005, 2.5])
def test_validate_inputs_rejects_a_caption_size_the_settings_file_would_reject(value):
    # Naming the widget disables core's own range check for it, so the file's entire rule is
    # re-checked here. The widget's own range cannot express "0 or at least 0.01".
    assert ContextAnchoredTileTestCaptions.VALIDATE_INPUTS(caption_megapixels=value) == (
        "caption_megapixels must be 0, which reads the picture's own size, or between 0.01 "
        f"and 2.0. Got {value}.")


@pytest.mark.parametrize("value", [0.0, 0.5])
def test_validate_inputs_accepts_zero_and_a_size_inside_the_range(value):
    assert ContextAnchoredTileTestCaptions.VALIDATE_INPUTS(caption_megapixels=value) is True


def test_the_captions_node_reports_the_settings_file_in_its_cache_key():
    # ComfyUI folds IS_CHANGED into the cache key, and core calls it as f(**inputs), so every
    # widget on the node arrives as a keyword.
    expected = captions.settings_fingerprint()

    assert ContextAnchoredTileTestCaptions.IS_CHANGED() == expected
    assert ContextAnchoredTileTestCaptions.IS_CHANGED(image=None, preset="artwork") == expected


# --- the Render node -------------------------------------------------------------------

# A 4x4 grid on a 600x600 canvas, base 152: big enough that a neighborhood clamps at a corner
# and reaches three columns in the middle, small enough to state every rect by hand.
RENDER_WIDGETS = {
    "seed": 1234,
    "sampler_name": "dpmpp_2m",
    "scheduler": "sgm_uniform",
    "steps": 20,
    "cfg": 3.5,
    "denoise": 0.5,
    "anchor_source": "source image",
    "surface": captions.VLM_METHOD_VISION,
    "canvas_tokens": 40,
    "crop_tokens": 12,
    "tiles": "",
    "with_neighbors": True,
}


class FakeGuider:
    def __init__(self, model, positive, negative, cfg):
        self.model = model
        self.positive = positive
        self.negative = negative
        self.cfg = cfg


class FakeVAE:
    latent_channels = 16
    latent_dim = 2


def _render_layout(**overrides):
    widgets = {"upscale_by": 1.0, "max_tile_width": 288, "max_tile_height": 288,
               "context_anchor": 32, "context_overlap": 32}
    widgets.update(overrides)
    picture = widgets.pop("image", None)
    return _solve(image=torch.rand(1, 600, 600, 3) if picture is None else picture, **widgets).layout


def _positional(height, width):
    # Every pixel carries its own canvas coordinate, so a crop taken from the wrong rect is
    # visible in the value rather than only in the shape.
    rows = torch.arange(height, dtype=torch.float32)[:, None] * 10000.0
    cols = torch.arange(width, dtype=torch.float32)[None, :]
    return (rows + cols)[None, :, :, None].expand(1, height, width, 3).contiguous()


def _captions_for(layout, only=None, style=None):
    # `only` is the tile numbers a partial caption set covers; None covers every tile.
    tiles = layout.layout.tiles
    return testing.TestCaptions(
        captions=tuple((f"caption {index}",) if only is None or index in only else None
                       for index in range(len(tiles))),
        style=style,
        tiles=() if only is None else tuple(only),
        target_size=layout.target_size,
        grid=(layout.layout.sol_x.n, layout.layout.sol_y.n),
        rects=tuple((tile.crop_rect.x0, tile.crop_rect.y0, tile.crop_rect.x1, tile.crop_rect.y1)
                    for tile in tiles),
        preset="standard",
    )


def _render(monkeypatch, layout=None, image=None, given_captions=None, negative=None,
            sigmas=None, recorded=None, **overrides):
    """Run the Render node with every collaborator faked; return (recorded, result).

    `recorded` may be passed in, so a test that expects a raise can still read how far the node
    got before it."""
    recorded = {} if recorded is None else recorded
    recorded.update({
        "calls": [],
        "encode_empty": None,
        "build_guider": None,
        "build_sigmas": None,
        "sigmas": torch.linspace(1.0, 0.0, 5) if sigmas is None else sigmas,
        "empty_cond": [("empty", {})],
    })

    def fake_encode_empty(clip):
        recorded["encode_empty"] = clip
        return recorded["empty_cond"]

    def fake_build_guider(model, positive, negative, cfg):
        recorded["build_guider"] = (model, positive, negative, cfg)
        return FakeGuider(model, positive, negative, cfg)

    def fake_build_sigmas(model, scheduler, steps, denoise):
        recorded["build_sigmas"] = (model, scheduler, steps, denoise)
        return recorded["sigmas"]

    def fake_refine_image(image, guider, sampler, sigmas, vae, noise, max_tile_width,
                          max_tile_height, context_anchor, context_overlap, mask=None,
                          vl_clip=None, vlm_method=None, anchor_source=None, sampler_name=None,
                          progress=None, preset=None, tile_captions=None, layout=None,
                          noise_fields=None):
        recorded["calls"].append({
            "image": image, "guider": guider, "sampler": sampler, "sigmas": sigmas, "vae": vae,
            "noise": noise, "max_tile_width": max_tile_width, "max_tile_height": max_tile_height,
            "context_anchor": context_anchor, "context_overlap": context_overlap, "mask": mask,
            "vl_clip": vl_clip, "vlm_method": vlm_method, "anchor_source": anchor_source,
            "sampler_name": sampler_name, "progress": progress, "preset": preset,
            "tile_captions": tile_captions, "layout": layout, "noise_fields": noise_fields,
        })
        return _positional(int(image.shape[1]), int(image.shape[2]))

    monkeypatch.setattr(upscale, "encode_empty", fake_encode_empty)
    monkeypatch.setattr(upscale, "build_guider", fake_build_guider)
    monkeypatch.setattr(upscale, "build_sigmas", fake_build_sigmas)
    monkeypatch.setattr(sampling, "refine_image", fake_refine_image)
    # The noise object stays REAL, since its rect is what the block run is pinned on. Only the
    # field provider is a sentinel, so the value refine_image receives is traceable to it.
    monkeypatch.setattr(upscale.SlicedCanvasNoise, "noise_fields",
                        lambda self, sampler, sigmas: ("fields", self.cell_origin))

    recorded["layout"] = _render_layout() if layout is None else layout
    target = recorded["layout"].target_size
    recorded["image"] = torch.rand(1, target[1], target[0], 3) if image is None else image
    recorded["model"] = object()
    recorded["clip"] = object()
    recorded["vae"] = FakeVAE()

    widgets = dict(RENDER_WIDGETS)
    widgets.update(overrides)
    result = ContextAnchoredTileTestRender().render_tiles(
        image=recorded["image"],
        layout=recorded["layout"],
        model=recorded["model"],
        clip=recorded["clip"],
        vae=recorded["vae"],
        captions=given_captions,
        negative=negative,
        **widgets,
    )
    return recorded, result


def test_the_full_run_hands_the_engine_the_layout_and_no_mask(comfy_stubs, monkeypatch):
    # An empty csv is one call over the entire canvas, and the layout override is what makes
    # the engine sample the grid the overlay drew instead of solving its own.
    recorded, result = _render(monkeypatch)
    call = recorded["calls"][0]

    assert len(recorded["calls"]) == 1
    assert call["image"] is recorded["image"]
    assert call["layout"] is recorded["layout"].layout
    assert call["mask"] is None
    assert call["noise_fields"] is None
    assert isinstance(call["noise"], upscale.Noise_RandomNoise) and call["noise"].seed == 1234
    assert result[0][0] is result[1][0]


def test_the_geometry_and_the_widgets_reach_the_engine(comfy_stubs, monkeypatch):
    recorded, _ = _render(monkeypatch, anchor_source="live canvas")
    call = recorded["calls"][0]

    assert (call["max_tile_width"], call["max_tile_height"]) == (288, 288)
    assert (call["context_anchor"], call["context_overlap"]) == (32, 32)
    assert call["vl_clip"] is recorded["clip"]
    assert call["vae"] is recorded["vae"]
    assert call["sigmas"] is recorded["sigmas"]
    assert call["anchor_source"] == "live canvas"
    assert call["sampler_name"] == "dpmpp_2m"
    # The ledger belongs to the production nodes, so the engine keeps its pre-ledger bars here.
    assert call["progress"] is None


@pytest.mark.parametrize("surface", list(captions.VLM_SURFACES))
def test_the_surface_widget_is_the_vlm_method_the_engine_branches_on(comfy_stubs, monkeypatch, surface):
    layout = _render_layout()
    given = None if surface == captions.VLM_METHOD_VISION else _captions_for(layout)
    recorded, _ = _render(monkeypatch, layout=layout, surface=surface, given_captions=given)

    assert recorded["calls"][0]["vlm_method"] == surface
    assert recorded["calls"][0]["preset"].surface == surface


def test_the_preset_carries_the_two_token_widgets_over_the_settings_file(comfy_stubs, monkeypatch):
    # The widgets are the reason this node exists: a token count is tried here without editing
    # settings.toml, and everything else in the block still comes from the file.
    recorded, _ = _render(monkeypatch, canvas_tokens=7, crop_tokens=0)
    preset = recorded["calls"][0]["preset"]
    shipped = captions.load_settings().vision

    assert (preset.vision.canvas_tokens, preset.vision.crop_tokens) == (7, 0)
    assert preset.vision.caption_megapixels == shipped.caption_megapixels


def test_the_connected_captions_reach_the_engine_on_a_caption_surface(comfy_stubs, monkeypatch):
    layout = _render_layout()
    given = _captions_for(layout)

    recorded, _ = _render(monkeypatch, layout=layout, surface=captions.VLM_METHOD_CAPTIONS,
                          given_captions=given)

    assert recorded["calls"][0]["tile_captions"] == given.captions


def test_the_style_caption_is_joined_onto_every_lane_for_the_engine(comfy_stubs, monkeypatch):
    # The captions object keeps the style apart, and the engine reads one caption per lane with
    # the style on top, the form captions.generate_tile_captions writes on a production run.
    layout = _render_layout()
    given = _captions_for(layout, style=("the medium",))

    recorded, _ = _render(monkeypatch, layout=layout, surface=captions.VLM_METHOD_CAPTIONS,
                          given_captions=given)

    assert recorded["calls"][0]["tile_captions"] == tuple(
        (f"the medium\ncaption {index}",) for index in range(16))


def test_the_style_caption_is_joined_onto_every_block_lane(comfy_stubs, monkeypatch):
    layout = _render_layout()
    given = _captions_for(layout, style=("the medium",))

    recorded, _ = _render(monkeypatch, layout=layout, tiles="10",
                          surface=captions.VLM_METHOD_CAPTIONS, given_captions=given)
    call = recorded["calls"][0]
    sub = call["layout"]

    assert call["tile_captions"] == tuple(
        (f"the medium\ncaption {(1 + tile.row) * 4 + 1 + tile.col}",) for tile in sub.layout.tiles)


def test_the_vision_surface_ignores_a_connected_captions_object(comfy_stubs, monkeypatch):
    # The vision surface builds no captions at all, and the engine rejects a caption set handed
    # to it, so a connected socket must not reach it.
    layout = _render_layout()

    recorded, _ = _render(monkeypatch, layout=layout, surface=captions.VLM_METHOD_VISION,
                          given_captions=_captions_for(layout))

    assert recorded["calls"][0]["tile_captions"] is None


def test_an_unconnected_negative_is_the_empty_encode(comfy_stubs, monkeypatch):
    recorded, _ = _render(monkeypatch)
    model, positive, negative, cfg = recorded["build_guider"]

    assert recorded["encode_empty"] is recorded["clip"]
    assert positive is recorded["empty_cond"] and negative is recorded["empty_cond"]
    assert (model, cfg) == (recorded["model"], 3.5)


def test_a_connected_negative_replaces_it(comfy_stubs, monkeypatch):
    negative = [("negative", {})]
    recorded, _ = _render(monkeypatch, negative=negative)

    assert recorded["build_guider"][2] is negative


def test_a_csv_makes_one_block_run_per_tile_over_its_neighborhood(comfy_stubs, monkeypatch):
    # The 3x3 block is the faithful stand-in for the full run: a lone lane has nobody to
    # consolidate its overlap bands against.
    layout = _render_layout()
    recorded, _ = _render(monkeypatch, layout=layout, tiles="5, 10")

    assert len(recorded["calls"]) == 2
    for index, call in zip((5, 10), recorded["calls"], strict=True):
        expected = grid.sub_layout(layout.layout, *grid.neighborhood(layout.layout, index))
        assert isinstance(call["layout"], grid.SubLayout)
        assert call["layout"].block == expected.block
        assert (call["layout"].first_col, call["layout"].first_row) == (expected.first_col, expected.first_row)
        assert call["layout"].parent_tiles is layout.layout.tiles


def test_with_neighbors_off_renders_the_tile_alone(comfy_stubs, monkeypatch):
    layout = _render_layout()
    recorded, _ = _render(monkeypatch, layout=layout, tiles="5", with_neighbors=False)
    sub = recorded["calls"][0]["layout"]

    assert (sub.first_col, sub.first_row) == (1, 1)
    assert len(sub.layout.tiles) == 1
    # The block IS the tile's crop rect, so its anchor bands stay the unrefined source.
    assert sub.block == grid.Rect(88, 88, 368, 368)


def test_the_mask_is_ones_on_the_blocks_region_and_zeros_elsewhere(comfy_stubs, monkeypatch):
    layout = _render_layout()
    recorded, _ = _render(monkeypatch, layout=layout, tiles="5")
    call = recorded["calls"][0]
    region = call["layout"].region

    expected = torch.zeros(1, 600, 600)
    expected[:, region.y0:region.y1, region.x0:region.x1] = 1.0
    assert torch.equal(call["mask"], expected)
    assert call["mask"].device == recorded["image"].device
    assert call["mask"].dtype == torch.float32


def test_the_block_noise_is_the_canvas_draw_sliced_at_the_block(comfy_stubs, monkeypatch):
    # A block that drew at its own shape would start every lane from cells the full run never
    # gave it, which makes it a different render rather than a reproduction.
    layout = _render_layout()
    recorded, _ = _render(monkeypatch, layout=layout, tiles="10")
    call = recorded["calls"][0]
    block = call["layout"].block

    assert isinstance(call["noise"], upscale.SlicedCanvasNoise)
    assert call["noise"].seed == 1234
    assert call["noise"].canvas_shape == (1, 16, 75, 75)
    assert call["noise"].cell_origin == (block.y0 // 8, block.x0 // 8)
    assert tuple(call["noise"].slice.shape) == (1, 16, (block.y1 - block.y0) // 8, (block.x1 - block.x0) // 8)
    assert call["noise_fields"] == ("fields", call["noise"].cell_origin)


def test_every_block_lane_carries_its_parent_tiles_caption(comfy_stubs, monkeypatch):
    # A lane conditioned on another tile's description would describe pixels it never samples.
    layout = _render_layout()
    given = _captions_for(layout)
    recorded, _ = _render(monkeypatch, layout=layout, tiles="10",
                          surface=captions.VLM_METHOD_VISION_CAPTIONS, given_captions=given)
    call = recorded["calls"][0]
    sub = call["layout"]

    assert (sub.first_col, sub.first_row) == (1, 1)
    assert call["tile_captions"] == tuple(
        given.captions[(1 + tile.row) * 4 + 1 + tile.col] for tile in sub.layout.tiles)
    assert call["tile_captions"][0] == ("caption 5",)


def test_a_caption_set_covering_the_block_is_enough_for_a_block_run(comfy_stubs, monkeypatch):
    # What Tile Test: Captions writes for tiles="5" with its with_neighbors on: the tile and
    # its eight bordering tiles, and nothing else.
    layout = _render_layout()
    given = _captions_for(layout, only=(5, 0, 1, 2, 4, 6, 8, 9, 10))
    recorded, _ = _render(monkeypatch, layout=layout, tiles="5",
                          surface=captions.VLM_METHOD_CAPTIONS, given_captions=given)

    assert recorded["calls"][0]["tile_captions"][4] == ("caption 5",)
    assert None not in recorded["calls"][0]["tile_captions"]


def test_a_block_lane_without_a_caption_is_rejected_before_any_model_call(comfy_stubs, monkeypatch):
    # Tile Test: Captions with its with_neighbors off writes the named tile only, and a block
    # run here with with_neighbors on has eight more lanes to condition.
    layout = _render_layout()
    recorded = {}

    with pytest.raises(ValueError, match=r"tile 5's block, and tiles 0, 1, 2, 4, 6, 8, 9, 10 have none. "
                                          r"The captions cover tiles 5. Turn with_neighbors on at Tile Test: Captions"):
        _render(monkeypatch, layout=layout, tiles="5", surface=captions.VLM_METHOD_CAPTIONS,
                given_captions=_captions_for(layout, only=(5,)), recorded=recorded)
    assert recorded["build_sigmas"] is None


def test_a_tile_alone_needs_only_its_own_caption(comfy_stubs, monkeypatch):
    layout = _render_layout()
    recorded, _ = _render(monkeypatch, layout=layout, tiles="5", with_neighbors=False,
                          surface=captions.VLM_METHOD_CAPTIONS,
                          given_captions=_captions_for(layout, only=(5,)))

    assert recorded["calls"][0]["tile_captions"] == (("caption 5",),)


def test_the_full_run_rejects_a_caption_set_that_covers_part_of_the_grid(comfy_stubs, monkeypatch):
    layout = _render_layout()
    recorded = {}

    with pytest.raises(ValueError, match=r"entire canvas, and the captions cover tiles 5, 6 only"):
        _render(monkeypatch, layout=layout, surface=captions.VLM_METHOD_CAPTIONS,
                given_captions=_captions_for(layout, only=(5, 6)), recorded=recorded)
    assert recorded["build_sigmas"] is None


def test_the_vision_surface_ignores_a_partial_caption_set(comfy_stubs, monkeypatch):
    layout = _render_layout()
    recorded, _ = _render(monkeypatch, layout=layout, surface=captions.VLM_METHOD_VISION,
                          given_captions=_captions_for(layout, only=(5,)))

    assert recorded["calls"][0]["tile_captions"] is None


def test_the_two_lists_carry_the_tile_crop_and_the_block(comfy_stubs, monkeypatch):
    layout = _render_layout()
    recorded, (rendered_tiles, blocks) = _render(monkeypatch, layout=layout, tiles="5,10")

    for position, index in enumerate((5, 10)):
        crop = layout.layout.tiles[index].crop_rect
        block = recorded["calls"][position]["layout"].block
        assert tuple(rendered_tiles[position].shape) == (1, crop.y1 - crop.y0, crop.x1 - crop.x0, 3)
        assert tuple(blocks[position].shape) == (1, block.y1 - block.y0, block.x1 - block.x0, 3)
        # The engine returns the entire picture, so the cut is what places each output.
        assert float(rendered_tiles[position][0, 0, 0, 0]) == crop.y0 * 10000 + crop.x0
        assert float(blocks[position][0, 0, 0, 0]) == block.y0 * 10000 + block.x0


def test_an_edge_rect_is_cut_back_to_the_picture(comfy_stubs, monkeypatch):
    # The layout is solved on the canvas padded up to a multiple of 8, so the last tile's rects
    # reach past the picture the node was handed.
    layout = _render_layout(image=torch.rand(1, 600, 604, 3))
    recorded, (rendered_tiles, blocks) = _render(monkeypatch, layout=layout, tiles="15")

    assert (layout.target_size, layout.layout.w) == ((604, 600), 608)
    assert recorded["calls"][0]["layout"].block == grid.Rect(304, 304, 608, 600)
    assert tuple(blocks[0].shape) == (1, 296, 300, 3)
    assert tuple(rendered_tiles[0].shape) == (1, 208, 212, 3)


def test_a_repeated_tile_number_is_rendered_once(comfy_stubs, monkeypatch):
    recorded, (rendered_tiles, _blocks) = _render(monkeypatch, tiles=" 10 , 5 ,10")

    assert len(recorded["calls"]) == 2
    assert len(rendered_tiles) == 2
    assert recorded["calls"][0]["layout"].first_col == 1


def test_denoise_zero_returns_the_input_before_any_model_call(comfy_stubs, monkeypatch):
    # build_sigmas answers an empty schedule at denoise 0, and the CLIP call below it costs
    # minutes on a cold cache, so nothing after it may run.
    layout = _render_layout()
    recorded, (rendered_tiles, blocks) = _render(monkeypatch, layout=layout, tiles="5",
                                                 sigmas=torch.FloatTensor([]), denoise=0.0)
    crop = layout.layout.tiles[5].crop_rect

    assert recorded["calls"] == [] and recorded["encode_empty"] is None
    assert torch.equal(rendered_tiles[0], recorded["image"][:, crop.y0:crop.y1, crop.x0:crop.x1, :3])
    assert tuple(blocks[0].shape) == (1, 520, 520, 3)


def test_denoise_zero_on_the_full_run_returns_the_picture(comfy_stubs, monkeypatch):
    recorded, (rendered_tiles, blocks) = _render(monkeypatch, sigmas=torch.FloatTensor([]),
                                                 denoise=0.0)

    assert len(rendered_tiles) == 1 and len(blocks) == 1
    assert torch.equal(rendered_tiles[0], recorded["image"][..., :3])


def test_the_render_node_rejects_a_batch_above_one(comfy_stubs, monkeypatch):
    with pytest.raises(ValueError, match="got a batch of 2"):
        _render(monkeypatch, image=torch.rand(2, 600, 600, 3))


def test_the_render_node_rejects_an_image_the_layout_was_not_solved_for(comfy_stubs, monkeypatch):
    with pytest.raises(ValueError, match=r"800x600 image, but the layout was solved for 600x600"):
        _render(monkeypatch, image=torch.rand(1, 600, 800, 3))


@pytest.mark.parametrize("text", ["two", "5,x", "1.5"])
def test_the_render_node_rejects_a_csv_entry_that_is_not_a_number(comfy_stubs, monkeypatch, text):
    with pytest.raises(ValueError, match="which is not a tile number"):
        _render(monkeypatch, tiles=text)


@pytest.mark.parametrize("number", [16, -1, 99])
def test_the_render_node_rejects_a_tile_number_outside_the_grid(comfy_stubs, monkeypatch, number):
    with pytest.raises(ValueError, match=r"this layout has 16 tiles numbered 0 to 15"):
        _render(monkeypatch, tiles=str(number))


def test_the_render_node_rejects_both_token_counts_at_zero(comfy_stubs, monkeypatch):
    # The settings file rejects the same pair at load: a tile needs rows from one of the two.
    with pytest.raises(ValueError, match="canvas_tokens 0 and crop_tokens 0"):
        _render(monkeypatch, canvas_tokens=0, crop_tokens=0)


def test_a_caption_surface_with_no_captions_connected_is_rejected(comfy_stubs, monkeypatch):
    with pytest.raises(ValueError, match=r"'captions' surface with no captions connected"):
        _render(monkeypatch, surface=captions.VLM_METHOD_CAPTIONS)


def test_captions_written_at_another_size_are_rejected(comfy_stubs, monkeypatch):
    other = _captions_for(_render_layout(image=torch.rand(1, 600, 604, 3)))

    with pytest.raises(ValueError, match=r"written at 604x600, and the layout was solved for 600x600"):
        _render(monkeypatch, surface=captions.VLM_METHOD_CAPTIONS, given_captions=other)


def test_captions_from_a_different_grid_at_the_same_size_are_rejected(comfy_stubs, monkeypatch):
    # Same canvas, coarser tiles: the captions describe rects this run never samples, and the
    # message names both grids so the two nodes can be told apart.
    other = _captions_for(_render_layout(max_tile_width=384, max_tile_height=384))

    with pytest.raises(ValueError, match=r"captions for a 2x2 tile grid, and this layout is 4x4"):
        _render(monkeypatch, surface=captions.VLM_METHOD_CAPTIONS, given_captions=other)


def test_captions_whose_rects_differ_at_the_same_grid_are_rejected(comfy_stubs, monkeypatch):
    # A grid solved at another context_anchor keeps its 4x4 shape and moves every crop rect.
    other = _captions_for(_render_layout(context_anchor=16))

    with pytest.raises(ValueError, match=r"4x4 tile grid whose tile rects are not this 4x4 layout"):
        _render(monkeypatch, surface=captions.VLM_METHOD_CAPTIONS, given_captions=other)


def test_a_block_whose_ring_reaches_past_its_neighbour_is_rejected_before_the_clip_call(comfy_stubs, monkeypatch):
    # grid.sub_layout raises when context_anchor plus context_overlap exceeds the tile base, and
    # the solve happens in the validation stage so the message beats the text encoder load.
    layout = _render_layout(max_tile_width=280, max_tile_height=280, context_anchor=128,
                            context_overlap=0)
    recorded = {}

    with pytest.raises(ValueError, match="so a tile's frozen ring reaches past its bordering tile"):
        _render(monkeypatch, layout=layout, tiles="5", recorded=recorded)
    assert recorded["build_sigmas"] is None and recorded["encode_empty"] is None


def test_the_render_node_rejects_an_image_that_is_not_an_image_tensor(comfy_stubs, monkeypatch):
    # node._validate_image is the production nodes' own first check, and it runs first here too.
    with pytest.raises(ValueError, match=r"got 3 dimensions"):
        _render(monkeypatch, image=torch.rand(600, 600, 3))


def test_the_surface_widget_offers_exactly_the_engines_own_surfaces(comfy_stubs):
    widget = ContextAnchoredTileTestRender.INPUT_TYPES()["required"]["surface"]

    assert widget[0] == list(captions.VLM_SURFACES)
    assert widget[1]["default"] == captions.VLM_METHOD_VISION


def test_the_token_widgets_default_to_the_settings_files_own_values(comfy_stubs):
    required = ContextAnchoredTileTestRender.INPUT_TYPES()["required"]
    vision = captions.load_settings().vision

    assert required["canvas_tokens"][1]["default"] == vision.canvas_tokens
    assert required["crop_tokens"][1]["default"] == vision.crop_tokens
    for name in ("canvas_tokens", "crop_tokens"):
        assert (required[name][1]["min"], required[name][1]["max"]) == (0, vl.MAX_VISION_TOKENS)


def test_the_render_node_returns_two_image_lists():
    assert ContextAnchoredTileTestRender.RETURN_TYPES == ("IMAGE", "IMAGE")
    assert ContextAnchoredTileTestRender.RETURN_NAMES == ("tiles", "blocks")
    assert ContextAnchoredTileTestRender.OUTPUT_IS_LIST == (True, True)
