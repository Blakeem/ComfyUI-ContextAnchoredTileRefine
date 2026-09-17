"""captions.py: the two caption conditioning surfaces of the VL nodes.

What is pinned here: the settled instruction/token pair each vlm_method asks for, the
reasoning-turn strip, that the VLM reads a 384-budget COPY and never the sampled tile,
per-batch-row captioning, the mask-path framing (captions describe the region crop, the
vision encode reads the full image at the bbox offset), the cancellable pre-pass, and every
fail-fast guard. A duck-typed clip stands in for the VL text encoder; no comfy install and
no model are needed.
"""
import dataclasses

import pytest
import torch
from test_vl import Tile, VLGuider, layout_tiles, strip_tiles, sync_sampler

from context_anchored_tile_refine import captions, grid, sampling, vl
from context_anchored_tile_refine.grid import Rect

SIGMAS = torch.linspace(1.0, 0.0, 5)  # 4 steps

# Fixture geometry, mirroring test_vl.py: canvas 192x128 px, encode 96x64 px -> a 3x2
# merged-cell grid, n_rows = 6.
CANVAS_H, CANVAS_W = 128, 192
ENC_H, ENC_W = 64, 96
N_ROWS = 6
TAIL = 4


def a_vision(canvas=1, crop=0, caption_mp=None):
    # The [vision] table a test hands the pre-pass: under the stubbed resample only on/off
    # matters for the two token counts, and the caption picture defaults to the 384-budget
    # every test that predates the presets captioned at.
    return captions.VisionSettings(
        canvas_tokens=canvas, crop_tokens=crop,
        caption_megapixels=captions.VL_INPUT_BUDGET_MEGAPIXELS if caption_mp is None else caption_mp)


def a_preset(tile="describe", tile_tokens=256, style="", style_tokens=128,
             surface=None, caption_mp=None):
    # A resolved settings block, which is what the caption pipeline takes. The defaults keep
    # every test that predates the presets asking its own question at its own budget.
    return captions.Preset(
        surface=captions.VLM_METHOD_CAPTIONS if surface is None else surface,
        label="test",
        vision=a_vision(caption_mp=caption_mp),
        tile_instruction=tile, tile_max_tokens=tile_tokens,
        style_instruction=style, style_max_tokens=style_tokens)


class FakeCaptionClip:
    """Duck-typed VL clip with a text generator.

    The token stream mirrors the Krea 2 layout captions._tokenize_images parses (prefix,
    vision_start, ONE dict image token, vision_end, tail), and the tail grows with the text
    so two different captions really do produce two different sequence lengths. Every
    tokenize/generate/encode is recorded, and the encode is deterministic: feature value ==
    sequence position.
    """

    def __init__(self, n_rows=N_ROWS, tail_len=TAIL, answer=None, seq_override=None,
                 text_seq_override=None, text_pooled=None, text_extras=None):
        self.n_rows = n_rows
        self.tail_len = tail_len
        self.answer = answer if answer is not None else (lambda image, instruction: "<think>weighing it up</think>a plain caption")
        self.seq_override = seq_override
        # The text-only half of the slice+caption surface, knobbed separately so a broken
        # caption encode can be tested without also breaking the vision encode. Krea 2 returns
        # pooled_output None on a text-only encode (probe C, docs/vl-conditioning-encode-cost.md
        # section 10), which is what makes the concatenation lossless.
        self.text_seq_override = text_seq_override
        self.text_pooled = text_pooled
        self.text_extras = {} if text_extras is None else text_extras
        self.tokenize_calls = []
        self.generate_calls = []
        self.encoded = []
        self._answers = {}

    def tokenize(self, text, images=None, llama_template=None, thinking=None):
        image = None if images is None else images[0]
        self.tokenize_calls.append({"text": text, "image": image, "llama_template": llama_template, "thinking": thinking})
        body = text[len(vl.VISION_BLOCK):] if text.startswith(vl.VISION_BLOCK) else text
        tail_rows = self.tail_len + len(body.split())
        stream = [(10, 1.0)] * 5
        if image is not None:
            stream += [(151652, 1.0), ({"type": "image"}, 1.0), (151653, 1.0)]
        stream += [(20, 1.0)] * tail_rows
        # "_probe" rides behind the stream key so the production `next(iter(tokens))` still
        # picks the stream; it is what lets generate/encode see what they were handed.
        return {"qwen3vl_4b": [stream], "_probe": (text, image, tail_rows)}

    def generate(self, tokens, **kwargs):
        text, image, _tail = tokens["_probe"]
        self.generate_calls.append({"text": text, "image": image, **kwargs})
        handle = len(self.generate_calls)
        self._answers[handle] = self.answer(image, text)
        return handle

    def decode(self, token_ids, skip_special_tokens=True):
        return self._answers[token_ids]

    def encode_from_tokens_scheduled(self, tokens):
        text, image, tail_rows = tokens["_probe"]
        self.encoded.append(text)
        if image is None:                         # a text-only encode is only its own rows
            seq = tail_rows if self.text_seq_override is None else self.text_seq_override
            extras = dict(self.text_extras, pooled_output=self.text_pooled,
                          attention_mask=torch.ones(1, seq))
        else:
            seq = 1 + self.n_rows + 1 + tail_rows if self.seq_override is None else self.seq_override
            # Krea 2 returns pooled_output None on an image encode as well (measured), which is
            # what lets two vision blocks and a caption concatenate.
            extras = {"pooled_output": None, "attention_mask": torch.ones(1, seq)}
        tensor = torch.arange(seq, dtype=torch.float32).reshape(1, seq, 1).expand(1, seq, 8).clone()
        return [[tensor, extras]]


@pytest.fixture
def stubbed_slices(comfy_stubs, monkeypatch):
    # Encode geometry pinned to the fixture grid; _convert identity so the slice tensors
    # stay inspectable. comfy_stubs serves vl.encode_picture's interrupt check.
    monkeypatch.setattr(vl, "resample_picture", lambda source, budget: (source, ENC_H, ENC_W))
    monkeypatch.setattr(vl, "_convert", lambda cond_list: cond_list)


# --- the settled instructions --------------------------------------------------------

def test_settled_instructions_are_the_ab_settled_strings():
    # A/B-settled wording, EU spelling included: the owner compared US against EU in ComfyUI
    # and US dropped detail. Any reflow, reword or Americanisation is a silent quality change,
    # so the exact strings are pinned here rather than merely described.
    assert captions.GROUP_CLAUSE == "Name whole objects and count repeated objects as one entry."
    assert captions.SETTLED_POSITION_INSTRUCTION == (
        "List up to eight main things in this image, one per line, each a short phrase naming "
        "the thing, its position in the frame, and how much of it shows. "
        "Name whole objects and count repeated objects as one entry.")
    assert captions.SETTLED_RICH_INSTRUCTION == (
        "Describe this image, one short line per part. Start with the overall style, palette "
        "and lighting. Then say what fills the left, the centre, the right, the top and the "
        "bottom, giving each part its own description with what is there, its colour and what "
        "its surface is made of.")
    assert captions.RICH_GROUPED_INSTRUCTION == (
        captions.SETTLED_RICH_INSTRUCTION + " " + captions.GROUP_CLAUSE)
    assert "centre" in captions.SETTLED_RICH_INSTRUCTION
    assert "colour" in captions.SETTLED_RICH_INSTRUCTION
    assert captions.SETTLED_POSITION_MAX_TOKENS == 512
    assert captions.SETTLED_RICH_MAX_TOKENS == 768


# --- the settings file ------------------------------------------------------------------

# One valid file, as a template every broken-file case below edits one line of: the
# [vision] table, then one preset block.
VISION = (
    '[vision]\n'
    'canvas_tokens = 165\n'
    'crop_tokens = 100\n'
    'caption_megapixels = 0.147456\n'
    '\n')
PRESET = (
    '[presets.demo]\n'
    'tile_caption_instruction = "ask about the tile"\n'
    'tile_caption_max_tokens = 768\n'
    'global_style_instruction = ""\n'
    'global_style_max_tokens = 512\n')
GOOD_SETTINGS = VISION + PRESET


def write_settings(tmp_path, body, monkeypatch=None):
    # A settings file at a throwaway path, optionally installed as the one in force. The
    # method cache is cleared either way, since it is built once per session by design.
    path = tmp_path / "settings.toml"
    path.write_text(body)
    if monkeypatch is not None:
        monkeypatch.setattr(captions, "settings_path", lambda: path)
    captions.vlm_methods.cache_clear()
    return path


def test_settings_toml_ships_the_owner_tested_wording():
    # The live prompts, pinned character for character: the owner's testing found small
    # wording changes lose consistency, so an accidental edit fails here. A deliberate
    # prompt change updates this pin alongside settings.toml.
    settings = captions.load_settings()
    presets = settings.presets
    # The [vision] table: the block A/B's settled point (TESTS.md test 10), and the caption
    # picture at the vision encode's old size, the three-scene A/B's winner.
    assert settings.vision == captions.VisionSettings(
        canvas_tokens=165, crop_tokens=110, caption_megapixels=captions.SHIPPED_CAPTION_MEGAPIXELS)
    assert captions.SHIPPED_CAPTION_MEGAPIXELS == 768 * 1024 / 1_000_000
    # `prompted` is FIRST, which is what makes it the default preset the selector offers
    # unlabeled: the owner's wording under test since 2026-09-16, which asks for the image's
    # prompt and holds the caption to the crop. `standard` is the pre-settings-file constants
    # character for character, so a workflow that spells its label out still asks what it
    # asked then.
    assert list(presets) == ["prompted", "standard", "artwork"]
    assert presets["prompted"]["tile_caption_instruction"] == (
        "Concise prose containing object's relative and absolute positions within the "
        'foreground and background of the cropped image. Full prompt: "{PROMPT}". Keep it '
        "concise. Do not include anything not in the crop.")
    assert presets["prompted"]["global_style_instruction"] == ""
    assert presets["standard"]["tile_caption_instruction"] == captions.RICH_GROUPED_INSTRUCTION
    assert presets["standard"]["global_style_instruction"] == ""
    artwork = presets["artwork"]
    assert artwork["tile_caption_instruction"] == (
        "succinct prose containing relative and absolute positions of specific things with "
        "object and character identifying demographics.")
    assert artwork["global_style_instruction"] == (
        "succinct flowing prose of only the overall style and artistic medium and physical "
        "medium. No objects or items in the scene.")
    for preset in presets.values():
        assert preset["tile_caption_max_tokens"] == 768
        assert preset["global_style_max_tokens"] == 768


def test_every_preset_adds_one_option_per_caption_surface():
    # A preset's own two options sit together and in file order, so the selector reads the way
    # the settings file was written. The vision-only surface leads and carries no label,
    # because it asks the VLM nothing. The FIRST preset's two options carry no label either,
    # which is what makes them the strings a pre-preset workflow already holds.
    assert list(captions.vlm_methods()) == [
        "vision tokens",
        "vision tokens and captions", "captions",
        "vision tokens and captions (standard)", "captions (standard)",
        "vision tokens and captions (artwork)", "captions (artwork)",
    ]
    assert captions.default_vlm_method() == captions.VLM_METHOD_VISION_CAPTIONS


def test_the_default_preset_answers_to_both_forms_of_its_name():
    # The selector offers the default preset unlabeled, but a workflow that spells its label
    # out (what a workflow saved between 1.6.0 and the default preset carries) must reach the
    # SAME block rather than a "no such preset" error.
    unlabeled = captions.resolve_method("vision tokens and captions")
    labeled = captions.resolve_method("vision tokens and captions (prompted)")
    assert unlabeled == labeled
    assert unlabeled.label == "prompted"
    # A later preset resolves by its label alone, and `standard` still asks the settled wording.
    standard = captions.resolve_method("vision tokens and captions (standard)")
    assert standard.tile_instruction == captions.RICH_GROUPED_INSTRUCTION


# --- the prompt input -----------------------------------------------------------------

def _prompted_preset(tile='Full prompt: "{PROMPT}". Name it.', style=""):
    return captions.Preset(
        surface=captions.VLM_METHOD_CAPTIONS, label="demo",
        vision=captions.VisionSettings(canvas_tokens=165, crop_tokens=110, caption_megapixels=0.15),
        tile_instruction=tile, tile_max_tokens=768, style_instruction=style, style_max_tokens=768)


def test_with_prompt_fills_both_instructions_and_strips_the_prompt():
    # A multiline widget hands its text over with a trailing newline, which must not land
    # inside the instruction's quotes. Every other field rides through untouched.
    preset = _prompted_preset(style="Style of {PROMPT}, and again {PROMPT}.")

    filled = captions.with_prompt(preset, "  a fox in the centre\n")

    assert filled.tile_instruction == 'Full prompt: "a fox in the centre". Name it.'
    assert filled.style_instruction == "Style of a fox in the centre, and again a fox in the centre."
    assert filled == dataclasses.replace(preset, tile_instruction=filled.tile_instruction,
                                         style_instruction=filled.style_instruction)


def test_with_prompt_keeps_braces_in_the_prompt_literal():
    # A literal replace, never str.format: a prompt can carry its own braces.
    filled = captions.with_prompt(_prompted_preset(), "a {red} fox")

    assert filled.tile_instruction == 'Full prompt: "a {red} fox". Name it.'


def test_with_prompt_is_a_no_op_on_a_preset_without_the_placeholder():
    # The shipped `standard` and `artwork` presets and the vision-only surface read no prompt,
    # so a typed prompt changes nothing there, blank or not.
    for method in ("captions (standard)", "captions (artwork)", "vision tokens"):
        preset = captions.resolve_method(method)
        assert captions.with_prompt(preset, "a fox") == preset, method
        assert captions.with_prompt(preset, "") == preset, method


@pytest.mark.parametrize(("tile", "style", "key"), [
    ('Full prompt: "{PROMPT}".', "", "tile_caption_instruction"),
    ("Name it.", "Style of {PROMPT}.", "global_style_instruction"),
])
@pytest.mark.parametrize("prompt", ["", "   \n"])
def test_with_prompt_refuses_a_blank_prompt_where_an_instruction_asks_for_one(tile, style, key, prompt):
    with pytest.raises(RuntimeError, match=rf"preset 'demo' asks for \{{PROMPT\}} in its {key}.*prompt input is empty"):
        captions.with_prompt(_prompted_preset(tile=tile, style=style), prompt)


def test_the_caption_pass_refuses_an_unfilled_placeholder_before_any_generate(comfy_stubs):
    # A direct caller that skipped with_prompt would otherwise ask every tile a question
    # holding the literal placeholder.
    clip = FakeCaptionClip()

    with pytest.raises(RuntimeError, match=r"still carries \{PROMPT\} in its tile_caption_instruction"):
        captions.generate_caption_set(clip, torch.rand(1, 64, 64, 3), [], _prompted_preset())

    assert clip.generate_calls == []


def test_the_method_list_is_built_once_per_session(tmp_path, monkeypatch):
    # The frontend caches a node's definition at startup, so a list that changed between
    # calls would offer values the backend then rejects. A preset added mid-session must
    # therefore NOT appear until a restart, which is what the cache buys — so the file is
    # rewritten here with no clear and the first answer has to stand. The renamed preset is
    # the SECOND one, because the first is the unlabeled default and a rename there would not
    # show in the list at all.
    two_presets = GOOD_SETTINGS + PRESET.replace("[presets.demo]", "[presets.extra]")
    unlabeled = ["vision tokens", "vision tokens and captions", "captions"]
    path = write_settings(tmp_path, two_presets, monkeypatch)
    assert list(captions.vlm_methods()) == [
        *unlabeled, "vision tokens and captions (extra)", "captions (extra)"]

    path.write_text(two_presets.replace("[presets.extra]", "[presets.added]"))
    assert list(captions.vlm_methods()) == [
        *unlabeled, "vision tokens and captions (extra)", "captions (extra)"]

    captions.vlm_methods.cache_clear()             # a restart, and the new preset appears
    assert list(captions.vlm_methods()) == [
        *unlabeled, "vision tokens and captions (added)", "captions (added)"]


@pytest.mark.parametrize(("vlm_method", "surface", "label"), [
    ("vision tokens", "vision tokens", ""),
    ("captions (artwork)", "captions", "artwork"),
    ("vision tokens and captions (standard)", "vision tokens and captions", "standard"),
    # What a workflow saved before the presets existed carries.
    ("captions", "captions", ""),
    ("vision tokens and captions", "vision tokens and captions", ""),
])
def test_a_method_splits_into_its_surface_and_its_label(vlm_method, surface, label):
    assert captions.method_surface(vlm_method) == surface
    assert captions.method_label(vlm_method) == label


def test_an_unlabeled_caption_method_takes_the_first_preset(tmp_path, monkeypatch):
    # A workflow saved before the presets existed still runs, on the file's first block —
    # whichever block that is, so a user who reorders their own copy moves the default with it.
    write_settings(tmp_path, GOOD_SETTINGS, monkeypatch)
    legacy = captions.resolve_method("vision tokens and captions")
    assert legacy.label == "demo"
    assert legacy.surface == captions.VLM_METHOD_VISION_CAPTIONS
    assert legacy.tile_instruction == "ask about the tile"


def test_each_caption_method_asks_its_own_presets_question():
    # Both caption surfaces ask the SAME tile question of a given preset, as they have since
    # 2026-08-16, and the wording comes from the settings file rather than a code constant.
    # The settled constants stay defined for tests-AB's judged arms, and nothing selects them.
    settings = captions.load_settings()
    presets = settings.presets
    for label, block in presets.items():
        for surface in captions.CAPTION_SURFACES:
            preset = captions.resolve_method(f"{surface} ({label})")
            assert preset.surface == surface
            assert preset.label == label
            assert preset.tile_instruction == block["tile_caption_instruction"]
            assert preset.tile_max_tokens == block["tile_caption_max_tokens"]
            # Every option carries the one [vision] table.
            assert preset.vision == settings.vision
    assert presets["artwork"]["tile_caption_instruction"] != captions.SETTLED_POSITION_INSTRUCTION


def test_vision_tokens_resolves_to_the_vision_table_and_no_preset():
    # "vision tokens" never reaches the VLM's generator, so it takes no preset at all; it
    # does take the [vision] table, which sizes every surface's vision rows.
    preset = captions.resolve_method(captions.VLM_METHOD_VISION)
    assert preset.surface == captions.VLM_METHOD_VISION
    assert preset.label == ""
    assert preset.tile_instruction == ""
    assert preset.style_instruction == ""
    assert preset.vision == captions.load_settings().vision


def test_a_blank_style_instruction_turns_the_style_caption_off(tmp_path, monkeypatch):
    # "" is the documented off switch, and whitespace must not sneak past it.
    write_settings(tmp_path, GOOD_SETTINGS.replace('global_style_instruction = ""',
                                                 'global_style_instruction = "  "'), monkeypatch)

    preset = captions.resolve_method("captions (demo)")
    assert preset.style_instruction == ""
    assert preset.style_max_tokens == 512


def test_a_method_naming_an_absent_preset_is_a_named_hard_error(tmp_path, monkeypatch):
    # The selector is built at startup while the wording is read per run, so a preset renamed
    # mid-session leaves a stale option behind. It must name the preset, not fail obscurely.
    write_settings(tmp_path, GOOD_SETTINGS, monkeypatch)

    with pytest.raises(RuntimeError, match="asks for preset 'gone'"):
        captions.resolve_method("captions (gone)")


@pytest.mark.parametrize(("content", "message"), [
    (None, "is missing at"),
    ('[presets.demo\n', "is not valid TOML"),
    ('tile_caption_instruction = "x"\n' + GOOD_SETTINGS, "unknown top-level keys"),
    ('[other.demo]\nx = 1\n', "unknown top-level keys"),
    ('# nothing at all\n', r"has no \[vision\] table"),
    (PRESET, r"has no \[vision\] table"),
    (VISION, "defines no presets"),
    (VISION + '[presets]\n', "defines no presets"),
    (VISION + '[presets."bad (label)"]\n', "not usable"),
    (VISION + '[presets]\ndemo = 1\n', r"must be a \[presets.demo\] table"),
    ('vision = 1\n' + PRESET, r"\[vision\] in .* must be a table"),
    (GOOD_SETTINGS.replace('crop_tokens = 100\n', ''), "missing \\['crop_tokens'\\]"),
    (GOOD_SETTINGS.replace('crop_tokens = 100\n', 'crop_tokens = 100\ncrop_megapixels = 0.1\n'),
     "unknown keys \\['crop_megapixels'\\]"),
    (GOOD_SETTINGS.replace("canvas_tokens = 165", 'canvas_tokens = "165"'), "must be of type int"),
    (GOOD_SETTINGS.replace("canvas_tokens = 165", "canvas_tokens = 1.5"), "must be of type int"),
    (GOOD_SETTINGS.replace("crop_tokens = 100", "crop_tokens = true"), "must be of type int"),
    (GOOD_SETTINGS.replace("crop_tokens = 100", "crop_tokens = -1"), "between 0 and 1953"),
    (GOOD_SETTINGS.replace("canvas_tokens = 165", "canvas_tokens = 2000"), "between 0 and 1953"),
    (GOOD_SETTINGS.replace("canvas_tokens = 165", "canvas_tokens = 0").replace("crop_tokens = 100", "crop_tokens = 0"),
     "both to 0"),
    (GOOD_SETTINGS.replace("caption_megapixels = 0.147456", 'caption_megapixels = "big"'), "must be of type float"),
    (GOOD_SETTINGS.replace("caption_megapixels = 0.147456", "caption_megapixels = 8.0"), "between 0.01 and 2.0"),
    (GOOD_SETTINGS.replace("caption_megapixels = 0.147456", "caption_megapixels = -1.0"), "between 0.01 and 2.0"),
    # Below the floor the budget rounds to no pixels and the resample would build a 0 x 0 image.
    (GOOD_SETTINGS.replace("caption_megapixels = 0.147456", "caption_megapixels = 1e-9"), "between 0.01 and 2.0"),
    (GOOD_SETTINGS.replace('tile_caption_max_tokens = 768\n', ''), "missing \\['tile_caption_max_tokens'\\]"),
    (GOOD_SETTINGS + 'globl_style_instruction = "typo"\n', "unknown keys \\['globl_style_instruction'\\]"),
    # A user's own copy from before the [vision] table names the move, not "unknown key".
    (GOOD_SETTINGS + 'tile_caption_megapixels = 0.5\n', "no longer reads"),
    (GOOD_SETTINGS.replace("tile_caption_max_tokens = 768", 'tile_caption_max_tokens = "768"'), "must be of type int"),
    (GOOD_SETTINGS.replace("tile_caption_max_tokens = 768", "tile_caption_max_tokens = true"), "must be of type int"),
    (GOOD_SETTINGS.replace('tile_caption_instruction = "ask about the tile"',
                         'tile_caption_instruction = "  "'), "need a question"),
    (GOOD_SETTINGS.replace("tile_caption_max_tokens = 768", "tile_caption_max_tokens = 0"), "between 1 and 4096"),
    (GOOD_SETTINGS.replace("tile_caption_max_tokens = 768", "tile_caption_max_tokens = 9999"), "between 1 and 4096"),
])
def test_a_broken_settings_file_is_a_named_hard_error(tmp_path, content, message):
    # Every defect fails before any clip.generate spends GPU time, naming the file and the
    # defect. The unknown-key case is the typo guard: a misspelled key would otherwise
    # change nothing, silently.
    path = tmp_path / "settings.toml"
    if content is not None:
        path.write_text(content)

    with pytest.raises(RuntimeError, match=message):
        captions.load_settings(path)


def test_a_toml_int_is_accepted_where_a_float_is_asked_for(tmp_path):
    # 0 is the documented "the crop's own size" value and TOML parses it as an int, so the
    # float keys must take one.
    path = write_settings(tmp_path, GOOD_SETTINGS.replace("caption_megapixels = 0.147456",
                                                          "caption_megapixels = 0"))

    assert captions.load_settings(path).vision.caption_megapixels == 0


def test_a_non_utf8_settings_file_is_a_named_hard_error(tmp_path):
    # An editor saving in a legacy codepage raises UnicodeDecodeError, not TOMLDecodeError:
    # a cp1252 curly quote must still land in the named RuntimeError.
    path = tmp_path / "settings.toml"
    path.write_bytes(b'[presets.demo]\ntile_caption_instruction = "caf\x92"\n')

    with pytest.raises(RuntimeError, match="not UTF-8"):
        captions.load_settings(path)


def test_the_users_own_copy_wins_over_the_shipped_file(tmp_path, monkeypatch):
    # settings.user.toml is the edit surface that a node update never replaces, so it is used
    # whenever it exists and the shipped file is ignored.
    # The autouse fixture renames USER_SETTINGS_NAME so a developer's own copy cannot reach
    # the gate. This is the one test that needs the real pair, so it pins both names.
    assert captions.SETTINGS_NAME == "settings.toml"
    monkeypatch.setattr(captions, "USER_SETTINGS_NAME", "settings.user.toml")
    monkeypatch.setattr(captions, "SETTINGS_DIR", tmp_path)
    (tmp_path / captions.SETTINGS_NAME).write_text(GOOD_SETTINGS)
    assert captions.settings_path().name == captions.SETTINGS_NAME

    (tmp_path / captions.USER_SETTINGS_NAME).write_text(
        GOOD_SETTINGS.replace("[presets.demo]", "[presets.mine]"))
    assert captions.settings_path().name == captions.USER_SETTINGS_NAME
    assert list(captions.load_settings().presets) == ["mine"]


def a_settings_dir(tmp_path, monkeypatch):
    # The shipped file at a throwaway directory, so a fingerprint test can edit the file in
    # force without touching the real one.
    monkeypatch.setattr(captions, "SETTINGS_DIR", tmp_path)
    (tmp_path / captions.SETTINGS_NAME).write_text(GOOD_SETTINGS)


def test_the_settings_fingerprint_is_stable_across_calls(tmp_path, monkeypatch):
    a_settings_dir(tmp_path, monkeypatch)

    first = captions.settings_fingerprint()

    assert first == captions.settings_fingerprint()
    assert first.startswith(f"{captions.SETTINGS_NAME}:")


def test_the_settings_fingerprint_changes_when_the_file_changes(tmp_path, monkeypatch):
    # This is the value ComfyUI compares with the previous run's, so an edited preset has to
    # move it or the node is served from the cache and never runs.
    a_settings_dir(tmp_path, monkeypatch)
    before = captions.settings_fingerprint()

    (tmp_path / captions.SETTINGS_NAME).write_text(
        GOOD_SETTINGS.replace("ask about the tile", "ask something else"))

    assert captions.settings_fingerprint() != before


def test_the_settings_fingerprint_changes_when_a_user_copy_appears(tmp_path, monkeypatch):
    # A user copy carrying the shipped file's exact bytes still takes the run over, which is
    # why the file name rides beside the digest.
    a_settings_dir(tmp_path, monkeypatch)
    monkeypatch.setattr(captions, "USER_SETTINGS_NAME", "settings.user.toml")
    before = captions.settings_fingerprint()

    (tmp_path / captions.USER_SETTINGS_NAME).write_text(GOOD_SETTINGS)

    assert captions.settings_fingerprint() != before


def test_a_missing_settings_file_fingerprints_without_raising(tmp_path, monkeypatch):
    # IS_CHANGED runs while the prompt is queued, so a missing file must not stop the queue.
    # The run-time read is what raises the message that names the file.
    monkeypatch.setattr(captions, "SETTINGS_DIR", tmp_path)

    assert captions.settings_fingerprint() == f"missing:{captions.SETTINGS_NAME}"


@pytest.mark.parametrize(("megapixels", "size", "expected"), [
    (captions.VL_INPUT_BUDGET_MEGAPIXELS, (200, 300), captions.VL_INPUT_BUDGET),
    (1.0, (200, 300), 1_000_000),
    # 0 is the crop's own area, so the VL model reads every pixel the crop has...
    (0, (200, 300), 200 * 300),
    # ...up to the cap, which a 4000x3000 crop is well past.
    (0, (3000, 4000), 2_000_000),
])
def test_zero_megapixels_reads_the_crops_own_size_up_to_the_cap(megapixels, size, expected):
    source = torch.zeros(1, size[0], size[1], 3)

    assert captions.caption_budget_pixels(megapixels, source) == expected


def test_a_presets_megapixels_reach_the_resample_as_real_pixel_sizes(comfy_stubs):
    # The two halves joined: a preset's budget through caption_budget_pixels and into
    # resample_for_vl, which is the only place the number becomes an image the VLM reads.
    # 0 is the tile's own size, so that crop passes through at its native dimensions.
    tile = torch.zeros(1, 1024, 1536, 3)

    for megapixels, expected in ((captions.VL_INPUT_BUDGET_MEGAPIXELS, (314, 470)),
                                 (1.0, (816, 1225)),
                                 (0, (1024, 1536))):
        budget = captions.caption_budget_pixels(megapixels, tile)
        out = captions.resample_for_vl(tile, budget)
        assert tuple(out.shape[1:3]) == expected, megapixels


def test_the_settings_file_reaches_the_registry_archive():
    # settings.toml is load-bearing for the VL nodes appearing at all, so it must not be
    # excluded from the published archive the way docs/ and tests/ are.
    excluded = [line.strip() for line in
                (captions.SETTINGS_DIR / ".comfyignore").read_text().splitlines()
                if line.strip() and not line.startswith("#")]

    assert (captions.SETTINGS_DIR / captions.SETTINGS_NAME).is_file()
    assert captions.SETTINGS_NAME not in excluded


# --- strip_thinking / clean_caption ---------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("plain answer", "plain answer"),
    ("  padded  ", "padded"),
    ("<think>reasoning here</think>the answer", "the answer"),
    ("<think>a</think>mid<think>b</think>end", "midend"),
    # A stray close with no open (the model reopened mid-answer): keep what follows the LAST
    # close, which is core's own rule in TextGenerateLTX2Prompt.
    ("<think>one</think>middle</think>the answer", "the answer"),
    # Unbalanced open with no close at all. With thinking=True there is no answer in the
    # string, only a reasoning turn that ran out of budget, so "" is the honest result and it
    # is what makes generate_caption's fallback chain fire. Core's own rule, and returning the
    # reasoning instead would send it to the DiT as the tile's prompt.
    ("<think>never closed", ""),
    ("<think>reasoning that ran out of tok", ""),
])
def test_strip_thinking(raw, expected):
    # Mandatory, not cosmetic: the caption is encoded as text, so an unstripped block reaches
    # the DiT as hundreds of tokens of the model talking to itself.
    assert captions.strip_thinking(raw) == expected


def test_clean_caption_returns_the_original_string_when_nothing_is_removed():
    text = "**bold** opener\nsecond line\nthird line"
    assert captions.clean_caption(text) is text


@pytest.mark.parametrize("raw,expected", [
    ("**Answer:**\na tree, left", "a tree, left"),
    ("a tree, left\nWait, I need to rephrase.", "a tree, left"),
    ("a tree, left\na tree, left.\na cart, right", "a tree, left\na cart, right"),
])
def test_clean_caption_cuts_headings_meta_and_exact_repeats(raw, expected):
    assert captions.clean_caption(raw) == expected


# --- resample_for_vl ------------------------------------------------------------------

def test_resample_for_vl_uses_a_384_budget_copy_not_the_tile(comfy_stubs):
    # Prime directive 1: a sampled tile is never resampled. What the VLM reads is a separate
    # area-resampled COPY at 384x384 total pixels, aspect preserved.
    tile_pixels = torch.rand(1, 512, 1024, 3)

    vl_input = captions.resample_for_vl(tile_pixels)

    shape, width, height, method, crop = comfy_stubs["common_upscale_calls"][-1]
    assert shape == (1, 3, 512, 1024)          # channels-first, the crop itself
    assert (width, height) == (543, 272)       # sqrt(384*384 / (1024*512)) preserved 2:1
    assert abs(width * height - captions.VL_INPUT_BUDGET) / captions.VL_INPUT_BUDGET < 0.01
    assert (method, crop) == ("area", "disabled")
    assert vl_input.shape == (1, 272, 543, 3)
    assert vl_input is not tile_pixels
    assert vl_input.shape != tile_pixels.shape


# --- generate_caption -----------------------------------------------------------------

def test_generate_caption_asks_with_thinking_and_strips_the_reasoning():
    clip = FakeCaptionClip(answer=lambda image, instruction: "<think>hmm</think>a fox, centre")

    text = captions.generate_caption(clip, torch.zeros(1, 8, 8, 3),
                                     captions.SETTLED_POSITION_INSTRUCTION, 512)

    assert text == "a fox, centre"
    call = clip.tokenize_calls[0]
    assert call["text"] == captions.SETTLED_POSITION_INSTRUCTION
    assert call["thinking"] is True
    assert clip.generate_calls == [{
        "text": captions.SETTLED_POSITION_INSTRUCTION,
        "image": call["image"],
        "do_sample": False,
        "max_length": 512,
        "repetition_penalty": 1.05,
    }]


def test_generate_caption_falls_back_through_sampling_then_a_simpler_question():
    # A stop-token-degenerate crop answers empty; the settled chain retries with sampling on,
    # then re-asks a plainer question, before giving up.
    answers = ["", "", "<think>x</think>a wall of tools"]

    def answer(image, instruction):
        return answers[min(len(answers) - 1, len(clip.generate_calls) - 1)]

    clip = FakeCaptionClip(answer=answer)
    text = captions.generate_caption(clip, torch.zeros(1, 8, 8, 3), "describe", 256)

    assert text == "a wall of tools"
    assert [c["do_sample"] for c in clip.generate_calls] == [False, True, False]
    assert clip.generate_calls[1]["seed"] == 42
    assert clip.tokenize_calls[-1]["text"] == "Write one short sentence describing this image."
    assert clip.tokenize_calls[-1]["thinking"] is False


def test_generate_caption_raises_when_every_fallback_is_empty():
    clip = FakeCaptionClip(answer=lambda image, instruction: "")
    with pytest.raises(RuntimeError, match="empty answer after every fallback"):
        captions.generate_caption(clip, torch.zeros(1, 8, 8, 3), "describe", 256)


def test_generate_caption_rejects_a_clip_that_cannot_generate():
    class NoGeneratorClip:
        def tokenize(self, text, images=None, **kwargs):
            return {"l": [[(1, 1.0)]]}

    with pytest.raises(RuntimeError, match="cannot generate text"):
        captions.generate_caption(NoGeneratorClip(), torch.zeros(1, 8, 8, 3), "describe", 256)


def test_generate_caption_rejects_a_tokenizer_that_refuses_images():
    class StrictSignatureClip(FakeCaptionClip):
        def tokenize(self, text):
            return {"l": [[(1, 1.0)]]}

    with pytest.raises(RuntimeError, match="does not accept images"):
        captions.generate_caption(StrictSignatureClip(), torch.zeros(1, 8, 8, 3), "describe", 256)


def test_generate_caption_rejects_a_clip_without_image_tokens():
    class NoImageTokenClip(FakeCaptionClip):
        def tokenize(self, text, images=None, **kwargs):
            return {"qwen3vl_4b": [[(10, 1.0)] * 8], "_probe": (text, None, 4)}

    with pytest.raises(RuntimeError, match="no image tokens"):
        captions.generate_caption(NoImageTokenClip(), torch.zeros(1, 8, 8, 3), "describe", 256)


# --- the caption cache ----------------------------------------------------------------

def test_a_repeated_caption_request_is_served_from_the_cache():
    # A seed re-roll re-executes the node, and a greedy caption of the same picture, question
    # and budget cannot differ, so the second pass must not reach the VLM at all.
    clip = FakeCaptionClip(answer=lambda image, instruction: "a fox, centre")
    picture = torch.rand(1, 8, 8, 3)

    first = captions.generate_caption(clip, picture, "describe", 256)
    second = captions.generate_caption(clip, picture.clone(), "describe", 256)

    assert first == second == "a fox, centre"
    assert len(clip.generate_calls) == 1


@pytest.mark.parametrize("changed", ["pixels", "instruction", "max_length", "scope"])
def test_every_part_of_the_key_misses_when_it_changes(changed):
    clip = FakeCaptionClip(answer=lambda image, instruction: "a fox, centre")
    picture = torch.zeros(1, 8, 8, 3)
    request = {"vl_input": picture, "instruction": "describe", "max_length": 256,
               "scope": ("tile", 0, 0, 8, 8, 0, 0)}

    captions.generate_caption(clip, **request)
    if changed == "pixels":
        request["vl_input"] = picture.clone()
        request["vl_input"][0, 0, 0, 0] = 1.0
    elif changed == "instruction":
        request["instruction"] = "describe it differently"
    elif changed == "max_length":
        request["max_length"] = 512
    else:
        request["scope"] = ("tile", 8, 0, 16, 8, 0, 0)
    captions.generate_caption(clip, **request)

    assert len(clip.generate_calls) == 2


def test_a_different_clip_object_never_reads_another_clips_entry():
    # A second CLIP is a second model, and its answer to the same question is its own.
    picture = torch.zeros(1, 8, 8, 3)
    first = FakeCaptionClip(answer=lambda image, instruction: "first model")
    second = FakeCaptionClip(answer=lambda image, instruction: "second model")

    captions.generate_caption(first, picture, "describe", 256)
    text = captions.generate_caption(second, picture, "describe", 256)

    assert text == "second model"
    assert len(second.generate_calls) == 1


def test_a_bfloat16_picture_is_cached_rather_than_raising():
    # resample_for_vl keeps whatever dtype the IMAGE arrived with, and bfloat16 has no numpy
    # dtype, so the key hashes a float32 view of the picture.
    clip = FakeCaptionClip(answer=lambda image, instruction: "a fox, centre")
    picture = torch.zeros(1, 8, 8, 3, dtype=torch.bfloat16)

    captions.generate_caption(clip, picture, "describe", 256)
    captions.generate_caption(clip, picture.clone(), "describe", 256)

    assert len(clip.generate_calls) == 1


def test_the_oldest_entry_is_evicted_past_the_bound(monkeypatch):
    # The cache is bounded so a long session cannot grow it without limit.
    monkeypatch.setattr(captions, "CAPTION_CACHE_ENTRIES", 2)
    clip = FakeCaptionClip(answer=lambda image, instruction: "a fox, centre")
    picture = torch.zeros(1, 8, 8, 3)

    for index in range(3):
        captions.generate_caption(clip, picture, "describe", 256, scope=("tile", index))
    captions.generate_caption(clip, picture, "describe", 256, scope=("tile", 0))
    captions.generate_caption(clip, picture, "describe", 256, scope=("tile", 2))

    assert captions.CAPTION_CACHE_ENTRIES == 2
    assert len(clip.generate_calls) == 4          # the third request evicted scope ("tile", 0)


def test_two_byte_equal_tiles_in_one_run_are_captioned_separately(comfy_stubs):
    # comfy_stubs' resample hands every tile the same zero picture, so only the tile's place
    # in the run keeps the two requests apart.
    tiles = [Tile(Rect(0, 0, 16, 16)), Tile(Rect(16, 0, 32, 16))]
    clip = FakeCaptionClip()

    captions.generate_tile_captions(clip, torch.rand(1, 16, 32, 3), tiles, a_preset())

    assert len(clip.generate_calls) == 2


def test_a_second_pre_pass_over_the_same_tiles_reaches_no_vlm(comfy_stubs):
    # The whole point: a re-run captions nothing again, and the progress the UI reads is
    # still reported for every tile.
    reported = []

    class Recorder:
        def caption_done(self, index, count):
            reported.append((index, count))

    source = torch.rand(1, 16, 32, 3)
    tiles = [Tile(Rect(0, 0, 16, 16)), Tile(Rect(16, 0, 32, 16))]
    clip = FakeCaptionClip()

    first = captions.generate_tile_captions(clip, source, tiles, a_preset())
    asked = len(clip.generate_calls)
    second = captions.generate_tile_captions(clip, source, tiles, a_preset(),
                                             progress=Recorder())

    assert second == first
    assert len(clip.generate_calls) == asked == 2
    assert reported == [(1, 2), (2, 2)]


# --- generate_tile_captions -----------------------------------------------------------

def test_each_tile_is_captioned_from_its_own_crop(comfy_stubs, monkeypatch):
    # Identity resample so the crop the VLM is handed stays readable; the 384 budget is
    # covered by its own test above.
    monkeypatch.setattr(captions, "resample_for_vl", lambda pixels, budget=None: pixels)
    source = torch.rand(1, CANVAS_H, CANVAS_W, 3)
    tiles = [Tile(Rect(0, 0, 96, CANVAS_H)), Tile(Rect(96, 0, CANVAS_W, CANVAS_H))]
    clip = FakeCaptionClip(answer=lambda image, instruction: f"tile at {float(image[0, 0, 0, 0]):.6f}")

    out = captions.generate_tile_captions(clip, source, tiles, a_preset())

    assert out == [[f"tile at {float(source[0, 0, 0, 0]):.6f}"],
                   [f"tile at {float(source[0, 0, 96, 0]):.6f}"]]
    handed = [call["image"] for call in clip.tokenize_calls]
    assert torch.equal(handed[0], source[:, :, 0:96, :])
    assert torch.equal(handed[1], source[:, :, 96:CANVAS_W, :])


def test_a_batch_is_captioned_per_row_from_its_own_pixels(comfy_stubs, monkeypatch):
    # Core's tokenizer attaches images[0] alone, so a whole [B,H,W,3] crop would describe
    # every row with row 0's picture. Row b must be read from row b.
    monkeypatch.setattr(captions, "resample_for_vl", lambda pixels, budget=None: pixels)
    source = torch.zeros(2, 16, 16, 3)
    source[1] = 1.0
    tiles = [Tile(Rect(0, 0, 16, 16))]
    clip = FakeCaptionClip(answer=lambda image, instruction: f"row {int(image[0, 0, 0, 0])}")

    out = captions.generate_tile_captions(clip, source, tiles, a_preset())

    assert out == [["row 0", "row 1"]]
    assert [tuple(call["image"].shape) for call in clip.tokenize_calls] == [(1, 16, 16, 3)] * 2


def test_the_caption_pre_pass_is_cancellable_and_reports_progress(comfy_stubs):
    # One clip.generate per tile per row at up to 768 tokens: without an interrupt check and
    # a ProgressBar the node is uncancellable and silent until the first tile samples.
    source = torch.rand(2, 16, 16, 3)
    tiles = [Tile(Rect(0, 0, 16, 16)), Tile(Rect(0, 0, 16, 16)), Tile(Rect(0, 0, 16, 16))]

    captions.generate_tile_captions(FakeCaptionClip(), source, tiles, a_preset())

    assert comfy_stubs["interrupt_calls"] == len(tiles)
    pbar = comfy_stubs["progress_bars"][-1]
    assert pbar.total == len(tiles) * 2
    assert [update[0] for update in pbar.updates] == [1, 2, 3, 4, 5, 6]


def test_a_ledger_replaces_the_standalone_bar_and_receives_the_same_counters(comfy_stubs):
    # With the VL run's ledger in hand the pre-pass builds NO bar of its own — a second bar
    # is the display reset the ledger exists to remove — and reports each finished caption
    # with the counters that fed that bar, which is what a status line renders as "i/n".
    reported = []

    class Recorder:
        def caption_done(self, index, count):
            reported.append((index, count))

    source = torch.rand(1, 16, 16, 3)
    tiles = [Tile(Rect(0, 0, 16, 16)), Tile(Rect(0, 0, 16, 16))]
    before = len(comfy_stubs["progress_bars"])

    captions.generate_tile_captions(FakeCaptionClip(), source, tiles, a_preset(),
                                    progress=Recorder())

    assert reported == [(1, 2), (2, 2)]
    assert len(comfy_stubs["progress_bars"]) == before
    # The interrupt check is unconditional: a ledger must not make the pre-pass uncancellable.
    assert comfy_stubs["interrupt_calls"] == len(tiles)


def test_a_ledgers_caption_counters_stay_run_wide_across_the_picture_loop(comfy_stubs):
    # refine_image refines one picture at a time, so picture 2 of 2 reports 3/4 and 4/4 —
    # the ledger re-bases those onto that picture's own segment.
    reported = []

    class Recorder:
        def caption_done(self, index, count):
            reported.append((index, count))

    tiles = [Tile(Rect(0, 0, 16, 16)), Tile(Rect(0, 0, 16, 16))]

    captions.generate_tile_captions(FakeCaptionClip(), torch.rand(1, 16, 16, 3), tiles,
                                    a_preset(), batch_size=2, batch_index=1,
                                    progress=Recorder())

    assert reported == [(3, 4), (4, 4)]


def test_generated_captions_are_cleaned_before_they_are_returned(comfy_stubs):
    # clean_caption runs on the way out (caption_clean=True on every settled render), so the
    # DiT never reads the VLM's own headings or its repetition loop as content.
    clip = FakeCaptionClip(answer=lambda image, instruction: "**Answer:**\na fox, centre")

    out = captions.generate_tile_captions(clip, torch.rand(1, 16, 16, 3), [Tile(Rect(0, 0, 16, 16))],
                                          a_preset())

    assert out == [["a fox, centre"]]


# --- the whole-image style caption ----------------------------------------------------

def test_a_style_instruction_captions_the_style_source_first_and_prepends_it(comfy_stubs, monkeypatch):
    # ONE whole-image style caption, generated before any tile and placed on top of every
    # tile caption, so that all tiles follow one style description. It reads style_source
    # rather than source, since on the region path that is the full image.
    monkeypatch.setattr(captions, "resample_for_vl", lambda pixels, budget=None: pixels)
    source = torch.rand(1, 16, 16, 3)
    style_source = torch.rand(1, 32, 32, 3)
    tiles = [Tile(Rect(0, 0, 16, 16)), Tile(Rect(0, 0, 16, 16))]
    clip = FakeCaptionClip(answer=lambda image, instruction:
                           "oil on canvas" if instruction == "style q" else "a fox")

    out = captions.generate_tile_captions(clip, source, tiles, a_preset(style="style q"),
                                          style_source=style_source)

    assert out == [["oil on canvas\na fox"], ["oil on canvas\na fox"]]
    first = clip.tokenize_calls[0]
    assert first["text"] == "style q"
    assert torch.equal(first["image"], style_source)
    assert clip.generate_calls[0]["max_length"] == 128
    assert clip.generate_calls[1]["max_length"] == 256


def test_the_caption_set_keeps_the_style_apart_and_the_join_puts_it_on_top(comfy_stubs, monkeypatch):
    # generate_caption_set is what the Tile Test: Captions node lists from, and the join is the
    # engine's form, so the two together must equal generate_tile_captions byte for byte.
    monkeypatch.setattr(captions, "resample_for_vl", lambda pixels, budget=None: pixels)
    tiles = [Tile(Rect(0, 0, 16, 16)), Tile(Rect(0, 0, 16, 16))]
    clip = FakeCaptionClip(answer=lambda image, instruction:
                           "oil on canvas" if instruction == "style q" else "a fox")

    style, own = captions.generate_caption_set(clip, torch.rand(1, 16, 16, 3), tiles,
                                               a_preset(style="style q"))

    assert style == ["oil on canvas"]
    assert own == [["a fox"], ["a fox"]]
    assert captions.join_style_captions(style, own) == [["oil on canvas\na fox"]] * 2


def test_the_caption_set_without_a_style_has_nothing_to_join(comfy_stubs, monkeypatch):
    monkeypatch.setattr(captions, "resample_for_vl", lambda pixels, budget=None: pixels)
    clip = FakeCaptionClip(answer=lambda image, instruction: "a fox")

    style, own = captions.generate_caption_set(clip, torch.rand(1, 16, 16, 3),
                                               [Tile(Rect(0, 0, 16, 16))], a_preset())

    assert style == []
    assert own == [["a fox"]]
    assert captions.join_style_captions(style, own) is own


def test_the_style_caption_is_cleaned_like_any_other(comfy_stubs, monkeypatch):
    monkeypatch.setattr(captions, "resample_for_vl", lambda pixels, budget=None: pixels)
    clip = FakeCaptionClip(answer=lambda image, instruction:
                           "**Answer:**\nwatercolour" if instruction == "style q" else "a fox")

    out = captions.generate_tile_captions(clip, torch.rand(1, 16, 16, 3),
                                          [Tile(Rect(0, 0, 16, 16))], a_preset(style="style q"))

    assert out == [["watercolour\na fox"]]


def test_the_style_caption_counts_as_the_segments_first_chunk(comfy_stubs):
    # The run-wide counters span (tiles + 1) per picture, so picture 2 of 2 starts at 4/6:
    # the ledger's caption segment and preset_picture size themselves by the same arithmetic.
    reported = []

    class Recorder:
        def caption_done(self, index, count):
            reported.append((index, count))

    tiles = [Tile(Rect(0, 0, 16, 16)), Tile(Rect(0, 0, 16, 16))]

    captions.generate_tile_captions(FakeCaptionClip(), torch.rand(1, 16, 16, 3), tiles,
                                    a_preset(style="style q"), batch_size=2, batch_index=1,
                                    progress=Recorder())

    assert reported == [(4, 6), (5, 6), (6, 6)]


def test_a_style_canvas_with_a_different_batch_is_rejected(comfy_stubs):
    with pytest.raises(RuntimeError, match="style canvas has"):
        captions.generate_tile_captions(FakeCaptionClip(), torch.rand(2, 16, 16, 3),
                                        [Tile(Rect(0, 0, 16, 16))], a_preset(style="style q"),
                                        style_source=torch.rand(1, 32, 32, 3))


@pytest.mark.parametrize(("caption_mp", "expected"), [
    (1.0, [1_000_000, 1_000_000]),
    # 0 is each picture's own size: the style source first, then the tile crop.
    (0, [20 * 30, 40 * 60]),
])
def test_both_captions_read_the_vision_tables_caption_size(comfy_stubs, monkeypatch, caption_mp, expected):
    # The style question and the tile question read ONE picture size, the [vision] table's.
    budgets = []
    monkeypatch.setattr(captions, "resample_for_vl",
                        lambda pixels, budget=None: budgets.append(budget) or pixels)

    captions.generate_tile_captions(FakeCaptionClip(), torch.rand(1, 40, 60, 3),
                                    [Tile(Rect(0, 0, 60, 40))],
                                    a_preset(style="style q", caption_mp=caption_mp),
                                    style_source=torch.rand(1, 20, 30, 3))

    assert budgets == expected


# --- build_caption_conds --------------------------------------------------------------

def test_caption_conds_encode_each_tiles_caption_as_plain_text(stubbed_slices):
    clip = FakeCaptionClip()

    conds = captions.build_caption_conds(clip, [["a fox"], ["a cart, right"]])

    assert clip.encoded == ["a fox", "a cart, right"]
    assert all(call["image"] is None for call in clip.tokenize_calls)
    # Text-only: no vision grid rows at all, so the two encodes differ in length.
    assert [cond[0][0].shape[1] for cond in conds] == [TAIL + 2, TAIL + 3]


def test_caption_conds_take_the_single_row_of_each_tile(stubbed_slices):
    # [tile][row] with exactly ONE row: refine_image refines one picture at a time, so a
    # tile's positive is one plain text encode with nothing concatenated across rows.
    clip = FakeCaptionClip()

    conds = captions.build_caption_conds(clip, [["a fox here"]])

    tensor, extras = conds[0][0]
    assert tensor.shape == (1, TAIL + 3, 8)
    # Krea 2 produces no pooled output on a text-only encode; the extras still pass through.
    assert "pooled_output" in extras and extras["pooled_output"] is None


# --- build_slice_caption_conds --------------------------------------------------------

def _vision_rows(tile, offset_x=0, offset_y=0):
    # What the tile keeps of the shared pure-vision encode: vision_start, its own grid cells,
    # vision_end — and NO template tail (expected_seq = n_rows + 2 empties that range), because
    # the one tail in the concatenated stream arrives with the caption rows.
    return vl.slice_indices(tile.crop_rect, CANVAS_H, CANVAS_W, ENC_H, ENC_W, N_ROWS + 2,
                            offset_x, offset_y)


def test_slice_caption_conds_cat_each_tiles_vision_rows_and_its_own_caption(stubbed_slices):
    # The settled surface (2026-08-16): the tile's vision rows, then that tile's caption
    # encoded TEXT-ONLY, concatenated on the row axis. With the crop rows off the vision half
    # is the tile's slice of ONE canvas encode. The fake's feature value is the row's position
    # in its own encode, so the two halves are readable apart: vision rows carry their slice
    # indices, caption rows count 0..n-1.
    clip = FakeCaptionClip()
    tiles = strip_tiles(Rect(0, 0, 96, CANVAS_H), Rect(96, 0, CANVAS_W, CANVAS_H))
    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)

    conds = captions.build_slice_caption_conds(clip, source, tiles, [["a fox"], ["a cart"]], a_vision(canvas=1, crop=0))

    # One canvas encode for the run, then one cheap text encode per tile.
    assert clip.encoded == [vl.VISION_BLOCK, "a fox", "a cart"]
    caption_rows = TAIL + 2                                  # tail + the caption's two words
    for cond, tile in zip(conds, tiles, strict=True):
        indices = _vision_rows(tile)
        tensor, extras = cond[0]
        assert tensor.shape == (1, len(indices) + caption_rows, 8)
        assert tensor[0, :len(indices), 0].tolist() == indices
        assert tensor[0, len(indices):, 0].tolist() == list(range(caption_rows))
        # Extras are the VISION encode's; the full-canvas attention mask is still dropped.
        assert "attention_mask" not in extras
        assert extras["pooled_output"] is None


def test_slice_caption_conds_pay_one_canvas_encode_and_one_crop_encode_per_tile(stubbed_slices, monkeypatch):
    # The counting check behind the cost claim: this surface runs the same vision encodes the
    # vision-only surface does, the canvas once and each tile's crop once.
    clip = FakeCaptionClip()
    tiles = layout_tiles(2, 2)
    calls = []
    real_encode = vl._encode_one
    monkeypatch.setattr(vl, "_encode_one", lambda *a, **k: (calls.append(a[1]), real_encode(*a, **k))[1])

    captions.build_slice_caption_conds(clip, torch.zeros(1, CANVAS_H, CANVAS_W, 3), tiles,
                                       [["a fox"]] * 4, a_vision(canvas=1, crop=1))

    assert len(calls) == 5
    assert clip.encoded == [vl.VISION_BLOCK] * 5 + ["a fox"] * 4


def test_slice_caption_conds_put_each_tiles_crop_rows_before_its_canvas_slice(stubbed_slices):
    # Both vision sources on: a tile's rows are every cell of its own crop encode (no tail),
    # then its canvas slice (no tail), then its caption with the one tail. The caption half is
    # untouched by the vision layout.
    clip = FakeCaptionClip()
    tiles = strip_tiles(Rect(0, 0, 96, CANVAS_H), Rect(96, 0, CANVAS_W, CANVAS_H))
    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)

    conds = captions.build_slice_caption_conds(clip, source, tiles, [["a fox"]] * 2, a_vision(canvas=1, crop=1))

    assert clip.encoded == [vl.VISION_BLOCK] * 3 + ["a fox"] * 2
    caption_rows = TAIL + 2
    crop_block = list(range(N_ROWS + 2))
    for cond, tile in zip(conds, tiles, strict=True):
        indices = crop_block + _vision_rows(tile)
        tensor, _extras = cond[0]
        assert tensor.shape == (1, len(indices) + caption_rows, 8)
        assert tensor[0, :len(indices), 0].tolist() == indices
        assert tensor[0, len(indices):, 0].tolist() == list(range(caption_rows))


def test_slice_caption_conds_cut_a_region_tiles_crop_from_the_full_image_at_the_bbox(stubbed_slices):
    # Mask path: the tile indexes the bbox crop while the vision encodes read the FULL image
    # at the bbox origin, so its crop picture is the offset rect of that image and its canvas
    # rows are sliced in the full image's frame.
    clip = FakeCaptionClip()
    source = torch.arange(CANVAS_H * CANVAS_W * 3, dtype=torch.float32).reshape(1, CANVAS_H, CANVAS_W, 3)
    tile = Tile(Rect(0, 0, 32, 32))

    conds = captions.build_slice_caption_conds(clip, source, [tile], [["a fox"]], a_vision(canvas=1, crop=1),
                                               offset_x=96, offset_y=48)

    encoded_pixels = [call["image"] for call in clip.tokenize_calls
                      if call["text"] == vl.VISION_BLOCK]
    assert [tuple(pixels.shape) for pixels in encoded_pixels] == [(1, CANVAS_H, CANVAS_W, 3), (1, 32, 32, 3)]
    assert torch.equal(encoded_pixels[1], source[:, 48:80, 96:128, :])
    indices = list(range(N_ROWS + 2)) + _vision_rows(tile, 96, 48)
    assert conds[0][0][0][0, :len(indices), 0].tolist() == indices


def test_slice_caption_conds_encode_one_picture_at_a_time(stubbed_slices):
    # Core's tokenizer attaches images[0] alone, so the encode is always handed ONE picture.
    # Under refine_image's picture loop that is the whole batch, and nothing is concatenated.
    clip = FakeCaptionClip()
    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)

    conds = captions.build_slice_caption_conds(clip, source, [Tile(Rect(0, 0, CANVAS_W, CANVAS_H))],
                                               [["a fox"]], a_vision(canvas=1, crop=1))

    handed = [call["image"] for call in clip.tokenize_calls if call["image"] is not None]
    assert handed and all(tuple(image.shape) == (1, CANVAS_H, CANVAS_W, 3) for image in handed)
    tensor, extras = conds[0][0]
    assert tensor.shape[0] == 1
    assert extras["pooled_output"] is None


def test_slice_caption_conds_reject_a_caption_count_that_is_not_the_batch(stubbed_slices):
    # A row without its own caption would be conditioned on another row's picture.
    clip = FakeCaptionClip()
    source = torch.zeros(2, CANVAS_H, CANVAS_W, 3)

    with pytest.raises(RuntimeError, match="captioned a different number of times"):
        captions.build_slice_caption_conds(clip, source, [Tile(Rect(0, 0, CANVAS_W, CANVAS_H))],
                                           [["a fox"]], a_vision())


def test_slice_caption_conds_reject_a_vision_encoder_whose_layout_disagrees(stubbed_slices):
    # The vision half is vl._encode_one's own fail-fast, reached unchanged by this surface.
    clip = FakeCaptionClip(seq_override=1 + N_ROWS + 1 + TAIL + 99)

    with pytest.raises(RuntimeError, match="encoded conditioning has"):
        captions.build_slice_caption_conds(clip, torch.zeros(1, CANVAS_H, CANVAS_W, 3),
                                           [Tile(Rect(0, 0, CANVAS_W, CANVAS_H))], [["a fox"]], a_vision())


def test_slice_caption_conds_reject_a_caption_encode_of_the_wrong_length(stubbed_slices):
    # The concatenation's contract: a text-only encode of the caption must be exactly the
    # caption+tail rows the in-stream token layout says it is. Anything else means the
    # template strip changed, and cat'ing it would silently reshape every tile's positive.
    clip = FakeCaptionClip(text_seq_override=TAIL + 99)

    with pytest.raises(RuntimeError, match="text-only caption encode has"):
        captions.build_slice_caption_conds(clip, torch.zeros(1, CANVAS_H, CANVAS_W, 3),
                                           [Tile(Rect(0, 0, CANVAS_W, CANVAS_H))], [["a fox"]], a_vision())


def test_slice_caption_conds_reject_caption_extras_the_vision_encode_lacks(stubbed_slices):
    # The merged entry keeps the VISION encode's extras, so anything only the caption encode
    # carries would vanish without a trace.
    clip = FakeCaptionClip(text_extras={"guidance": torch.ones(1)})

    with pytest.raises(RuntimeError, match="caption encode carries conditioning extras"):
        captions.build_slice_caption_conds(clip, torch.zeros(1, CANVAS_H, CANVAS_W, 3),
                                           [Tile(Rect(0, 0, CANVAS_W, CANVAS_H))], [["a fox"]], a_vision())


def test_slice_caption_conds_reject_a_caption_encode_with_a_real_pooled_output(stubbed_slices):
    # Same argument, for the one extra that is always present: Krea 2 returns None here, and a
    # CLIP that returns a real vector is outside what this surface was settled on.
    clip = FakeCaptionClip(text_pooled=torch.zeros(1, 4))

    with pytest.raises(RuntimeError, match="caption encode has a real pooled_output"):
        captions.build_slice_caption_conds(clip, torch.zeros(1, CANVAS_H, CANVAS_W, 3),
                                           [Tile(Rect(0, 0, CANVAS_W, CANVAS_H))], [["a fox"]], a_vision())


# --- through the pipeline: the three-way branch, through the REAL dispatch -------------
# A vl_clip sends refine_image to the sync engine, so these run end to end through it (the
# fake model rides the real stepper): a refine_image mock would build no conditioning at all,
# which is the whole thing under test. 80x80 image at cap 56 / overlap 16 solves to a 2x2
# grid, so tiles really do select different rows of the 256x256 (8x8 merged cell) encode
# geometry pinned below.
PIPE_ENC = 256
PIPE_ROWS = (PIPE_ENC // vl.MERGED_CELL) ** 2


@pytest.fixture
def pipeline_clip(monkeypatch):
    monkeypatch.setattr(vl, "resample_picture", lambda source, budget: (source, PIPE_ENC, PIPE_ENC))
    return FakeCaptionClip(n_rows=PIPE_ROWS)


@pytest.fixture
def style_off(monkeypatch):
    # A preset may turn the whole-image style caption on, and the shipped (artwork) one does.
    # Tests that pin the caption surfaces' own per-tile shape run with it off whichever preset
    # they name, and the style tests below cover it on.
    resolve = captions.resolve_method
    monkeypatch.setattr(captions, "resolve_method",
                        lambda method: dataclasses.replace(resolve(method), style_instruction=""))


PIPE_PROMPT = "a fox in the centre"


def _run(image, guider, clip, vlm_method, mask=None, ctx=0, prompt=PIPE_PROMPT):
    # What the VL nodes hand the engine: the resolved preset with the prompt written into it
    # (node.py's own with_prompt call), since the shipped default asks for {PROMPT}. `prompt`
    # None leaves the resolve to the engine's dispatch, for the tests that pin its rejections.
    preset = None
    if prompt is not None:
        preset = captions.with_prompt(captions.resolve_method(vlm_method), prompt)
    return sampling.refine_image(
        image, guider, sync_sampler(), SIGMAS, *_engine(), max_tile_width=56,
        max_tile_height=56, context_anchor=ctx, context_overlap=16, mask=mask, vl_clip=clip,
        vlm_method=vlm_method, preset=preset,
    )


def _engine():
    from test_tiling import GridNoise, GridVAE
    return GridVAE(), GridNoise()


def test_vision_tokens_is_the_default_and_still_routes_through_build_global_slices(comfy_stubs, pipeline_clip, monkeypatch):
    # The byte-identical guarantee is structural: the default arm calls the unchanged
    # vl.build_global_slices and never touches the VLM at all.
    image = torch.rand(1, 80, 80, 3)
    seen = []
    real_build = vl.build_global_slices
    monkeypatch.setattr(vl, "build_global_slices", lambda *a, **k: (seen.append(k), real_build(*a, **k))[1])

    default = sampling.refine_image(image, VLGuider(), sync_sampler(), SIGMAS, *_engine(),
                                    max_tile_width=56, max_tile_height=56, context_anchor=0,
                                    context_overlap=16, vl_clip=pipeline_clip)
    explicit = _run(image, VLGuider(), pipeline_clip, "vision tokens")

    assert torch.equal(default, explicit)
    assert len(seen) == 2
    assert all(call == {"offset_x": 0, "offset_y": 0, "budget_tiles": None} for call in seen)
    assert pipeline_clip.generate_calls == []


def test_vision_tokens_reads_the_settings_file_once_and_never_the_vlm(comfy_stubs, pipeline_clip, monkeypatch):
    # "vision tokens" reads the file ONCE per picture, for its [vision] table, and never
    # reaches the VLM's generator. A broken file therefore fails this surface too, before
    # any encode.
    reads = []
    real_load = captions.load_settings
    monkeypatch.setattr(captions, "load_settings", lambda *a, **k: (reads.append(1), real_load(*a, **k))[1])

    out = _run(torch.rand(1, 80, 80, 3), VLGuider(), pipeline_clip, "vision tokens")

    assert out.shape == (1, 80, 80, 3)
    assert len(reads) == 1
    assert pipeline_clip.generate_calls == []


@pytest.mark.parametrize("method", ["vision tokens and captions", "captions",
                                    "vision tokens and captions (standard)"])
def test_each_caption_method_reaches_the_vlm_with_its_presets_wording(comfy_stubs, pipeline_clip, method):
    # End to end: the selected preset's own wording and budget reach every clip.generate, with
    # the node's prompt written into the default preset's {PROMPT}, and a preset with a style
    # instruction writes ONE whole-image caption before any tile. The unlabeled options are
    # what a workflow saved before the presets carries.
    preset = captions.with_prompt(captions.resolve_method(method), PIPE_PROMPT)
    image = torch.rand(1, 80, 80, 3)

    _run(image, VLGuider(), pipeline_clip, method)

    if preset.label == "prompted":
        assert PIPE_PROMPT in preset.tile_instruction
        assert "{PROMPT}" not in preset.tile_instruction

    style_on = bool(preset.style_instruction)
    assert len(pipeline_clip.generate_calls) == 4 + (1 if style_on else 0)
    tile_calls = pipeline_clip.generate_calls
    if style_on:
        style_call, *tile_calls = pipeline_clip.generate_calls
        assert style_call["text"] == preset.style_instruction
        assert style_call["max_length"] == preset.style_max_tokens
    assert {call["text"] for call in tile_calls} == {preset.tile_instruction}
    assert {call["max_length"] for call in tile_calls} == {preset.tile_max_tokens}
    # Every caption is generated with the reasoning turn on, which strip_thinking then cuts.
    asked = len(pipeline_clip.generate_calls)
    assert all(call["thinking"] is True for call in pipeline_clip.tokenize_calls[:asked])


def test_captions_only_gives_each_tile_a_text_only_positive(comfy_stubs, pipeline_clip, style_off):
    # The caption IS the tile's whole positive: no vision block, no grid rows.
    image = torch.rand(1, 80, 80, 3)
    guider = VLGuider()
    pristine = guider.original_conds

    _run(image, guider, pipeline_clip, "captions")

    assert len(guider.seen_conds) == 4
    assert pipeline_clip.encoded == ["a plain caption"] * 4     # one text encode per tile
    for seen in guider.seen_conds:
        assert seen["positive"][0]["cross_attn"].shape[1] == TAIL + 3
    assert guider.original_conds is pristine


def test_vision_and_captions_cats_the_shared_slice_and_a_text_only_caption(comfy_stubs, pipeline_clip, style_off):
    image = torch.rand(1, 80, 80, 3)
    guider = VLGuider()

    _run(image, guider, pipeline_clip, "vision tokens and captions")

    # ONE pure-vision canvas encode for the whole picture and one crop encode per tile, then
    # one text-only caption encode per tile — no VISION_BLOCK prefix on the caption encode.
    assert pipeline_clip.encoded == [vl.VISION_BLOCK] * 5 + ["a plain caption"] * 4
    caption_rows = TAIL + 3                              # tail + the caption's three words
    from test_tiling import _layout
    layout = _layout(80, 80, 56, 56, overlap=16)
    for tile, seen in zip(layout.tiles, guider.seen_conds, strict=True):
        # The shipped [vision] table: every crop cell (no tail), the canvas slice (no tail),
        # then the caption with the one tail.
        indices = list(range(PIPE_ROWS + 2)) + vl.slice_indices(tile.crop_rect, 80, 80, PIPE_ENC, PIPE_ENC, PIPE_ROWS + 2)
        rows = seen["positive"][0]["cross_attn"][0, :, 0].tolist()
        assert rows == indices + list(range(caption_rows))


def test_the_shipped_style_caption_rides_on_top_of_every_tile_caption(comfy_stubs, pipeline_clip):
    # What the DiT reads on a preset that asks for a style caption: every tile's caption is
    # encoded with the one style caption on top, newline-joined. (artwork) is the shipped
    # preset that turns it on, and the default (standard) leaves it off.
    method = "captions (artwork)"
    style_text = captions.resolve_method(method).style_instruction
    pipeline_clip.answer = lambda img, instruction: (
        "oil on canvas" if instruction == style_text else "a plain caption")

    _run(torch.rand(1, 80, 80, 3), VLGuider(), pipeline_clip, method)

    assert pipeline_clip.encoded == ["oil on canvas\na plain caption"] * 4


def test_the_mask_path_styles_from_the_full_image(comfy_stubs, pipeline_clip):
    # The tile captions read the region crop while the style caption reads the FULL image,
    # so a masked refine's style stays global. The run's first caption resample is the
    # style input, and its source is the whole 80x80 picture rather than the 64x64 bbox.
    image = torch.rand(1, 80, 80, 3)
    mask = torch.zeros(1, 80, 80)
    mask[:, 16:64, 16:64] = 1.0

    _run(image, VLGuider(), pipeline_clip, "captions (artwork)", mask=mask, ctx=8)

    style_shape = comfy_stubs["common_upscale_calls"][0][0]
    assert style_shape == (1, 3, 80, 80)


def test_the_mask_path_captions_the_region_crop_and_encodes_the_full_image(comfy_stubs, pipeline_clip, style_off):
    # Two different canvases, deliberately: the caption describes the tile the sampler
    # actually runs (the bbox crop), while the vision encode still reads the whole image at
    # the bbox origin so the region stays globally informed.
    image = torch.rand(1, 80, 80, 3)
    mask = torch.zeros(1, 80, 80)
    mask[:, 16:64, 16:64] = 1.0

    _run(image, VLGuider(), pipeline_clip, "vision tokens and captions", mask=mask, ctx=8)

    y0, y1, x0, x1 = sampling._expand_snap_clamp(sampling._mask_bbox(mask >= 0.5), 8, 80, 80)
    assert (y0, y1, x0, x1) == (8, 72, 8, 72)
    # The text-only caption encode tokenizes with no image at all, so only the calls that were
    # handed pixels are read here. VISION_BLOCK alone is the vision encode's own text (the
    # caption tail probe appends the caption to it).
    with_image = [call for call in pipeline_clip.tokenize_calls if call["image"] is not None]
    caption_inputs = [call["image"] for call in with_image if call["llama_template"] is None]
    encode_inputs = [call["image"] for call in with_image if call["text"] == vl.VISION_BLOCK]
    # comfy_stubs' common_upscale returns the resampled COPY, so only the shape is readable —
    # which is the point: what the VLM reads is never the sampled tile.
    assert caption_inputs and all(tuple(x.shape[1:3]) != (y1 - y0, x1 - x0) for x in caption_inputs)
    # The canvas encode reads the FULL image, and every crop encode is the tile's rect cut
    # from that image at the bbox origin, so the region is read in its place in the whole
    # picture rather than on its own.
    sx = grid.solve_axis(x1 - x0, 56, 8, 16, axis="width")
    sy = grid.solve_axis(y1 - y0, 56, 8, 16, axis="height")
    tiles = grid.build_layout(x1 - x0, y1 - y0, sx, sy, 8, 16).tiles
    assert len(tiles) > 1
    assert torch.equal(encode_inputs[0], image)
    assert len(encode_inputs) == 1 + len(tiles)
    for pixels, tile in zip(encode_inputs[1:], tiles, strict=True):
        crop = tile.crop_rect
        expected = image[:, max(0, crop.y0 + y0):min(80, crop.y1 + y0), max(0, crop.x0 + x0):min(80, crop.x1 + x0), :]
        assert torch.equal(pixels, expected)


@pytest.mark.parametrize("method", ["captions", "vision tokens and captions"])
def test_a_two_picture_batch_with_different_length_captions_completes(comfy_stubs, pipeline_clip, style_off, method):
    # The case that used to raise: every picture is captioned from its OWN pixels, two
    # captions rarely tokenize to the same length, and one shared cond could not concatenate
    # them. Refining one picture at a time removes the concatenation entirely.
    image = torch.rand(2, 80, 80, 3)
    pipeline_clip.answer = lambda img, instruction: (
        "a fox" if len(pipeline_clip.generate_calls) <= 4 else "a cart in a very busy market")

    out = _run(image, VLGuider(), pipeline_clip, method)

    assert out.shape == (2, 80, 80, 3)
    assert len(pipeline_clip.generate_calls) == 8            # 4 tiles x 2 pictures
    # Both captions reach the encoder as plain text on BOTH surfaces; the slice surface adds
    # its one shared pure-vision canvas encode per picture on top.
    expected = {"a fox", "a cart in a very busy market"}
    if method == "vision tokens and captions":
        expected.add(vl.VISION_BLOCK)
    assert set(pipeline_clip.encoded) == expected


def test_an_unknown_vlm_method_is_rejected_by_name(comfy_stubs, pipeline_clip):
    with pytest.raises(ValueError, match="names no conditioning surface"):
        _run(torch.rand(1, 80, 80, 3), VLGuider(), pipeline_clip, "vision", prompt=None)


def test_a_method_naming_an_absent_preset_is_rejected_through_the_dispatch(comfy_stubs, pipeline_clip):
    # The selector is built at startup and the wording read per run, so a preset deleted
    # mid-session leaves a stale option that must fail by name rather than obscurely.
    with pytest.raises(RuntimeError, match="asks for preset 'gone'"):
        _run(torch.rand(1, 80, 80, 3), VLGuider(), pipeline_clip, "captions (gone)", prompt=None)


def test_slice_caption_conds_forward_budget_tiles_to_the_vision_rows(stubbed_slices, monkeypatch):
    # The vision half of this surface is vl.build_vision_rows itself, so a block run's canvas
    # sample must be sized off the full grid's tiles here exactly as on the vision-only
    # surface.
    seen = []
    real_budget = vl.canvas_budget_pixels
    monkeypatch.setattr(vl, "canvas_budget_pixels",
                        lambda tiles, h, w, tokens: (seen.append(tiles),
                                                     real_budget(tiles, h, w, tokens))[1])
    full = layout_tiles(3, 1)
    block = full[:1]

    captions.build_slice_caption_conds(FakeCaptionClip(), torch.zeros(1, CANVAS_H, CANVAS_W, 3),
                                       block, [["a fox"]], a_vision(canvas=1, crop=0),
                                       budget_tiles=full)

    assert seen == [full]
