"""ContextAnchoredTileUpscaleVLVideo's wiring, plus the two upscale.py builders it adds.

The node's job is to build the sampling objects from widgets, run the conditioning encode
and the whole-clip upscale IN THAT ORDER, and hand everything to video.refine_video. So the
collaborators are recorders: what is pinned here is which value reaches which parameter and
in which order, not any pixel math (test_video / test_vl_video / test_upscale cover the real
functions). Everything runs on the comfy stubs — no ComfyUI install, no GPU.
"""
import inspect

import pytest
import torch

from context_anchored_tile_refine import sampling, upscale, video, vl_video
from context_anchored_tile_refine.node import ContextAnchoredTileUpscaleVLVideo

REQUIRED_ORDER = [
    "image",
    "model",
    "clip",
    "vae",
    "seed",
    "sampler_name",
    "scheduler",
    "steps",
    "denoise",
    "upscale_by",
    "max_tile_width",
    "max_tile_height",
    "context_anchor",
    "context_overlap",
]

TYPE_BY_NAME = {
    "image": "IMAGE",
    "model": "MODEL",
    "clip": "CLIP",
    "vae": "VAE",
    "seed": "INT",
    "steps": "INT",
    "denoise": "FLOAT",
    "upscale_by": "FLOAT",
    "max_tile_width": "INT",
    "max_tile_height": "INT",
    "context_anchor": "INT",
    "context_overlap": "INT",
    "upscale_model": "UPSCALE_MODEL",
    "av_latent": "LATENT",
}

# Step 32 on all four geometry INTs: every one of them feeds a crop dimension, and H3's
# crops must be multiples of 32 (VAE 16 x DiT patch 2).
WIDGET_OPTIONS = {
    "seed": {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True},
    "steps": {"default": 8, "min": 1, "max": 10000},
    "denoise": {"default": 0.22, "min": 0.0, "max": 1.0, "step": 0.01},
    "upscale_by": {"default": 2.0, "min": 0.01, "max": 8.0, "step": 0.01},
    "max_tile_width": {"default": 1408, "min": 256, "max": 16384, "step": 32},
    "max_tile_height": {"default": 1408, "min": 256, "max": 16384, "step": 32},
    "context_anchor": {"default": 32, "min": 0, "max": 512, "step": 32},
    "context_overlap": {"default": 32, "min": 0, "max": 512, "step": 32},
}

WIDGETS = {
    "seed": 1234,
    "sampler_name": "res_multistep",
    "scheduler": "simple",
    "steps": 8,
    "denoise": 0.22,
    "upscale_by": 2.0,
    "max_tile_width": 1408,
    "max_tile_height": 1408,
    "context_anchor": 32,
    "context_overlap": 32,
}


class NestedPair:
    """The unbind() surface an H3 AV LATENT's samples present (comfy/nested_tensor.py)."""

    def __init__(self, *streams):
        self.streams = streams

    def unbind(self):
        return self.streams


class FakeGuider:
    def __init__(self, model, positive):
        self.model = model
        self.original_conds = {"positive": positive}


def _drive(monkeypatch, image=None, upscale_model=None, av_latent=None, upscaled=None,
           real_sigmas=False, real_refine=False, **overrides):
    """Run refine() with every collaborator recorded; return (recorded, result)."""
    recorded = {
        "order": [],
        "encode_empty_calls": [],
        "sample_conditioning_frames": None,
        "prepare_upscaled": None,
        "encode_global": None,
        "prepare_upscaled_video": None,
        "build_basic_guider": None,
        "build_sigmas": None,
        "refine_video": None,
        # 40x40 picks: NOT on the /32 grid, so the node's reflect pad is observable at the
        # encoder, and small enough that the pad stays inside reflect's pad < dim rule.
        "picks": torch.rand(2, 40, 40, 3),
        "picks_upscaled": torch.rand(2, 40, 40, 3),
        "stamps": [0.0, 0.5],
        "upscaled": torch.rand(5, 128, 128, 3) if upscaled is None else upscaled,
        "sigmas": torch.linspace(1.0, 0.0, 9),
        "refined": torch.rand(5, 128, 128, 3),
    }
    rows = 3
    recorded["pack"] = vl_video.GlobalEncode(
        torch.arange(rows, dtype=torch.float32).reshape(1, rows, 1),
        torch.arange(rows), [("text", 0), ("grid", 1)], 128, 128, {})

    def fake_sample_conditioning_frames(frames):
        recorded["sample_conditioning_frames"] = frames
        return recorded["picks"], recorded["stamps"]

    def fake_prepare_upscaled(image, upscale_model, upscale_by):
        recorded["prepare_upscaled"] = {"image": image, "upscale_model": upscale_model, "upscale_by": upscale_by}
        return recorded["picks_upscaled"]

    def fake_encode_global(clip, picks, stamps):
        recorded["order"].append("encode_global")
        recorded["encode_global"] = {"clip": clip, "picks": picks, "stamps": stamps}
        return recorded["pack"]

    def fake_prepare_upscaled_video(frames, upscale_model, upscale_by):
        recorded["order"].append("prepare_upscaled_video")
        recorded["prepare_upscaled_video"] = {"frames": frames, "upscale_model": upscale_model, "upscale_by": upscale_by}
        return recorded["upscaled"]

    def fake_encode_empty(clip):
        recorded["encode_empty_calls"].append(clip)
        return [("empty", {})]

    def fake_build_basic_guider(model, positive):
        recorded["build_basic_guider"] = {"model": model, "positive": positive}
        return FakeGuider(model, positive)

    def fake_build_sigmas(model, scheduler, steps, denoise):
        recorded["build_sigmas"] = {"model": model, "scheduler": scheduler, "steps": steps, "denoise": denoise}
        return recorded["sigmas"]

    def fake_refine_video(frames, guider, sampler, sigmas, vae, seed, max_tile_width,
                          max_tile_height, context_anchor, context_overlap, vl_pack,
                          audio_latent=None):
        recorded["refine_video"] = {
            "frames": frames,
            "guider": guider,
            "sampler": sampler,
            "sigmas": sigmas,
            "vae": vae,
            "seed": seed,
            "max_tile_width": max_tile_width,
            "max_tile_height": max_tile_height,
            "context_anchor": context_anchor,
            "context_overlap": context_overlap,
            "vl_pack": vl_pack,
            "audio_latent": audio_latent,
        }
        return recorded["refined"]

    monkeypatch.setattr(vl_video, "sample_conditioning_frames", fake_sample_conditioning_frames)
    monkeypatch.setattr(vl_video, "encode_global", fake_encode_global)
    monkeypatch.setattr(upscale, "prepare_upscaled", fake_prepare_upscaled)
    monkeypatch.setattr(upscale, "prepare_upscaled_video", fake_prepare_upscaled_video)
    monkeypatch.setattr(upscale, "encode_empty", fake_encode_empty)
    monkeypatch.setattr(upscale, "build_basic_guider", fake_build_basic_guider)
    if not real_sigmas:
        monkeypatch.setattr(upscale, "build_sigmas", fake_build_sigmas)
    if not real_refine:
        monkeypatch.setattr(video, "refine_video", fake_refine_video)

    recorded["image"] = torch.rand(5, 64, 64, 3) if image is None else image
    recorded["model"] = object()
    recorded["clip"] = object()
    recorded["vae"] = object()

    widgets = dict(WIDGETS)
    widgets.update(overrides)
    result = ContextAnchoredTileUpscaleVLVideo().refine(
        image=recorded["image"],
        model=recorded["model"],
        clip=recorded["clip"],
        vae=recorded["vae"],
        upscale_model=upscale_model,
        av_latent=av_latent,
        **widgets,
    )
    return recorded, result


# --- input schema ---------------------------------------------------------------------

def test_required_order_is_pinned(comfy_stubs):
    assert list(ContextAnchoredTileUpscaleVLVideo.INPUT_TYPES()["required"]) == REQUIRED_ORDER


def test_optional_is_upscale_model_and_av_latent(comfy_stubs):
    assert list(ContextAnchoredTileUpscaleVLVideo.INPUT_TYPES()["optional"]) == ["upscale_model", "av_latent"]


def test_input_type_strings(comfy_stubs):
    input_types = ContextAnchoredTileUpscaleVLVideo.INPUT_TYPES()
    all_inputs = {**input_types["required"], **input_types["optional"]}
    for name, type_string in TYPE_BY_NAME.items():
        assert all_inputs[name][0] == type_string, name


def test_widget_options_exact(comfy_stubs):
    required = ContextAnchoredTileUpscaleVLVideo.INPUT_TYPES()["required"]
    for name, expected in WIDGET_OPTIONS.items():
        options = required[name][1]
        for key, value in expected.items():
            assert options[key] == value, "{}[{}]".format(name, key)


def test_combo_inputs_come_from_comfy_samplers(comfy_stubs):
    import comfy.samplers

    required = ContextAnchoredTileUpscaleVLVideo.INPUT_TYPES()["required"]
    assert required["sampler_name"][0] is comfy.samplers.KSampler.SAMPLERS
    assert required["scheduler"][0] is comfy.samplers.KSampler.SCHEDULERS
    # A default outside its own list is an invalid widget the UI silently re-points.
    assert required["sampler_name"][1]["default"] == "res_multistep"
    assert required["sampler_name"][1]["default"] in required["sampler_name"][0]
    assert required["scheduler"][1]["default"] == "simple"
    assert required["scheduler"][1]["default"] in required["scheduler"][0]


def test_has_no_prompt_mask_cfg_or_negative_inputs(comfy_stubs):
    # All four absences are design decisions: H3 is CFG-free (no cfg, no negative), the mask
    # path is image-only, and any positive text re-admits phantom objects.
    input_types = ContextAnchoredTileUpscaleVLVideo.INPUT_TYPES()
    all_inputs = {**input_types["required"], **input_types["optional"]}
    for absent in ("mask", "cfg", "negative", "positive", "text", "prompt", "noise", "guider"):
        assert absent not in all_inputs, absent


def test_every_input_has_a_tooltip(comfy_stubs):
    input_types = ContextAnchoredTileUpscaleVLVideo.INPUT_TYPES()
    all_inputs = {**input_types["required"], **input_types["optional"]}
    for name, definition in all_inputs.items():
        tooltip = definition[1].get("tooltip")
        assert isinstance(tooltip, str) and tooltip, name


def test_the_video_contract_tooltips_state_what_the_inputs_actually_are(comfy_stubs):
    # image: a batch here is ONE clip's frames, not independent images. av_latent: the
    # unconnected fallback is silently substituted, so the tooltip must say so rather than
    # let a user assume the audio context was theirs.
    input_types = ContextAnchoredTileUpscaleVLVideo.INPUT_TYPES()
    image_tooltip = input_types["required"]["image"][1]["tooltip"]
    av_tooltip = input_types["optional"]["av_latent"][1]["tooltip"]

    assert "24 fps" in image_tooltip and "ONE video" in image_tooltip
    assert "Unconnected" in av_tooltip and "zero audio latent" in av_tooltip
    assert "recommended" in av_tooltip


def test_node_class_attributes(comfy_stubs):
    assert ContextAnchoredTileUpscaleVLVideo.RETURN_TYPES == ("IMAGE",)
    assert ContextAnchoredTileUpscaleVLVideo.FUNCTION == "refine"
    assert callable(ContextAnchoredTileUpscaleVLVideo.refine)
    assert ContextAnchoredTileUpscaleVLVideo.CATEGORY == "image/upscaling"


def test_refine_signature_matches_input_names(comfy_stubs):
    # ComfyUI calls FUNCTION with the input names as keywords: a rename on either side is a
    # TypeError at execution time, which this catches at test time instead.
    input_types = ContextAnchoredTileUpscaleVLVideo.INPUT_TYPES()
    parameters = list(inspect.signature(ContextAnchoredTileUpscaleVLVideo.refine).parameters)
    assert parameters == ["self"] + list(input_types["required"]) + list(input_types["optional"])


# --- VALIDATE_INPUTS ------------------------------------------------------------------

def test_validate_accepts_32_aligned_geometry():
    assert ContextAnchoredTileUpscaleVLVideo.VALIDATE_INPUTS(
        max_tile_width=1408, max_tile_height=1408, context_anchor=32, context_overlap=64) is True
    # 0 rings are legal (hard seams, no anchor); unset widgets are not this node's business.
    assert ContextAnchoredTileUpscaleVLVideo.VALIDATE_INPUTS(
        max_tile_width=256, max_tile_height=16384, context_anchor=0, context_overlap=0) is True
    assert ContextAnchoredTileUpscaleVLVideo.VALIDATE_INPUTS() is True


@pytest.mark.parametrize("kwargs, expected", [
    ({"max_tile_width": 1400}, "max_tile_width must be a multiple of 32"),
    # The image node's /8 grid is NOT enough here: 1416 is a legal image-node value.
    ({"max_tile_height": 1416}, "max_tile_height must be a multiple of 32"),
    ({"context_anchor": 8}, "context_anchor must be a multiple of 32"),
    ({"context_overlap": 40}, "context_overlap must be a multiple of 32"),
    ({"max_tile_width": 224}, "max_tile_width must be between 256 and 16384"),
    ({"max_tile_height": 16416}, "max_tile_height must be between 256 and 16384"),
    ({"context_anchor": 544}, "context_anchor must be between 0 and 512"),
    ({"context_overlap": -32}, "context_overlap must be between 0 and 512"),
])
def test_validate_rejects_off_grid_and_out_of_range(kwargs, expected):
    message = ContextAnchoredTileUpscaleVLVideo.VALIDATE_INPUTS(**kwargs)
    assert isinstance(message, str) and message.startswith(expected)


# --- refine() plumbing ----------------------------------------------------------------

def test_returns_a_one_tuple_of_refine_videos_result(comfy_stubs, monkeypatch):
    recorded, result = _drive(monkeypatch)

    assert isinstance(result, tuple) and len(result) == 1
    assert result[0] is recorded["refined"]


def test_the_encode_happens_before_the_full_clip_canvases_are_allocated(comfy_stubs, monkeypatch):
    # The RAM-safety ordering, and the whole reason the node owns the encode instead of
    # video.py: the 32B text encoder must be gone before the full-clip canvases exist.
    recorded, _ = _drive(monkeypatch)

    assert recorded["order"] == ["encode_global", "prepare_upscaled_video"]


def test_the_conditioning_picks_come_from_the_input_frames(comfy_stubs, monkeypatch):
    recorded, _ = _drive(monkeypatch)

    assert recorded["sample_conditioning_frames"] is recorded["image"]
    assert recorded["encode_global"]["clip"] is recorded["clip"]
    assert recorded["encode_global"]["stamps"] is recorded["stamps"]


def test_the_picks_get_the_same_upscale_pipeline_as_the_canvas(comfy_stubs, monkeypatch):
    # Every stage is per-frame, so upscaling the picks alone gives the encoder exactly the
    # content the canvas will hold — the identity the stage ordering depends on.
    upscale_model = object()
    recorded, _ = _drive(monkeypatch, upscale_model=upscale_model)

    assert recorded["prepare_upscaled"]["image"] is recorded["picks"]
    assert recorded["prepare_upscaled"]["upscale_model"] is upscale_model
    assert recorded["prepare_upscaled"]["upscale_by"] == 2.0
    assert recorded["prepare_upscaled_video"]["upscale_model"] is upscale_model
    assert recorded["prepare_upscaled_video"]["upscale_by"] == 2.0


def test_the_picks_reach_the_encoder_padded_onto_the_32_grid(comfy_stubs, monkeypatch):
    # Same reflect pad the canvas gets, so the encode's cell grid maps onto the padded
    # canvas the tile rects are cut from.
    recorded, _ = _drive(monkeypatch)

    assert recorded["picks_upscaled"].shape == (2, 40, 40, 3)
    assert tuple(recorded["encode_global"]["picks"].shape) == (2, 64, 64, 3)


def test_refine_video_receives_the_upscaled_clip_not_the_input(comfy_stubs, monkeypatch):
    recorded, _ = _drive(monkeypatch)

    assert recorded["prepare_upscaled_video"]["frames"] is recorded["image"]
    assert recorded["refine_video"]["frames"] is recorded["upscaled"]


def test_the_vl_pack_is_handed_to_refine_video_already_encoded(comfy_stubs, monkeypatch):
    recorded, _ = _drive(monkeypatch)

    assert recorded["refine_video"]["vl_pack"] is recorded["pack"]


def test_the_placeholder_positive_is_the_whole_clip_encode_not_a_second_clip_load(comfy_stubs, monkeypatch):
    recorded, _ = _drive(monkeypatch)
    positive = recorded["build_basic_guider"]["positive"]

    assert len(positive) == 1
    assert positive[0][0] is recorded["pack"].cond
    assert list(positive[0][1]) == ["minimax_token_tags"]
    assert positive[0][1]["minimax_token_tags"] is recorded["pack"].tags
    # No empty encode: loading a second CLIP for a placeholder every tile overwrites anyway
    # is exactly the residency this node's stage order exists to avoid.
    assert recorded["encode_empty_calls"] == []


def test_builders_receive_the_model_and_their_widgets(comfy_stubs, monkeypatch):
    recorded, _ = _drive(monkeypatch)

    assert recorded["build_basic_guider"]["model"] is recorded["model"]
    assert recorded["build_sigmas"] == {"model": recorded["model"], "scheduler": "simple", "steps": 8, "denoise": 0.22}
    assert recorded["refine_video"]["guider"].model is recorded["model"]
    assert recorded["refine_video"]["sigmas"] is recorded["sigmas"]


def test_sampler_is_built_from_the_widget_name(comfy_stubs, monkeypatch):
    recorded, _ = _drive(monkeypatch, sampler_name="euler")

    assert comfy_stubs["sampler_object_calls"] == ["euler"]
    assert recorded["refine_video"]["sampler"] == ("sampler_object", "euler")


def test_geometry_and_seed_pass_through_unchanged(comfy_stubs, monkeypatch):
    recorded, _ = _drive(monkeypatch, seed=4242, max_tile_width=1280, max_tile_height=896,
                         context_anchor=64, context_overlap=96)
    call = recorded["refine_video"]

    assert call["vae"] is recorded["vae"]
    assert call["seed"] == 4242
    assert call["max_tile_width"] == 1280
    assert call["max_tile_height"] == 896
    assert call["context_anchor"] == 64
    assert call["context_overlap"] == 96


def test_denoise_zero_returns_the_upscaled_clip_untouched(comfy_stubs, monkeypatch):
    # The real build_sigmas and the real refine_video: denoise 0 must travel all the way to
    # the empty-sigmas early return, so "upscale only" costs no VAE roundtrip.
    recorded, result = _drive(monkeypatch, denoise=0.0, real_sigmas=True, real_refine=True)

    assert torch.equal(result[0], recorded["upscaled"])
    assert result[0] is not recorded["upscaled"]


# --- av_latent ------------------------------------------------------------------------

def test_unconnected_av_latent_reaches_refine_video_as_none(comfy_stubs, monkeypatch):
    recorded, _ = _drive(monkeypatch)

    assert recorded["refine_video"]["audio_latent"] is None


def test_connected_av_latent_hands_over_its_audio_stream_only(comfy_stubs, monkeypatch):
    audio = torch.rand(1, 32, 2, 8)
    latent = {"samples": NestedPair(torch.rand(1, 24, 2, 4, 4), audio)}
    recorded, _ = _drive(monkeypatch, av_latent=latent)
    passed = recorded["refine_video"]["audio_latent"]

    assert torch.equal(passed, audio)
    assert passed.device.type == "cpu"


def test_rejects_a_latent_whose_samples_are_not_a_nested_pair(comfy_stubs, monkeypatch):
    with pytest.raises(ValueError, match="must be the MiniMax H3 sampler's AV LATENT"):
        _drive(monkeypatch, av_latent={"samples": object()})


def test_rejects_a_latent_with_the_wrong_stream_count(comfy_stubs, monkeypatch):
    latent = {"samples": NestedPair(torch.rand(1, 24, 2, 4, 4))}
    with pytest.raises(ValueError, match=r"holds 1 latent stream\(s\)"):
        _drive(monkeypatch, av_latent=latent)


# --- input validation -----------------------------------------------------------------

def test_rejects_a_non_4d_image(comfy_stubs, monkeypatch):
    with pytest.raises(ValueError, match="B,H,W,C"):
        _drive(monkeypatch, image=torch.rand(96, 104, 3))


def test_an_upscale_result_below_32px_is_rejected_naming_upscale_by(comfy_stubs, monkeypatch):
    # Rejected from the TARGET size, before a single frame is resized: below 32px an axis
    # cannot carry one /32 crop, and the reflect pad would raise naming neither the node nor
    # the widget that caused it.
    with pytest.raises(ValueError, match=r"upscale_by 0\.1 takes the 64x64 input to 6x6"):
        _drive(monkeypatch, upscale_by=0.1)


# --- upscale.build_basic_guider -------------------------------------------------------

def test_build_basic_guider_keeps_the_positive_in_original_conds(comfy_stubs):
    import comfy.samplers

    model = object()
    positive = [("positive", {})]
    guider = upscale.build_basic_guider(model, positive)

    assert isinstance(guider, comfy.samplers.CFGGuider)
    assert guider.model_patcher is model
    # The swap seam video.py keys on, and no negative at all: H3 is CFG-free.
    assert guider.original_conds == {"positive": positive}


# --- upscale.prepare_upscaled_video ---------------------------------------------------

def test_prepare_upscaled_video_returns_the_input_when_nothing_would_change(comfy_stubs):
    frames = torch.rand(9, 64, 64, 3)

    assert upscale.prepare_upscaled_video(frames, None, 1.0) is frames
    # A same-size lanczos is an 8-bit round trip, so "no change" must mean no call at all.
    assert comfy_stubs["common_upscale_calls"] == []


def test_prepare_upscaled_video_resizes_in_frame_chunks_into_an_fp16_canvas(comfy_stubs):
    frames = torch.rand(9, 64, 48, 3)
    out = upscale.prepare_upscaled_video(frames, None, 2.0)

    assert out.shape == (9, 128, 96, 3)
    assert out.dtype == torch.float16
    # 9 frames at UPSCALE_FRAME_CHUNK 8 = two passes, each onto the full target size.
    batches = [call[0][0] for call in comfy_stubs["common_upscale_calls"]]
    assert batches == [8, 1]
    assert all(call[1:] == (96, 128, "lanczos", "disabled") for call in comfy_stubs["common_upscale_calls"])
