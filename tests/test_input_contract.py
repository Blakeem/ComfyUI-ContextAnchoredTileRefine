from context_anchored_tile_refine.node import (
    ContextAnchoredTileRefine,
    ContextAnchoredTileRefineVL,
    ContextAnchoredTileUpscaleVL,
)

REQUIRED_ORDER = [
    "image",
    "guider",
    "sampler",
    "sigmas",
    "vae",
    "noise",
    "max_tile_width",
    "max_tile_height",
    "context_anchor",
    "context_overlap",
]

TYPE_BY_NAME = {
    "image": "IMAGE",
    "guider": "GUIDER",
    "sampler": "SAMPLER",
    "sigmas": "SIGMAS",
    "vae": "VAE",
    "noise": "NOISE",
    "max_tile_width": "INT",
    "max_tile_height": "INT",
    "context_anchor": "INT",
    "context_overlap": "INT",
    "mask": "MASK",
}

INT_WIDGET_OPTIONS = {
    "max_tile_width": {"default": 1024, "min": 256, "max": 16384, "step": 8},
    "max_tile_height": {"default": 1024, "min": 256, "max": 16384, "step": 8},
    # 32/32 is the A/B-settled compromise: invisible on detailed character scenes without
    # paying for a wide halo. Smooth-gradient scenes are handled by raising context_overlap.
    "context_anchor": {"default": 32, "min": 0, "max": 512, "step": 8},
    "context_overlap": {"default": 32, "min": 0, "max": 512, "step": 8},
}

FLOAT_WIDGET_OPTIONS = {
}


def test_required_order_is_pinned():
    input_types = ContextAnchoredTileRefine.INPUT_TYPES()
    assert list(input_types["required"]) == REQUIRED_ORDER


def test_optional_is_mask_only():
    input_types = ContextAnchoredTileRefine.INPUT_TYPES()
    assert list(input_types["optional"]) == ["mask"]


def test_input_type_strings():
    input_types = ContextAnchoredTileRefine.INPUT_TYPES()
    all_inputs = {**input_types["required"], **input_types["optional"]}
    for name, type_string in TYPE_BY_NAME.items():
        assert all_inputs[name][0] == type_string, name


def test_int_widget_options_exact():
    required = ContextAnchoredTileRefine.INPUT_TYPES()["required"]
    for name, expected in INT_WIDGET_OPTIONS.items():
        options = required[name][1]
        for key, value in expected.items():
            assert options[key] == value, "{}[{}]".format(name, key)


def test_float_widget_options_exact():
    required = ContextAnchoredTileRefine.INPUT_TYPES()["required"]
    for name, expected in FLOAT_WIDGET_OPTIONS.items():
        options = required[name][1]
        for key, value in expected.items():
            assert options[key] == value, "{}[{}]".format(name, key)


def test_seam_behaviour_is_not_exposed_as_widgets():
    # Everything about HOW a seam is hidden is baked in: the feather curve, the minimum-error
    # routing, and the blend width. Only geometry that genuinely depends on the image stays
    # tunable. Each name below was a real widget during A/B testing and was removed once the
    # comparison settled, so re-adding one would reopen a closed decision by accident.
    input_types = ContextAnchoredTileRefine.INPUT_TYPES()
    all_inputs = {**input_types["required"], **input_types["optional"]}
    for retired in ("feather_plateau", "feather_falloff", "seam_mode",
                    "warp_amount", "warp_scale", "seam_blend", "cut_width"):
        assert retired not in all_inputs, retired


def test_every_input_has_a_tooltip():
    input_types = ContextAnchoredTileRefine.INPUT_TYPES()
    all_inputs = {**input_types["required"], **input_types["optional"]}
    for name, definition in all_inputs.items():
        tooltip = definition[1].get("tooltip")
        assert isinstance(tooltip, str) and tooltip, name


def test_node_class_attributes():
    assert ContextAnchoredTileRefine.RETURN_TYPES == ("IMAGE",)
    assert ContextAnchoredTileRefine.FUNCTION == "refine"
    assert callable(ContextAnchoredTileRefine.refine)
    assert ContextAnchoredTileRefine.CATEGORY == "sampling/custom_sampling"
    assert not hasattr(ContextAnchoredTileRefine, "IS_CHANGED")


def test_validate_inputs_accepts_defaults():
    assert (
        ContextAnchoredTileRefine.VALIDATE_INPUTS(
            max_tile_width=1024, max_tile_height=1024, context_anchor=32, context_overlap=32
        )
        is True
    )


def test_validate_inputs_accepts_zero_context_overlap():
    # context_overlap=0 disables the directional feather (hard seams) — allowed for A/B testing.
    assert ContextAnchoredTileRefine.VALIDATE_INPUTS(context_overlap=0) is True


def test_validate_inputs_rejects_non_multiple_of_8():
    result = ContextAnchoredTileRefine.VALIDATE_INPUTS(context_overlap=100)
    assert isinstance(result, str)
    assert "context_overlap" in result


def test_validate_inputs_rejects_below_min():
    result = ContextAnchoredTileRefine.VALIDATE_INPUTS(max_tile_width=128)
    assert isinstance(result, str)
    assert "max_tile_width" in result


def test_vl_required_order_is_base_plus_clip():
    input_types = ContextAnchoredTileRefineVL.INPUT_TYPES()
    assert list(input_types["required"]) == REQUIRED_ORDER + ["clip"]
    assert input_types["required"]["clip"][0] == "CLIP"


def test_vl_optional_is_mask_only():
    # The masked VL refine encodes the whole image and offsets the region's tiles
    # into its frame (vl.py slice_indices offsets), so the mask input is supported.
    input_types = ContextAnchoredTileRefineVL.INPUT_TYPES()
    assert list(input_types["optional"]) == ["mask"]


def test_vl_every_input_has_a_tooltip():
    input_types = ContextAnchoredTileRefineVL.INPUT_TYPES()
    all_inputs = {**input_types["required"], **input_types["optional"]}
    for name, definition in all_inputs.items():
        tooltip = definition[1].get("tooltip")
        assert isinstance(tooltip, str) and tooltip, name


def test_vl_node_class_attributes():
    assert ContextAnchoredTileRefineVL.RETURN_TYPES == ("IMAGE",)
    assert ContextAnchoredTileRefineVL.FUNCTION == "refine"
    assert callable(ContextAnchoredTileRefineVL.refine)
    assert ContextAnchoredTileRefineVL.CATEGORY == "sampling/custom_sampling"


def test_vl_input_types_does_not_leak_into_base():
    # The subclass edits the dict the base INPUT_TYPES call returned; a cached/shared
    # dict would silently grow 'clip' and lose 'mask' on the base node.
    ContextAnchoredTileRefineVL.INPUT_TYPES()
    base = ContextAnchoredTileRefine.INPUT_TYPES()
    assert list(base["required"]) == REQUIRED_ORDER
    assert list(base["optional"]) == ["mask"]


# --- the all-in-one upscale node ----------------------------------------------------
# Its INPUT_TYPES reads comfy.samplers for the two combo lists, so every test below takes
# comfy_stubs rather than relying on a real install.

UPSCALE_REQUIRED_ORDER = [
    "image",
    "model",
    "clip",
    "vae",
    "seed",
    "sampler_name",
    "scheduler",
    "steps",
    "cfg",
    "denoise",
    "upscale_by",
    "max_tile_width",
    "max_tile_height",
    "context_anchor",
    "context_overlap",
]

UPSCALE_TYPE_BY_NAME = {
    "image": "IMAGE",
    "model": "MODEL",
    "clip": "CLIP",
    "vae": "VAE",
    "seed": "INT",
    "steps": "INT",
    "cfg": "FLOAT",
    "denoise": "FLOAT",
    "upscale_by": "FLOAT",
    "max_tile_width": "INT",
    "max_tile_height": "INT",
    "context_anchor": "INT",
    "context_overlap": "INT",
    "upscale_model": "UPSCALE_MODEL",
    "negative": "CONDITIONING",
}

UPSCALE_WIDGET_OPTIONS = {
    "seed": {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True},
    "steps": {"default": 20, "min": 1, "max": 10000},
    # 3.5 is the Krea 2 working default; the base node inherits cfg from its GUIDER input.
    "cfg": {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01},
    "denoise": {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "upscale_by": {"default": 2.0, "min": 0.01, "max": 8.0, "step": 0.01},
    # Larger than the base node's 1024/1024: this node owns the whole run, and the tall
    # default suits the portrait crops the VL path was settled on.
    "max_tile_width": {"default": 1536, "min": 256, "max": 16384, "step": 8},
    "max_tile_height": {"default": 2048, "min": 256, "max": 16384, "step": 8},
    "context_anchor": {"default": 32, "min": 0, "max": 512, "step": 8},
    "context_overlap": {"default": 32, "min": 0, "max": 512, "step": 8},
}


def test_upscale_required_order_is_pinned(comfy_stubs):
    input_types = ContextAnchoredTileUpscaleVL.INPUT_TYPES()
    assert list(input_types["required"]) == UPSCALE_REQUIRED_ORDER


def test_upscale_optional_is_model_and_negative(comfy_stubs):
    input_types = ContextAnchoredTileUpscaleVL.INPUT_TYPES()
    assert list(input_types["optional"]) == ["upscale_model", "negative"]


def test_upscale_has_no_mask_or_prompt_input(comfy_stubs):
    # Both absences are design decisions, not omissions: a mask needs the refine node
    # (region/vision-grid coordinates are unresolved), and any positive text re-admits the
    # phantom objects the vision-only positive exists to remove.
    input_types = ContextAnchoredTileUpscaleVL.INPUT_TYPES()
    all_inputs = {**input_types["required"], **input_types["optional"]}
    for absent in ("mask", "positive", "text", "prompt"):
        assert absent not in all_inputs, absent


def test_upscale_input_type_strings(comfy_stubs):
    input_types = ContextAnchoredTileUpscaleVL.INPUT_TYPES()
    all_inputs = {**input_types["required"], **input_types["optional"]}
    for name, type_string in UPSCALE_TYPE_BY_NAME.items():
        assert all_inputs[name][0] == type_string, name


def test_upscale_combo_inputs_come_from_comfy_samplers(comfy_stubs):
    import comfy.samplers

    required = ContextAnchoredTileUpscaleVL.INPUT_TYPES()["required"]
    assert required["sampler_name"][0] is comfy.samplers.KSampler.SAMPLERS
    assert required["scheduler"][0] is comfy.samplers.KSampler.SCHEDULERS
    # A default outside its own list is an invalid widget the UI silently re-points.
    assert required["sampler_name"][1]["default"] == "dpmpp_2m"
    assert required["sampler_name"][1]["default"] in required["sampler_name"][0]
    assert required["scheduler"][1]["default"] == "sgm_uniform"
    assert required["scheduler"][1]["default"] in required["scheduler"][0]


def test_upscale_widget_options_exact(comfy_stubs):
    required = ContextAnchoredTileUpscaleVL.INPUT_TYPES()["required"]
    for name, expected in UPSCALE_WIDGET_OPTIONS.items():
        options = required[name][1]
        for key, value in expected.items():
            assert options[key] == value, "{}[{}]".format(name, key)


def test_upscale_every_input_has_a_tooltip(comfy_stubs):
    input_types = ContextAnchoredTileUpscaleVL.INPUT_TYPES()
    all_inputs = {**input_types["required"], **input_types["optional"]}
    for name, definition in all_inputs.items():
        tooltip = definition[1].get("tooltip")
        assert isinstance(tooltip, str) and tooltip, name


def test_upscale_node_class_attributes(comfy_stubs):
    assert ContextAnchoredTileUpscaleVL.RETURN_TYPES == ("IMAGE",)
    assert ContextAnchoredTileUpscaleVL.FUNCTION == "refine"
    assert callable(ContextAnchoredTileUpscaleVL.refine)
    assert ContextAnchoredTileUpscaleVL.CATEGORY == "sampling/custom_sampling"
    # VALIDATE_INPUTS is inherited, so the /8 + range checks cover this node's geometry
    # widgets too — including its larger tile defaults.
    assert (
        ContextAnchoredTileUpscaleVL.VALIDATE_INPUTS(
            max_tile_width=1536, max_tile_height=2048, context_anchor=32, context_overlap=32
        )
        is True
    )
    assert isinstance(ContextAnchoredTileUpscaleVL.VALIDATE_INPUTS(max_tile_width=1001), str)


def test_upscale_refine_signature_matches_input_names(comfy_stubs):
    # ComfyUI calls FUNCTION with the input names as keywords: a rename on either side
    # is a TypeError at execution time, which this catches at test time instead.
    import inspect

    input_types = ContextAnchoredTileUpscaleVL.INPUT_TYPES()
    parameters = list(inspect.signature(ContextAnchoredTileUpscaleVL.refine).parameters)
    expected = ["self"] + list(input_types["required"]) + list(input_types["optional"])
    assert parameters == expected


def test_upscale_input_types_does_not_leak_into_the_other_nodes(comfy_stubs):
    ContextAnchoredTileUpscaleVL.INPUT_TYPES()

    base = ContextAnchoredTileRefine.INPUT_TYPES()
    assert list(base["required"]) == REQUIRED_ORDER
    assert list(base["optional"]) == ["mask"]

    vl = ContextAnchoredTileRefineVL.INPUT_TYPES()
    assert list(vl["required"]) == REQUIRED_ORDER + ["clip"]
    assert list(vl["optional"]) == ["mask"]
