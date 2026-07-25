from context_anchored_tile_refine.node import ContextAnchoredTileRefine

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
    "seam_mode",
    "warp_amount",
    "warp_scale",
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
    "warp_amount": "FLOAT",
    "warp_scale": "INT",
    "mask": "MASK",
}

INT_WIDGET_OPTIONS = {
    "max_tile_width": {"default": 1024, "min": 256, "max": 16384, "step": 8},
    "max_tile_height": {"default": 1024, "min": 256, "max": 16384, "step": 8},
    # 32/32 is the A/B-settled compromise: invisible on detailed character scenes without
    # paying for a wide halo. Smooth-gradient scenes are handled by raising context_overlap.
    "context_anchor": {"default": 32, "min": 0, "max": 512, "step": 8},
    "context_overlap": {"default": 32, "min": 0, "max": 512, "step": 8},
    "warp_scale": {"default": 64, "min": 8, "max": 512, "step": 8},
}

FLOAT_WIDGET_OPTIONS = {
    "warp_amount": {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
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


def test_seam_mode_is_a_combo_defaulting_to_min_error():
    # seam_mode is a COMBO (a tuple of option strings), not a typed socket. min_error is the
    # A/B winner (it reads as natural rather than random), so it is the default; the other
    # two stay selectable for scenes where a routed cut has nothing to route through.
    from context_anchored_tile_refine.sampling import SEAM_MIN_ERROR, SEAM_MODES

    definition = ContextAnchoredTileRefine.INPUT_TYPES()["required"]["seam_mode"]
    assert list(definition[0]) == list(SEAM_MODES)        # node offers exactly the pipeline's modes
    assert list(SEAM_MODES) == ["straight", "warp", "min_error"]
    assert definition[1]["default"] == "min_error"
    # MUST be a list, not a tuple: ComfyUI gates its COMBO membership check on
    # isinstance(input_type, list), so a tuple silently skips validation entirely and an
    # API-submitted typo would run as `straight` with no error anywhere. Asserted on the
    # schema entry itself, since normalizing both sides with list()/tuple() would hide it.
    assert isinstance(definition[0], list), type(definition[0])
    # The node widget default and the pipeline's own default must not drift apart.
    import inspect

    from context_anchored_tile_refine import sampling

    assert inspect.signature(sampling.refine_image).parameters["seam_mode"].default == SEAM_MIN_ERROR
    assert inspect.signature(ContextAnchoredTileRefine.refine).parameters["seam_mode"].default == SEAM_MIN_ERROR


def test_ramp_shape_is_not_exposed_as_a_widget():
    # feather_plateau / feather_falloff were A/B widgets and are now baked into sampling.py.
    # Re-adding them as inputs would reopen a settled decision, so pin their absence.
    input_types = ContextAnchoredTileRefine.INPUT_TYPES()
    all_inputs = {**input_types["required"], **input_types["optional"]}
    assert "feather_plateau" not in all_inputs
    assert "feather_falloff" not in all_inputs


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
