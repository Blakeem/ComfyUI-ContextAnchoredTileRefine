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
    "context_anchor": {"default": 64, "min": 0, "max": 512, "step": 8},
    "context_overlap": {"default": 8, "min": 0, "max": 512, "step": 8},
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
            max_tile_width=1024, max_tile_height=1024, context_anchor=64, context_overlap=8
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
