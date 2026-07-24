> **Source:** https://docs.comfy.org/custom-nodes/v3_migration (mdx source: `custom-nodes/v3_migration.mdx` in [Comfy-Org/docs](https://github.com/Comfy-Org/docs); page title "V3 Migration") — **excerpt: lines 1-73 of 975 (Overview + Core Concepts + V1 vs V3 Architecture only)**
> **Doc set version:** docs.comfy.org (current, main branch)
> **Retrieved:** 2026-07-22
>
> **Why this is an excerpt:** the brief targets ComfyUI 0.3.45 and the classic (V1) `INPUT_TYPES`/`RETURN_TYPES`/`FUNCTION`/`CATEGORY`/`NODE_CLASS_MAPPINGS` node interface used throughout the rest of this doc set. docs.comfy.org's current guidance is increasingly written against a newer "V3" node schema (`io.ComfyNode` / `define_schema` / `comfy_entrypoint`). This excerpt captures only the framing that names the two schemas and shows the side-by-side shape, so the difference can be flagged consciously. The full step-by-step V1→V3 migration guide (the remaining ~900 lines: per-feature migration steps, input type reference, schema reference, advanced features, complete example) is out of scope per the brief and was not captured — re-fetch the source URL above if it becomes needed.

# V3 Migration

How to migrate your existing V1 nodes to the new V3 schema.

## Overview

The ComfyUI V3 schema introduces a more organized way of defining nodes, and future extensions to node features will only be added to V3 schema. You can use this guide to help you migrate your existing V1 nodes to the new V3 schema.

## Core Concepts

The V3 schema is kept on the new versioned Comfy API, meaning future revisions to the schema will be backwards compatible. ```comfy_api.latest``` will point to the latest numbered API that is still under development; the version before latest is what can be considered 'stable'. Version ```v0_0_2``` is the current (and first) API version so more changes will be made to it without warning. Once it is considered stable, a new version ```v0_0_3``` will be created for ```latest``` to point at.

```python
# use latest ComfyUI API
from comfy_api.latest import ComfyExtension, io, ui

# use a specific version of ComfyUI API
from comfy_api.v0_0_2 import ComfyExtension, io, ui
```

### V1 vs V3 Architecture

The biggest changes in V3 schema are:
- Inputs and Outputs defined by objects instead of a dictionary.
- The execution method is fixed to the name 'execute' and is a class method.
- ```def comfy_entrypoint()``` function that returns a ComfyExtension object defines exposed nodes instead of NODE_CLASS_MAPPINGS/NODE_DISPLAY_NAME_MAPPINGS 
- Node objects do not expose 'state' - ```def __init__(self)``` will have no effect on what is exposed in the node's functions, as all of them are class methods. The node class is sanitized before execution as well.

#### V1 (Legacy)
```python
class MyNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {...}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "my_category"

    def execute(self, ...):
        return (result,)

NODE_CLASS_MAPPINGS = {"MyNode": MyNode}
```

#### V3 (Modern)
```python
from comfy_api.latest import ComfyExtension, io

class MyNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MyNode",
            display_name="My Node",
            category="my_category",
            inputs=[...],
            outputs=[...]
        )

    @classmethod
    def execute(cls, ...) -> io.NodeOutput:
        return io.NodeOutput(result)

class MyExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MyNode]

async def comfy_entrypoint() -> ComfyExtension:
    return MyExtension()
```
