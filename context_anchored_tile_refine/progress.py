"""One progress ledger for a VL node execution: ONE bar, divided into budget segments.

THE PROBLEM. A VL run emits from FOUR independent ProgressBar objects: the upscale model
pass (upscale.py), core's per-token bar inside `clip.generate` (llama.py, sized to
max_length and abandoned at the stop token), the tile text pre-pass
(captions.generate_caption_set or tags.generate_tag_trace) and the sampling bar (sync.py).
Each new object RESETS the frontend display. The ledger replaces them with one bar whose
value only ever grows.

THE UNIT is ONE DiT EVAL OF ONE TILE, because that is the one cost the run knows exactly
before it starts (`stepper.plan_evals` x the tile count). Every other phase is budgeted in
that unit by a named constant below. The constants are calibration knobs, not magic numbers
scattered through the engine.

THE STRUCTURE, in one picture:

    plan      [ upscale ][ clip load ][ captions ][ vision encode ][ canvas encode ]
              [ sampling ][ decode ]              <- per PICTURE, repeated when B > 1
    cursor     ^ first entry not yet CLOSED, and the entry AT the cursor may be OPEN
    value      sum of every closed entry's real size + the open entry's fill
    total      value's base + every entry from the cursor onward, at its planned size

`open` walks the cursor forward to the named entry (dropping any the run skipped), `resize`
re-fits the OPEN entry when its true size becomes known, `advance` fills it. The two
invariants, and the only ones: the emitted VALUE never decreases, and the emitted TOTAL is
never below the value already emitted.

THE STATUS LINE rides the same emit: every text the frontend shows under the bar is derived
from the state above (which entry is open, its fill, the last caption counters), so the line
and the bar can never disagree and no engine file has to carry a second notion of "what is
happening now". It is emitted only when it CHANGES, which is what keeps a per-tile advance
inside one segment from restating the same words.

THE SHIM. `with ledger:` enters the toolkit `NodeBar`'s routing, so any bar
constructed by core INSIDE the run maps its updates into the ledger's current segment
instead of resetting the display. That is a comfy MODULE-global patch, normally against
this package's rules. It is accepted here because those bars are constructed inside core
functions with no instance to patch and no pbar parameter to pass (llama.py's token bar
takes none). The window is the run only, and the toolkit restores the real class on exit.
The ledger's OWN `NodeBar` is built in `__init__`, before the patch, so its bar is genuine.
SCOPE: the shim captures bars constructed through the `comfy.utils.ProgressBar` ATTRIBUTE
(llama.py's token bar, sd.py's VAE tiled fallbacks, upscale.py's tiled_scale bar). The
KNOWN ESCAPE is sd.py:367, whose `ProgressBar` comes from a module-level `from comfy.utils
import ProgressBar` binding, reachable from every CLIP encode this package makes when CLIP
hook scheduling is active. Nothing here claims totality.

Module scope is STDLIB ONLY: no torch, no comfy and no logit_classifier. Each is imported
lazily inside the methods that need it, and a subprocess test pins them.
"""

import time

# --- the budget, in DiT-eval units (see the module docstring for what the unit is) -------
#
# Only the sampling segment is EXACT (stepper.plan_evals x tile count, re-fit at stepper
# intake). Every constant here is an ASSUMED wall-time ratio against one tile eval, a
# tuning knob that sets that phase's SHARE of the total and nothing else. Changing one
# changes only the pacing, never the correctness of the value/total invariants. A segment
# buys a phase an honest share of the total and a named status line, NOT movement:
# CLIP_LOAD / VISION_ENCODE / CAPTION_ENCODE have no fill source (core builds no inner
# bar there), so they sit at zero fill and are credited whole when the next phase opens.
W_UPSCALE_STEP = 0.1          # one comfy tiled_scale step of the upscale model
W_CLIP_LOAD = 4.0             # the upscale node's FIRST CLIP call: CLIP.load_model + the
                              # text encoder's move onto the GPU, which no other bar covers
K_CAPTION = 12.0              # one VLM caption: an autoregressive decode of up to the
                              # preset's max_tokens, the run's slowest per-tile step
K_TAG_TILE = 4.0              # one tile's tag pass (propose, verify, locate): about 8 s per
                              # tile at 8K (tests-AB/ab_tile_tags.py) against 19-39 s per caption
W_ENCODE = 4.0                # ONE vision encode of the entire canvas (up to the 2 MP cap)
W_ENCODE_CROP = 0.5           # one tile's own crop encode (crop_tokens x 1024 px, ~0.1 MP)
W_ENCODE_CAPTION_TEXT = 0.5   # one per-tile caption TEXT encode (scales with tile count:
                              # captions.py encodes each tile's caption separately)
W_ENCODE_TILE = 0.5           # one tile's VAE window encode
W_DECODE_TILE = 1.0           # one tile's VAE window decode


def vision_encode_units(n_tiles, vision):
    # One picture's VISION_ENCODE units, the mirror of vl.build_vision_rows' own tower passes:
    # one canvas encode when the canvas rows are on, one crop encode per tile when the crop
    # rows are on (`vision` is the settings file's [vision] table). Shared by preset_picture
    # and the engine's open() so the two can never disagree.
    units = W_ENCODE if vision.canvas_tokens > 0 else 0.0
    if vision.crop_tokens > 0:
        units += W_ENCODE_CROP * max(int(n_tiles), 1)
    return units


def caption_segment(preset, n_tiles, rows):
    # One picture's CAPTIONS segment as (units, chunks): one chunk per tile row plus one per
    # style row. Shared by preset_picture and the engine's open(), so the ledger total and the
    # caption_done calls the text pass makes can never disagree. Lazy import: this module's
    # scope stays stdlib only.
    from . import captions

    tile_chunks = int(n_tiles) * int(rows)
    style_chunks = captions.style_row_count(preset, int(rows))
    tile_units = K_TAG_TILE if preset.tile_text == captions.TILE_TEXT_TAGS else K_CAPTION
    return tile_chunks * tile_units + style_chunks * K_CAPTION, tile_chunks + style_chunks


def caption_status_word(preset):
    # The CAPTIONS segment's status word. Both kinds share the segment, so the line names the
    # pass the preset runs.
    from . import captions

    return "tagging" if preset.tile_text == captions.TILE_TEXT_TAGS else "captioning"

# --- segment names ------------------------------------------------------------------------
# One string per phase, defined once so the engine, the ledger's plan and any status-text
# consumer cannot drift apart.
UPSCALE = "upscale"
CLIP_LOAD = "clip load"
CAPTIONS = "captions"
VISION_ENCODE = "vision encode"
CAPTION_ENCODE = "caption encode"
CANVAS_ENCODE = "canvas encode"
SAMPLING = "sampling"
DECODE = "decode"

# --- the status line -----------------------------------------------------------------------
# The fixed line per segment. CAPTIONS and SAMPLING carry a count and a percent and are built
# in `Ledger._status` instead. CAPTION_ENCODE shares VISION_ENCODE's line on purpose: both are
# the conditioning build that follows the captions, and the distinction between them is a
# vlm_method the watcher already chose, not news about the run.
STATUS_TEXT = {
    UPSCALE: "upscaling",
    CLIP_LOAD: "loading text encoder",
    VISION_ENCODE: "encoding image",
    CAPTION_ENCODE: "encoding image",
    CANVAS_ENCODE: "encoding tiles",
    DECODE: "decoding",
}


def build_plan(vlm_method, steps, batch=1, upscale_model=False, clip_load=False):
    """The ordered segment plan for one VL node execution, at PROVISIONAL sizes.

    Sizes that depend on the grid (every per-tile budget) and on the sampler (the eval
    total) are not knowable here. This plan is what gives the bar a total to start from,
    and each segment is re-fit as its truth arrives. PHASE ORDER FOLLOWS THE CODE, per
    vlm_method: sync.build_tile_positives runs the captions FIRST and the conditioning build
    after, so "vision tokens and captions" reads [captions][vision encode] and never the
    reverse. Lazy import: this module's scope stays torch-free, and captions.py owns the
    vlm_method strings so the two cannot drift.
    """
    from . import captions

    surface = captions.method_surface(vlm_method)
    plan = []
    if upscale_model:
        plan.append((UPSCALE, W_UPSCALE_STEP))
    if clip_load:
        plan.append((CLIP_LOAD, W_CLIP_LOAD))
    for _picture in range(max(int(batch), 1)):
        if surface == captions.VLM_METHOD_VISION:
            plan.append((VISION_ENCODE, W_ENCODE))
        elif surface == captions.VLM_METHOD_VISION_CAPTIONS:
            plan.append((CAPTIONS, K_CAPTION))
            plan.append((VISION_ENCODE, W_ENCODE + W_ENCODE_CAPTION_TEXT))
        else:
            plan.append((CAPTIONS, K_CAPTION))
            plan.append((CAPTION_ENCODE, W_ENCODE_CAPTION_TEXT))
        plan.append((CANVAS_ENCODE, W_ENCODE_TILE))
        plan.append((SAMPLING, float(max(int(steps), 0))))
        plan.append((DECODE, W_DECODE_TILE))
    return tuple(plan)


class Ledger:
    """The segment bookkeeping over the run's ONE toolkit `NodeBar`, which keeps it monotone.

    Created only by nodes: node.py's VL nodes through build_ledger and testing.py's
    Captions node through build_caption_ledger, because a node is where the run's whole
    shape is in hand. Every engine function takes one as an optional `progress=` and
    creates nothing when it is None, so direct callers and the tests-AB harnesses see no
    behavior change at all.
    No locking: the stepper's scheduler is cooperative-serial (exactly one lane runs at a
    time) and comfy's own progress hook is thread-safe.
    """

    def __init__(self, plan, unique_id=None):
        from . import captions

        # The nodes build the ledger before any GPU work, so a missing library is named here.
        self._toolkit = captions.comfy_toolkit()
        self._started = 0.0
        self._plan = [[str(name), float(units)] for name, units in plan]
        self._cursor = 0          # index of the first plan entry not yet CLOSED
        self._is_open = False     # is self._plan[self._cursor] currently open?
        self._done = 0.0          # units of every closed entry, at its real size
        self._fill = 0.0          # units filled inside the open entry
        self._chunks = 1          # sub-divisions of the open entry (one per caption)
        self._chunk_index = 0     # which sub-division is in progress
        self._chunk_base = None   # this segment's offset into a run-wide caption counter
        self._status_word = "captioning"  # the CAPTIONS line's verb, set at every open
        self.unique_id = unique_id
        self.segments = []        # the plan entries opened, in order (name, units)
        self.caption_counters = None  # last (i, n) a caption reported, for the captioning line
        # The "image b/B" prefix. B is READ OFF THE PLAN rather than passed in a second time:
        # build_plan appends one sampling entry per picture, so the plan is the one place the
        # batch is stated and the prefix cannot contradict the segments it labels.
        self._pictures = sum(1 for name, _units in self._plan if name == SAMPLING) or 1
        self._picture = 1
        # Built before __enter__ routes comfy.utils.ProgressBar, so the ledger's own bar is
        # core's class and can never route into the ledger itself.
        self._bar = self._toolkit.NodeBar(self._plan_units(), unique_id)

    # ---- reading -------------------------------------------------------------------------

    @property
    def value(self):
        return self._bar.value

    @property
    def total(self):
        return self._bar.total

    # ---- segments ------------------------------------------------------------------------

    def open(self, name, units=None, chunks=1, status_word="captioning"):
        """Close whatever is open and open the plan's next `name` entry at `units`.

        The cursor WALKS to that entry, dropping any planned entry the run skipped (an
        empty mask returns before the canvas encode, and a picture whose grid differs skips
        nothing but re-fits). A name the plan does not carry is inserted at the cursor, so
        an unplanned phase costs pacing rather than correctness.
        """
        self._close()
        index = next((i for i in range(self._cursor, len(self._plan)) if self._plan[i][0] == name), None)
        if index is None:
            self._plan.insert(self._cursor, [name, 0.0])
        else:
            del self._plan[self._cursor:index]
        if units is not None:
            self._plan[self._cursor][1] = float(units)
        self._is_open = True
        self._fill = 0.0
        self._chunks = max(int(chunks), 1)
        self._chunk_index = 0
        self._chunk_base = None
        self._status_word = status_word
        self.segments.append(self._plan[self._cursor])
        self._show()

    def resize(self, units):
        # The open segment's true size, once it is known (the upscale model's real step
        # count, including every OOM tile-halving retry). Never below what is already
        # filled: the total must stay >= the value already emitted.
        if self._is_open:
            self._plan[self._cursor][1] = max(float(units), self._fill)
            self._show()

    def preset(self, name, units):
        # Every PENDING match is pre-sized, so the total never collapses when a large segment
        # opens at its true size. Every match, not the first, because with B > 1 names repeat.
        start = self._cursor + 1 if self._is_open else self._cursor
        for entry in self._plan[start:]:
            if entry[0] == name:
                entry[1] = float(units)
        self._show()

    def preset_picture(self, preset, n_tiles, rows, eval_total, vision_units):
        # The same arithmetic the engine's open() calls carry, so open never moves the total.
        # `vision_units` is required: a forgotten one would size the segment without crop encodes.
        from . import captions

        count = max(int(n_tiles), 1)
        caption_units, _chunks = caption_segment(preset, count, max(int(rows), 1))
        if preset.surface == captions.VLM_METHOD_VISION:
            self.preset(VISION_ENCODE, float(vision_units))
        elif preset.surface == captions.VLM_METHOD_VISION_CAPTIONS:
            self.preset(CAPTIONS, caption_units)
            self.preset(VISION_ENCODE, float(vision_units) + count * W_ENCODE_CAPTION_TEXT)
        else:
            self.preset(CAPTIONS, caption_units)
            self.preset(CAPTION_ENCODE, count * W_ENCODE_CAPTION_TEXT)
        self.preset(CANVAS_ENCODE, count * W_ENCODE_TILE)
        self.preset(SAMPLING, float(eval_total) * count)
        self.preset(DECODE, count * W_DECODE_TILE)

    def close(self):
        self._close()
        self._show()

    def finish(self):
        # Also reached on denoise <= 0, where most of the plan never opens.
        self._close()
        self._done += sum(entry[1] for entry in self._plan[self._cursor:])
        self._cursor = len(self._plan)
        self._show()

    # ---- filling -------------------------------------------------------------------------

    def advance(self, units, preview=None):
        # UNIVERSAL RESUME RULE: the fill only ever grows and never passes the segment's
        # own boundary. That keeps a caption's retry chain and the upscale OOM retry (both
        # restart an inner bar mid-segment) monotone.
        if self._is_open:
            self._fill = min(max(self._fill, float(units)), self._plan[self._cursor][1])
        self._show(preview)

    def caption_done(self, index, count):
        # One caption finished. (index, count) are the text pass's RUN-WIDE counters
        # (captions.generate_caption_set or tags.generate_tag_trace), the same pair that
        # feeds that pass's standalone bar. A status-text reader therefore gets "captioning
        # i/n" spanning the whole batch, while the chunk arithmetic below re-bases them onto
        # THIS picture's segment (the first completion in a segment is its local chunk 1).
        self.caption_counters = (int(index), int(count))
        if not self._is_open:
            return
        if self._chunk_base is None:
            self._chunk_base = int(index) - 1
        self._chunk_index = int(index) - self._chunk_base
        self.advance(self._plan[self._cursor][1] * self._chunk_index / self._chunks)

    def route(self, value, total, preview=None):
        # The shim's one entry point: an inner bar's own (value, total) mapped into the
        # CURRENT segment, or into the current chunk of it, which is what turns core's
        # per-token bar into per-caption sub-progress. With nothing open the update is
        # emitted but cannot move the value.
        if not self._is_open or total <= 0:
            self._show(preview)
            return
        name, units = self._plan[self._cursor]
        # Core's token bar is sized to max_length and greedy decode stops early, so a caption
        # chunk fills at the toolkit's greedy ratio and holds until `caption_done` snaps it.
        ratio = self._toolkit.GREEDY_FILL_RATIO if name == CAPTIONS else 1.0
        low = units * self._chunk_index / self._chunks
        high = units * (self._chunk_index + 1) / self._chunks
        fraction = self._toolkit.token_fill(value, total, ratio)
        self.advance(low + (high - low) * fraction, preview)

    # ---- the scoped comfy.utils.ProgressBar patch -----------------------------------------

    def __enter__(self):
        self._started = time.perf_counter()
        self._bar.routing(self.route)
        return self

    def __exit__(self, exc_type, exc, traceback):
        seconds = time.perf_counter() - self._started

        self._bar.close(exc_type, f"done in {seconds:.1f} s")
        return False

    # ---- internals ------------------------------------------------------------------------

    def _close(self):
        if self._is_open:
            name, units = self._plan[self._cursor]
            self._done += units
            self._cursor += 1
            self._is_open = False
            self._fill = 0.0
            if name == DECODE:
                # A picture's block ENDS at its decode, so the "image b/B" prefix advances
                # here, never at a block's first segment, which is a different phase per
                # vlm_method and would have to be re-derived to be recognized.
                self._picture += 1

    def _plan_units(self):
        return self._done + sum(entry[1] for entry in self._plan[self._cursor:])

    def _value_units(self):
        return self._done + (self._fill if self._is_open else 0.0)

    def _status(self):
        """The line this state says, or None for "nothing to report, leave the last one".

        Read off the open segment alone, so a phase that reports nothing (a closed cursor
        between segments, a caption segment before its first completion, a sampling segment
        before its first step lands) leaves the previous phase's line standing rather than
        blanking the row.
        """
        if not self._is_open:
            return None
        name, units = self._plan[self._cursor]
        if name == CAPTIONS:
            # Reports before the first completion too, because the first caption (a VL encoder
            # cold load plus a full greedy decode) is the pre-pass's longest silent stretch.
            if self._chunk_base is None:
                if self.caption_counters is not None:
                    # Anticipate the next run-wide index, so the count never appears to
                    # restart at a picture boundary ("captioning 3/6", not "captioning 1/3").
                    return (f"{self._status_word} "
                            f"{min(self.caption_counters[0] + 1, self.caption_counters[1])}/{self.caption_counters[1]}")
                return f"{self._status_word} 1/{self._chunks}"
            return f"{self._status_word} {self.caption_counters[0]}/{self.caption_counters[1]}"
        if name == SAMPLING:
            # 0% IS a report: the diffusion model loads inside the first lane's sample() call,
            # and the previous phase's line standing across that load would mislabel it.
            percent = 100 if units <= 0 else round(100 * self._fill / units)
            return f"sampling {percent}%"
        return STATUS_TEXT.get(name)

    def _show(self, preview=None):
        text = self._status()

        self._bar.emit(self._value_units(), total_units=self._plan_units(), preview=preview)
        if text is None:
            return
        if self._pictures > 1:
            text = f"image {self._picture}/{self._pictures}: {text}"
        self._bar.status(text)


def build_ledger(vlm_method, steps, batch=1, upscale_model=False, clip_load=False, unique_id=None):
    # The one constructor node.py calls: the plan, then the ledger over it.
    return Ledger(build_plan(vlm_method, steps, batch, upscale_model, clip_load), unique_id=unique_id)


def build_caption_ledger(preset, n_tiles, unique_id=None):
    """The ledger for a run that writes tile texts and nothing else (the Captions test node):
    one CAPTIONS segment sized by `caption_segment` for one picture of `n_tiles` tiles,
    already open, for either preset kind. Without it that node emits from two bars per
    caption, core's per-token bar and the text pass's own, and the display resets at every
    caption."""
    units, chunks = caption_segment(preset, n_tiles, 1)
    ledger = Ledger(((CAPTIONS, units),), unique_id=unique_id)
    ledger.open(CAPTIONS, units, chunks=chunks, status_word=caption_status_word(preset))
    return ledger
