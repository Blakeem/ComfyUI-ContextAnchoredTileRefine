"""One progress ledger for a VL node execution: ONE bar, divided into budget segments.

THE PROBLEM. A VL run emits from FOUR independent ProgressBar objects — the upscale model
pass (upscale.py), core's per-token bar inside `clip.generate` (llama.py, sized to
max_length and abandoned at the stop token), the caption pre-pass (captions.py) and the
sampling bar (sync.py) — and each new object RESETS the frontend display. The ledger
replaces them with one bar whose value only ever grows.

THE UNIT is ONE DiT EVAL OF ONE TILE, because that is the one cost the run knows exactly
before it starts (`stepper.plan_evals` x the tile count). Every other phase is budgeted in
that unit by a named constant below; the constants are calibration knobs, not magic numbers
scattered through the engine.

THE STRUCTURE, in one picture:

    plan      [ upscale ][ clip load ][ captions ][ vision encode ][ canvas encode ]
              [ sampling ][ decode ]              <- per PICTURE, repeated when B > 1
    cursor     ^ first entry not yet CLOSED; the entry AT the cursor may be OPEN
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

THE SHIM. `with ledger:` enters logit_classifier's `routed_progress_bars`, so any bar
constructed by core INSIDE the run maps its updates into the ledger's current segment
instead of resetting the display. That is a comfy MODULE-global patch, normally against
this package's rules. It is accepted here because those bars are constructed inside core
functions with no instance to patch and no pbar parameter to pass (llama.py's token bar
takes none). The window is the run only, and the toolkit restores the real class on exit.
The ledger's OWN bar is built in `__init__`, before the patch, so it is genuine.
SCOPE: the shim captures bars constructed through the `comfy.utils.ProgressBar` ATTRIBUTE
(llama.py's token bar, sd.py's VAE tiled fallbacks, upscale.py's tiled_scale bar). The
KNOWN ESCAPE is sd.py:360, whose `ProgressBar` comes from a module-level `from comfy.utils
import ProgressBar` binding — reachable from every CLIP encode this package makes when CLIP
hook scheduling is active. Nothing here claims totality.

Module scope is STDLIB ONLY: no torch, no comfy and no logit_classifier. Each is imported
lazily inside the methods that need it, and a subprocess test pins them.
"""

import time
from contextlib import ExitStack

# --- the budget, in DiT-eval units (see the module docstring for what the unit is) -------
#
# Only the sampling segment is EXACT (stepper.plan_evals x tile count, re-fit at stepper
# intake). Every constant here is an ASSUMED wall-time ratio against one tile eval — a
# tuning knob that sets that phase's SHARE of the total and nothing else; changing one
# changes only the pacing, never the correctness of the value/total invariants. A segment
# buys a phase an honest share of the total and a named status line — NOT movement:
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

# Greedy generation hits its stop token at roughly 50-75% of max_length (owner-observed),
# so core's per-token bar fills only partway before it is abandoned. A caption's chunk is
# therefore mapped to fill at this fraction of max_length and HOLD at its boundary.
CAPTION_FILL_RATIO = 0.65

# comfy's ProgressBar carries integers, while the budget above is fractional; the emitted
# value is the unit count scaled by this. Purely a resolution choice.
EMIT_SCALE = 100


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
    tile_units = K_TAG_TILE if preset.kind == captions.TILE_TEXT_TAGS else K_CAPTION
    return tile_chunks * tile_units + style_chunks * K_CAPTION, tile_chunks + style_chunks


def caption_status_word(preset):
    # The CAPTIONS segment's status word. Both kinds share the segment, so the line names the
    # pass the preset runs.
    from . import captions

    return "tagging" if preset.kind == captions.TILE_TEXT_TAGS else "captioning"

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


def linear_fill(value, total):
    # A routed bar's own progress as a fraction of its own total — the default mapping.
    if total <= 0:
        return 1.0
    return min(1.0, max(0.0, value / total))


def caption_fill(index, max_length):
    # Core's token bar is sized to max_length but greedy decode stops early, so a linear
    # mapping would leave every caption's chunk visibly short. Token `index` fills the chunk
    # at CAPTION_FILL_RATIO of max_length and the chunk HOLDS there; the exact boundary is
    # reached by the completion snap (`caption_done`), never by a token.
    from . import captions

    return captions.comfy_toolkit().token_fill(index, max_length, CAPTION_FILL_RATIO)


# Which mapping a routed bar gets, by the name of the segment it lands in. Absent = linear.
SUB_FILL = {CAPTIONS: caption_fill}


def build_plan(vlm_method, steps, batch=1, upscale_model=False, clip_load=False):
    """The ordered segment plan for one VL node execution, at PROVISIONAL sizes.

    Sizes that depend on the grid (every per-tile budget) and on the sampler (the eval
    total) are not knowable here — this plan is what gives the bar a total to start from,
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
    """The run's ONE ProgressBar, plus the segment bookkeeping that keeps it monotone.

    Created ONLY in node.py (that is where the run's whole shape — node kind, upscale
    model, vlm_method, batch — is in hand, and it removes any two-owner ambiguity); every
    engine function takes one as an optional `progress=` and creates nothing when it is
    None, so direct callers and the tests-AB harnesses see no behavior change at all.
    No locking: the stepper's scheduler is cooperative-serial (exactly one lane runs at a
    time) and comfy's own progress hook is thread-safe.
    """

    def __init__(self, plan, unique_id=None):
        import comfy.utils

        from . import captions

        # The nodes build the ledger before any GPU work, so a missing library is named here.
        self._toolkit = captions.comfy_toolkit()
        self._routing = ExitStack()
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
        self._value = 0          # last EMITTED value, in scaled integer units
        self._total = 0
        self.unique_id = unique_id
        self.segments = []        # the plan entries opened, in order (name, units)
        self.chunks = None        # last (i, n) a caption reported, for the captioning line
        self._status_text = None  # last EMITTED status line; a repeat is never re-sent
        # The "image b/B" prefix. B is READ OFF THE PLAN rather than passed in a second time:
        # build_plan appends one sampling entry per picture, so the plan is the one place the
        # batch is stated and the prefix cannot contradict the segments it labels.
        self._pictures = sum(1 for name, _units in self._plan if name == SAMPLING) or 1
        self._picture = 1
        # Built before __enter__ routes comfy.utils.ProgressBar, so the ledger's own bar is
        # core's class and can never route into the ledger itself.
        self._pbar = comfy.utils.ProgressBar(self._scaled(self._plan_units()))
        self._emit()

    # ---- reading -------------------------------------------------------------------------

    @property
    def value(self):
        return self._value

    @property
    def total(self):
        return self._total

    # ---- segments ------------------------------------------------------------------------

    def open(self, name, units=None, chunks=1, status_word="captioning"):
        """Close whatever is open and open the plan's next `name` entry at `units`.

        The cursor WALKS to that entry, dropping any planned entry the run skipped (an
        empty mask returns before the canvas encode; a picture whose grid differs skips
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
        self._emit()

    def resize(self, units):
        # The open segment's true size, once it is known (the upscale model's real step
        # count, including every OOM tile-halving retry). Never below what is already
        # filled: the total must stay >= the value already emitted.
        if self._is_open:
            self._plan[self._cursor][1] = max(float(units), self._fill)
            self._emit()

    def preset(self, name, units):
        # A PENDING entry's true size, the moment it becomes known — BEFORE the segment
        # opens. Without this the displayed fraction inflates against a provisional total
        # and then collapses when a big segment finally opens at its true size (the
        # sampling segment is ~n_tiles x its provisional size, so on a 30-tile grid the
        # bar read ~95% at the end of the captions and dropped to ~45% at sampling).
        # EVERY pending match is set, not just the first: with B>1 the plan repeats names
        # per picture, and seeding only picture k's block would re-create the same
        # collapse at every picture boundary (96% -> 50% on a 2-image batch). Later
        # pictures are seeded from THIS picture's grid as an estimate; each one still
        # re-presets its own block at its own grid solve, so a differing grid re-fits by
        # the grid delta instead of by ~n_tiles-fold.
        start = self._cursor + 1 if self._is_open else self._cursor
        for entry in self._plan[start:]:
            if entry[0] == name:
                entry[1] = float(units)
        self._emit()

    def preset_picture(self, preset, n_tiles, rows, eval_total, vision_units):
        # One picture's whole block at true sizes, called at that picture's grid solve —
        # the mirror of build_plan's per-picture entries with the real multipliers, and
        # the same arithmetic the engine's open() calls carry (they re-set the identical
        # numbers, so open never moves the total again). `preset` is the run's resolved
        # block, since its kind and its style rows size the caption segment
        # (caption_segment), and `vision_units` is the pre-pass's vision encode cost
        # (vision_encode_units), required because a forgotten one would silently size the
        # segment without the crop encodes.
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
        # End the open segment with no successor yet — updates arriving now cannot move the
        # value at all (there is no segment to map them into).
        self._close()
        self._emit()

    def finish(self):
        # Normal end of run: everything still planned is counted done, so value == total.
        # Reached on the legitimate denoise <= 0 setting too, where the engine returns
        # before sampling and most of the plan is never opened.
        self._close()
        self._done += sum(entry[1] for entry in self._plan[self._cursor:])
        self._cursor = len(self._plan)
        self._emit()

    # ---- filling -------------------------------------------------------------------------

    def advance(self, units, preview=None):
        # UNIVERSAL RESUME RULE: the fill only ever grows and never passes the segment's
        # own boundary, which is what keeps a caption's retry chain and the upscale OOM
        # retry — both of which restart an inner bar mid-segment — monotone.
        if self._is_open:
            self._fill = min(max(self._fill, float(units)), self._plan[self._cursor][1])
        self._emit(preview)

    def caption_done(self, index, count):
        # One caption finished. (index, count) are generate_tile_captions' own RUN-WIDE
        # counters — the same pair that feeds its standalone bar — so a status-text reader
        # gets "captioning i/n" spanning the whole batch, while the chunk arithmetic below
        # re-bases them onto THIS picture's segment (the first completion in a segment is
        # its local chunk 1).
        self.chunks = (int(index), int(count))
        if not self._is_open:
            return
        if self._chunk_base is None:
            self._chunk_base = int(index) - 1
        self._chunk_index = int(index) - self._chunk_base
        self.advance(self._plan[self._cursor][1] * self._chunk_index / self._chunks)

    def route(self, value, total, preview=None):
        # The shim's one entry point: an inner bar's own (value, total) mapped into the
        # CURRENT segment — or into the current chunk of it, which is what turns core's
        # per-token bar into per-caption sub-progress. With nothing open the update is
        # emitted but cannot move the value.
        if not self._is_open or total <= 0:
            self._emit(preview)
            return
        units = self._plan[self._cursor][1]
        low = units * self._chunk_index / self._chunks
        high = units * (self._chunk_index + 1) / self._chunks
        fraction = SUB_FILL.get(self._plan[self._cursor][0], linear_fill)(value, total)
        self.advance(low + (high - low) * fraction, preview)

    # ---- the scoped comfy.utils.ProgressBar patch -----------------------------------------

    def __enter__(self):
        self._started = time.perf_counter()
        self._routing.enter_context(self._toolkit.routed_progress_bars(self.route))
        return self

    def __exit__(self, exc_type, exc, traceback):
        seconds = time.perf_counter() - self._started
        text = ""

        self._routing.close()
        # The frontend ignores an empty line, so the run's last line must say how it ended
        # or a phase line such as "sampling 40%" stays under the node.
        text = self._toolkit.run_outcome(exc_type, f"done in {seconds:.1f} s")
        self._send_status(text)
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
                # here — never at a block's first segment, which is a different phase per
                # vlm_method and would have to be re-derived to be recognized.
                self._picture += 1

    def _plan_units(self):
        return self._done + sum(entry[1] for entry in self._plan[self._cursor:])

    def _value_units(self):
        return self._done + (self._fill if self._is_open else 0.0)

    @staticmethod
    def _scaled(units):
        return round(units * EMIT_SCALE)

    def _status(self):
        """The line this state says, or None for "nothing to report — leave the last one".

        Read off the open segment alone, so a phase that reports nothing (a closed cursor
        between segments, a caption segment before its first completion, a sampling segment
        before its first step lands) leaves the previous phase's line standing rather than
        blanking the row.
        """
        if not self._is_open:
            return None
        name, units = self._plan[self._cursor]
        if name == CAPTIONS:
            # caption_done's RUN-WIDE counters, so the count spans a batch exactly as the
            # standalone caption bar does. `_chunk_base` is what says those counters belong
            # to THIS segment: it is cleared at every open and set by the segment's first
            # completion. BEFORE the first completion the line still reports — the first
            # caption is the single longest un-narrated stretch of the pre-pass (the VL
            # encoder cold-loads and runs a full greedy decode inside it), so a blank row
            # there is exactly the freeze this feature removes. The pre-completion count
            # is this segment's LOCAL chunk count (the run-wide total is unknowable until
            # a completion reports it).
            if self._chunk_base is None:
                if self.chunks is not None:
                    # A previous picture already reported run-wide counters: anticipate the
                    # next index, so the count never appears to restart at a picture
                    # boundary ("captioning 3/6", not "captioning 1/3").
                    return (f"{self._status_word} "
                            f"{min(self.chunks[0] + 1, self.chunks[1])}/{self.chunks[1]}")
                return f"{self._status_word} 1/{self._chunks}"
            return f"{self._status_word} {self.chunks[0]}/{self.chunks[1]}"
        if name == SAMPLING:
            # Percent of THIS segment, moved once per completed model eval (the stepper's
            # on_eval tick — n_tiles ticks per sigma step, so the percent walks in ~1%
            # increments instead of jumping a whole step at a time). 0% IS a report: the
            # diffusion model loads onto the GPU inside the first lane's sample() call,
            # between this segment's open and its first eval — letting the previous
            # phase's line ("encoding tiles") stand across that load would mislabel the
            # longest single wait of the run.
            percent = 100 if units <= 0 else round(100 * self._fill / units)
            return f"sampling {percent}%"
        return STATUS_TEXT.get(name)

    def _emit_status(self):
        text = self._status()
        if text is None:
            return
        if self._pictures > 1:
            text = f"image {self._picture}/{self._pictures}: {text}"
        self._send_status(text)

    def _send_status(self, text):
        if text == self._status_text:
            return
        self._status_text = text
        self._toolkit.send_status(self.unique_id, text)

    def _emit(self, preview=None):
        total = max(self._scaled(self._plan_units()), self._value)
        value = min(max(self._scaled(self._value_units()), self._value), total)
        self._value = value
        self._total = total
        self._pbar.update_absolute(value, total, preview)
        self._emit_status()


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
