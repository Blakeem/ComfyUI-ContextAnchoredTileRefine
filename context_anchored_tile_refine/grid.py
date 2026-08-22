"""Pure grid math: the authoritative tile-layout solver, mirrored by the reference
implementation in docs/tile-simulator.html "PURE GRID MATH". Stdlib only.

Bands per tile, outward from the core: core -> context_overlap (the directional
feather) -> context_anchor (the frozen halo). The per-seam ring width is
r = context_overlap + context_anchor."""
import math
from dataclasses import dataclass


class GridConfigError(ValueError):
    # Configuration error: caps too small for the chosen context_overlap +
    # context_anchor per seam side. Carries the simulator's failure fields as attributes.
    def __init__(self, L, cap, ctx, overlap, r, fail_n, fail_base, reason, axis=None):
        # `axis` ("width"/"height") is optional so the internal symbols above can be followed
        # by one sentence in the caller's own widget names. Nothing in the leading text names
        # anything the user typed, and VALIDATE_INPUTS cannot pre-empt this (it never sees the
        # image).
        widgets = "" if axis is None else (
            f". max_tile_{axis} {cap} cannot hold context_anchor {ctx} + context_overlap {overlap} "
            f"on both sides. Raise max_tile_{axis} or lower context_overlap."
        )
        super().__init__(
            f"caps too small for overlap + context: L={L} cap={cap} ctx={ctx} overlap={overlap} "
            f"(fails at n={fail_n} with base={fail_base}, reason={reason}){widgets}"
        )
        self.r = r
        self.fail_n = fail_n
        self.fail_base = fail_base
        self.reason = reason


@dataclass(frozen=True)
class AxisSolution:
    n: int
    base: int
    last: int
    overhead: int
    r: int


@dataclass(frozen=True)
class Rect:
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class ExpandedRect(Rect):
    clamped: bool


@dataclass(frozen=True)
class Neighbors:
    left: bool
    right: bool
    top: bool
    bottom: bool


@dataclass(frozen=True)
class Tile:
    col: int
    row: int
    cls: str
    nb: Neighbors
    core: Rect
    # core + context_overlap on EVERY neighbor side — the diffused region (the
    # binary denoise mask); the context_anchor halo is crop_rect \ overlap_inner_rect.
    overlap_inner_rect: ExpandedRect
    # core + r on every neighbor side — the SYMMETRIC sampled extent (what the model
    # sees). Kept symmetric on purpose so the frozen context_anchor sits a consistent
    # distance from the core on all sides.
    crop_rect: ExpandedRect
    # core + context_overlap on the TOP/LEFT neighbor sides ONLY — the directional
    # pasted region, feathered into the already-processed neighbor (raster order).
    paste_rect: ExpandedRect
    kept_top: bool
    kept_left: bool
    sampled_w: int
    sampled_h: int


@dataclass(frozen=True)
class Layout:
    w: int
    h: int
    sol_x: AxisSolution
    sol_y: AxisSolution
    ctx: int
    overlap: int
    r: int
    tiles: tuple
    total_sampled_px: int
    clamped_tiles: int


def round_up_multiple(x, multiple):
    # Smallest multiple of `multiple` that is >= x.
    return math.ceil(x / multiple) * multiple


def round8_up(x):
    # Smallest multiple of 8 that is >= x.
    return round_up_multiple(x, 8)


def solve_axis(L, cap, ctx, overlap=0, multiple=8, axis=None):
    # Per-axis grid solve (authoritative):
    #   r = context_anchor + context_overlap
    #   overhead(n) = 0 | r | 2r  for n = 1 | 2 | >= 3
    #   base(n) = round_up_multiple(ceil(L / n), multiple); choose the smallest n with
    #   base(n) + overhead(n) <= cap.
    # There is NO fade/overlap floor — a base is judged on the cap constraint alone
    # (owner's decision; keep it simple). Only "exhausted" remains as a failure.
    # `multiple` is the pixel granularity a sampled crop must land on: 8 for the
    # /8-latent image VAEs (the default, which keeps the image path bit-identical),
    # 32 for MiniMax H3 (VAE spatial factor 16 x DiT patch 2).
    r = ctx + overlap
    max_n = max(1, math.ceil(L / multiple)) + 1  # defensive bound; base(max_n) = multiple

    for n in range(1, max_n + 1):
        base = round_up_multiple(math.ceil(L / n), multiple)
        overhead = 0 if n == 1 else r if n == 2 else 2 * r

        if base + overhead <= cap:
            return AxisSolution(n=n, base=base, last=L - (n - 1) * base, overhead=overhead, r=r)
    # Unreachable unless the cap is smaller than the minimum base `multiple` + overhead.
    raise GridConfigError(L, cap, ctx, overlap, r, max_n, multiple, "exhausted", axis=axis)


def expand_rect(core, amount, nb, W, H):
    # Expand a core rect by `amount` on each side that has a neighbor, clamped to
    # the canvas.
    want_x0 = core.x0 - (amount if nb.left else 0)
    want_x1 = core.x1 + (amount if nb.right else 0)
    want_y0 = core.y0 - (amount if nb.top else 0)
    want_y1 = core.y1 + (amount if nb.bottom else 0)
    x0 = max(0, want_x0)
    x1 = min(W, want_x1)
    y0 = max(0, want_y0)
    y1 = min(H, want_y1)
    return ExpandedRect(
        x0=x0, y0=y0, x1=x1, y1=y1,
        clamped=(x0 != want_x0 or x1 != want_x1 or y0 != want_y0 or y1 != want_y1),
    )


def axis_class(i, n):
    # Per-axis position class: "single" (n=1), "end" (first/last), "mid".
    if n == 1:
        return "single"
    return "end" if i == 0 or i == n - 1 else "mid"


def tile_class_label(cx, cy):
    # 2D tile class label from the two per-axis classes.
    if cx == "single" and cy == "single":
        return "single"
    if cx == "mid" and cy == "mid":
        return "interior"
    if cx == "end" and cy == "end":
        return "corner"
    if (cx == "end" and cy == "mid") or (cx == "mid" and cy == "end"):
        return "edge"
    # one axis is single: a 1 x N strip
    return "strip middle" if cx == "mid" or cy == "mid" else "strip end"


def build_layout(W, H, sx, sy, ctx, overlap=0):
    # Full 2D layout from two per-axis solves; row-major (raster) tiles, all values in
    # canvas px. Two distinct geometries per tile:
    #   crop_rect          = core + r on EVERY neighbor side (symmetric sampled extent).
    #   overlap_inner_rect = core + overlap on every neighbor side (the diffused region).
    #   paste_rect         = core + overlap on the TOP/LEFT neighbor sides only
    #                        (directional pasted region, feathered into the neighbor).
    # In raster order a tile's already-processed neighbors are exactly the tile above
    # (row>0) and to the left (col>0), so kept_top/kept_left drive the directional
    # paste; each interior seam is feathered exactly once, by the later tile.
    r = ctx + overlap
    tiles = []
    total_sampled_px = 0
    clamped_tiles = 0

    for row in range(sy.n):
        for col in range(sx.n):
            core = Rect(
                x0=col * sx.base,
                y0=row * sy.base,
                x1=W if col == sx.n - 1 else (col + 1) * sx.base,
                y1=H if row == sy.n - 1 else (row + 1) * sy.base,
            )
            nb = Neighbors(left=col > 0, right=col < sx.n - 1, top=row > 0, bottom=row < sy.n - 1)
            kept_top = row > 0
            kept_left = col > 0

            overlap_inner_rect = expand_rect(core, overlap, nb, W, H)
            crop_rect = expand_rect(core, r, nb, W, H)
            # Directional: overlap on top/left only (never right/bottom), and clamped per axis
            # to that axis's base so paste_rect never reaches past the PREVIOUS tile's core
            # start. That is what keeps the invariant above true — each interior seam feathered
            # exactly once, by the later tile — rather than cross-dissolving over a whole
            # earlier core (the wide blend of two independent refinements CLAUDE.md prohibits).
            # A no-op whenever base >= overlap, i.e. every default configuration; solve_axis has
            # no fade floor by design, so base < overlap is reachable from widget-legal values.
            # One expand_rect call per axis, each with the other axis's sides forced off,
            # because the two clamps differ (sx.base vs sy.base).
            paste_x = expand_rect(core, min(overlap, sx.base), Neighbors(left=nb.left, right=False, top=False, bottom=False), W, H)
            paste_y = expand_rect(core, min(overlap, sy.base), Neighbors(left=False, right=False, top=nb.top, bottom=False), W, H)
            paste_rect = ExpandedRect(
                x0=paste_x.x0, y0=paste_y.y0, x1=paste_x.x1, y1=paste_y.y1,
                clamped=paste_x.clamped or paste_y.clamped,
            )
            sampled_w = crop_rect.x1 - crop_rect.x0
            sampled_h = crop_rect.y1 - crop_rect.y0
            label = tile_class_label(axis_class(col, sx.n), axis_class(row, sy.n))

            total_sampled_px += sampled_w * sampled_h
            if crop_rect.clamped or overlap_inner_rect.clamped:
                clamped_tiles += 1

            tiles.append(Tile(
                col=col, row=row, cls=label, nb=nb,
                core=core, overlap_inner_rect=overlap_inner_rect, crop_rect=crop_rect,
                paste_rect=paste_rect, kept_top=kept_top, kept_left=kept_left,
                sampled_w=sampled_w, sampled_h=sampled_h,
            ))
    return Layout(
        w=W, h=H, sol_x=sx, sol_y=sy, ctx=ctx, overlap=overlap, r=r,
        tiles=tuple(tiles), total_sampled_px=total_sampled_px, clamped_tiles=clamped_tiles,
    )
