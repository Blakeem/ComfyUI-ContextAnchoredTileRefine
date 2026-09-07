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


@dataclass(frozen=True)
class SubLayout:
    # One rectangular block of a parent layout's tiles, re-solved as a layout of its own, so an
    # engine run over `block` samples those tiles at the parent run's rects.
    # `block` is the region such a run samples, `region` the region it denoises. Both are in
    # parent canvas px. `layout` is block-local, its origin at (block.x0, block.y0).
    block: Rect
    region: Rect
    layout: Layout
    first_col: int
    first_row: int
    parent_tiles: tuple


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


def neighborhood(layout, index):
    # The 3x3 range of columns and rows around one tile, clamped to the grid, as the inclusive
    # bounds sub_layout takes.
    if index < 0 or index >= len(layout.tiles):
        raise ValueError(f"tile index {index} is outside the layout's {len(layout.tiles)} tiles")

    tile = layout.tiles[index]
    col0 = max(0, tile.col - 1)
    col1 = min(layout.sol_x.n - 1, tile.col + 1)
    row0 = max(0, tile.row - 1)
    row1 = min(layout.sol_y.n - 1, tile.row + 1)
    return col0, col1, row0, row1


def _axis_spans(first, last, axis):
    # The parent rects the block rule reads, reduced to one axis: the first in-range tile's core
    # start, crop start and diffused span, then the last in-range tile's crop end.
    if axis == "width":
        return (first.core.x0, first.crop_rect.x0,
                first.overlap_inner_rect.x0, first.overlap_inner_rect.x1, last.crop_rect.x1)
    return (first.core.y0, first.crop_rect.y0,
            first.overlap_inner_rect.y0, first.overlap_inner_rect.y1, last.crop_rect.y1)


def _sub_axis(sol, i0, i1, spans, length, ctx, overlap, axis):
    # One axis of a sub layout: the block span, the region span and the forced solution.
    # Which sides have tiles beyond them is read from the RANGE and never from a clamped rect,
    # because a crop that clamps to 0 is not evidence that no tile lies before it.
    core_start, crop_start, inner_start, inner_end, crop_end = spans
    r = ctx + overlap
    n = i1 - i0 + 1
    beyond_start = i0 > 0
    beyond_end = i1 < sol.n - 1

    # Under this the block's own tiles clamp their rings at the block edge while the parent's
    # reach past the bordering tile, so no block origin reproduces the parent crops.
    if n >= 2 and (beyond_start or beyond_end) and sol.base < r:
        raise ValueError(
            f"{axis} tile base {sol.base} is under context_anchor {ctx} plus context_overlap {overlap} "
            f"({r}), so a tile's frozen ring reaches past its bordering tile. context_anchor plus "
            f"context_overlap must not exceed the tile base for a block render."
        )

    if n == 1:
        # The lone tile samples the parent's crop extent as one tile whose core is the entire
        # block, and it denoises the parent's own diffused span whatever clamping that did.
        block = (crop_start, crop_end)
        region = (inner_start, inner_end)
    else:
        # build_layout places the first core at the block origin, so a block that started at the
        # crop would shift every base. The end ring stays and the last tile absorbs it as core.
        start = core_start if beyond_start else crop_start
        block = (start, crop_end)
        # The region predicates read the RECT, not the range: a block whose end reaches the canvas
        # edge must refine that edge strip even when a tile lies beyond it in the grid.
        region = (start + ctx if start > 0 else start,
                  crop_end - ctx if crop_end < length else crop_end)

    span = block[1] - block[0]
    overhead = 0 if n == 1 else r if n == 2 else 2 * r
    forced = AxisSolution(n=n, base=sol.base, last=span - (n - 1) * sol.base, overhead=overhead, r=r)
    return block, region, forced


def _check_block_crops(tiles, block, layout, first_col, first_row, drop_left, drop_top):
    # A caller must never sample the wrong pixels silently, so every block crop is checked against
    # the parent's before the sub layout is handed back. On a dropped-ring side the block crop sits
    # exactly r px inside the parent's.
    for tile in tiles:
        parent = layout.tiles[(first_row + tile.row) * layout.sol_x.n + first_col + tile.col]
        shifted = (tile.crop_rect.x0 + block.x0, tile.crop_rect.y0 + block.y0,
                   tile.crop_rect.x1 + block.x0, tile.crop_rect.y1 + block.y0)
        parent_crop = (parent.crop_rect.x0, parent.crop_rect.y0, parent.crop_rect.x1, parent.crop_rect.y1)
        want = (parent_crop[0] + (layout.r if drop_left and tile.col == 0 else 0),
                parent_crop[1] + (layout.r if drop_top and tile.row == 0 else 0),
                parent_crop[2], parent_crop[3])
        if shifted != want:
            raise RuntimeError(
                f"block tile col {tile.col} row {tile.row} samples {shifted} instead of {want}, "
                f"against parent tile col {parent.col} row {parent.row} crop {parent_crop}"
            )


def sub_layout(layout, col0, col1, row0, row1):
    # The layout of one rectangular block of a parent layout's tiles, given the inclusive column
    # and row bounds neighborhood returns.
    sol_x, sol_y = layout.sol_x, layout.sol_y
    if not (0 <= col0 <= col1 < sol_x.n) or not (0 <= row0 <= row1 < sol_y.n):
        raise ValueError(
            f"block range cols {col0}..{col1} rows {row0}..{row1} is not inside the "
            f"{sol_x.n} by {sol_y.n} tile grid"
        )

    first = layout.tiles[row0 * sol_x.n + col0]
    last_col = layout.tiles[row0 * sol_x.n + col1]
    last_row = layout.tiles[row1 * sol_x.n + col0]
    (bx0, bx1), (rx0, rx1), sx = _sub_axis(
        sol_x, col0, col1, _axis_spans(first, last_col, "width"), layout.w, layout.ctx, layout.overlap, "width")
    (by0, by1), (ry0, ry1), sy = _sub_axis(
        sol_y, row0, row1, _axis_spans(first, last_row, "height"), layout.h, layout.ctx, layout.overlap, "height")

    block = Rect(x0=bx0, y0=by0, x1=bx1, y1=by1)
    region = Rect(x0=rx0, y0=ry0, x1=rx1, y1=ry1)
    block_layout = build_layout(bx1 - bx0, by1 - by0, sx, sy, layout.ctx, layout.overlap)

    _check_block_crops(block_layout.tiles, block, layout, col0, row0,
                       drop_left=col0 > 0 and sx.n >= 2, drop_top=row0 > 0 and sy.n >= 2)
    return SubLayout(block=block, region=region, layout=block_layout,
                     first_col=col0, first_row=row0, parent_tiles=layout.tiles)
