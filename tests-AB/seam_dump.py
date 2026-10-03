"""Seam debug dump of a base node run: every tile's intermediates as PNGs, plus a manifest.

    with seam_dump(out_dir):
        sampling.refine_image(...)   # no vl_clip, so the raster path runs

Each tile writes t{index:02d}_r{row}c{col}_<part>.png:
    enc             the crop it encoded: the raw diffused region inside its live anchor ring
    decoded         its decoded crop, before the DC match
    alpha           the feather alpha over paste_rect
    region_before   the canvas under paste_rect before the composite
    region_after    the canvas under paste_rect after the composite
A tile whose crop equals its core (a 1x1 grid, or context_anchor and context_overlap both 0)
pastes hard and writes `decoded` only.
After the last tile the dump writes canvas_final.png and manifest.txt. canvas_final.png is what
_refine_tiles returns, which on the mask path is the refined bbox crop. The manifest holds one
line per tile with its rects and the applied DC offset. The offset is recorded there because
the PNGs hold the uncorrected decode.

Every _refine_tiles call rewrites the same file names. A batched IMAGE calls it once per
picture, so the last picture's files are the ones left.

The shipped package carries no debug hook. _refine_tiles resolves grid.solve_layout and every
sampling function wrapped here through its module at call time, so the wrappers see each
tile's tensors in raster order. Run ab_env.bootstrap() first, as for every harness module.
"""
import contextlib
import os

import torch
from PIL import Image


def _save_png(path, tensor):
    """Batch row 0 as an 8-bit PNG: [B,H,W,C] and [H,W,C] as RGB, [H,W] as grayscale."""
    pixels = tensor.detach().to("cpu", torch.float32)
    if pixels.ndim == 4:
        pixels = pixels[0]
    if pixels.ndim == 3:
        pixels = pixels[..., :3]
    Image.fromarray((pixels.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).contiguous().numpy()).save(path)


def _rect_text(rect):
    return f"({rect.x0},{rect.y0})-({rect.x1},{rect.y1})"


def _manifest_line(tile_idx, tile, dc_offset):
    dc_text = ""
    if dc_offset is not None:
        values = ",".join(f"{value:+.6f}" for value in dc_offset.tolist())
        dc_text = f" dc=[{values}]"
    return (f"t{tile_idx:02d} r{tile.row}c{tile.col} cls={tile.cls} kept_top={tile.kept_top} "
            f"kept_left={tile.kept_left} core={_rect_text(tile.core)} paste={_rect_text(tile.paste_rect)} "
            f"crop={_rect_text(tile.crop_rect)}{dc_text}")


class _TileRun:
    """One _refine_tiles call. The seam_dump wrappers report to it in the raster loop's call order."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.layout = None
        self.tile_idx = -1
        self.dc_offset = None
        self.region = None
        self.manifest = []

    def _tile(self):
        return self.layout.tiles[self.tile_idx]

    def save(self, part, tensor):
        tile = self._tile()
        _save_png(os.path.join(self.out_dir, f"t{self.tile_idx:02d}_r{tile.row}c{tile.col}_{part}.png"), tensor)

    def start_tile(self, pixels):
        # _refine_tiles encodes once per tile in layout order, so the encode count names the tile.
        self._finish_tile()
        if self.layout is None:
            raise RuntimeError("seam_dump saw a tile encode before any layout solve. _refine_tiles no longer "
                               "calls grid.solve_layout through the module, so the dump cannot name its tiles.")
        if self.tile_idx + 1 >= len(self.layout.tiles):
            raise RuntimeError(f"seam_dump counted more tile encodes than the {len(self.layout.tiles)} tile layout "
                               "holds. _refine_tiles no longer encodes once per tile.")
        self.tile_idx += 1
        self.dc_offset = None
        self.region = None
        tile = self._tile()
        crop, core = tile.crop_rect, tile.core
        if (crop.x0, crop.y0, crop.x1, crop.y1) != (core.x0, core.y0, core.x1, core.y1):
            self.save("enc", pixels)

    def record_dc(self, tile, dc_offset):
        if tile is not self._tile():
            raise RuntimeError(f"seam_dump counted tile {self.tile_idx} but seam_dc_offset received tile "
                               f"r{tile.row}c{tile.col}. _refine_tiles no longer encodes once per tile in layout order.")
        self.dc_offset = dc_offset

    def record_region(self, region):
        # region is a view into the live canvas. The next tile's encode precedes its composite, so
        # _finish_tile reads this tile's composite through the same view.
        self.save("region_before", region)
        self.region = region

    def _finish_tile(self):
        if self.tile_idx < 0:
            return
        if self.region is not None:
            self.save("region_after", self.region)
        self.manifest.append(_manifest_line(self.tile_idx, self._tile(), self.dc_offset))

    def finish(self, result):
        self._finish_tile()
        _save_png(os.path.join(self.out_dir, "canvas_final.png"), result)
        with open(os.path.join(self.out_dir, "manifest.txt"), "w", encoding="utf-8") as manifest_file:
            manifest_file.write("\n".join(self.manifest) + "\n")


@contextlib.contextmanager
def seam_dump(out_dir):
    """Dump every _refine_tiles call inside the block into out_dir.

    Raises after the block when no _refine_tiles call ran, since a VL run goes through sync.py
    and would otherwise leave an empty folder."""
    from context_anchored_tile_refine import grid, sampling

    run = None
    calls = 0
    refine_tiles = sampling._refine_tiles
    solve_layout = grid.solve_layout
    encode_pixels = sampling.encode_pixels
    fold_decoded_frames = sampling.fold_decoded_frames
    seam_dc_offset = sampling.seam_dc_offset
    seam_displacements = sampling.seam_displacements
    feather_alpha = sampling.feather_alpha

    def dumped_refine_tiles(*args, **kwargs):
        nonlocal run, calls
        calls += 1
        run = _TileRun(out_dir)
        try:
            result = refine_tiles(*args, **kwargs)
            run.finish(result)
            return result
        finally:
            run = None

    def dumped_solve_layout(*args, **kwargs):
        layout = solve_layout(*args, **kwargs)
        if run is not None:
            run.layout = layout
        return layout

    def dumped_encode_pixels(vae, pixels):
        if run is not None:
            run.start_tile(pixels)
        return encode_pixels(vae, pixels)

    def dumped_fold_decoded_frames(decoded):
        folded = fold_decoded_frames(decoded)
        if run is not None:
            run.save("decoded", folded)
        return folded

    def dumped_seam_dc_offset(tile, sub, region, region_mask=None):
        dc_offset = seam_dc_offset(tile, sub, region, region_mask)
        if run is not None:
            run.record_dc(tile, dc_offset)
        return dc_offset

    def dumped_seam_displacements(tile, sub, region):
        if run is not None:
            run.record_region(region)
        return seam_displacements(tile, sub, region)

    def dumped_feather_alpha(*args, **kwargs):
        alpha = feather_alpha(*args, **kwargs)
        if run is not None:
            run.save("alpha", alpha)
        return alpha

    hooks = (
        (sampling, "_refine_tiles", refine_tiles, dumped_refine_tiles),
        (grid, "solve_layout", solve_layout, dumped_solve_layout),
        (sampling, "encode_pixels", encode_pixels, dumped_encode_pixels),
        (sampling, "fold_decoded_frames", fold_decoded_frames, dumped_fold_decoded_frames),
        (sampling, "seam_dc_offset", seam_dc_offset, dumped_seam_dc_offset),
        (sampling, "seam_displacements", seam_displacements, dumped_seam_displacements),
        (sampling, "feather_alpha", feather_alpha, dumped_feather_alpha),
    )
    os.makedirs(out_dir, exist_ok=True)
    try:
        for module, name, _original, hook in hooks:
            setattr(module, name, hook)
        yield
    finally:
        for module, name, original, _hook in hooks:
            setattr(module, name, original)
    if calls == 0:
        raise RuntimeError(f"seam_dump wrote nothing to {out_dir}: no _refine_tiles call ran inside the block. "
                           "Only the base node's raster path is dumped.")
