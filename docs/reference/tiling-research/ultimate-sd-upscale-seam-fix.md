Source: https://github.com/Coyote-A/ultimate-upscale-for-automatic1111 (README.md; scripts/ultimate-upscale.py), branch `master`
Repo: Coyote-A/ultimate-upscale-for-automatic1111 — "Ultimate SD Upscale" extension for AUTOMATIC1111's stable-diffusion-webui
Retrieved: 2026-07-22

Note: captures only the tile redraw modes and seam-fix mechanism (mask-blur + gradient-mask
compositing passes) — not the full extension manual (upscaler selection, UI wiring, gradio
layout code). The ComfyUI port is referenced by the README as
https://github.com/ssitu/ComfyUI_UltimateSDUpscale but not itself fetched here.

## README

Now you have the opportunity to use a large denoise (0.3-0.5) and not spawn many artifacts. Works on any video card, since you can use a 512x512 tile size and the image will converge.

### API Usage

```javascript
{
"script_name" : "ultimate sd upscale",
"script_args" : [
	null, // _ (not used)
	512, // tile_width
	512, // tile_height
	8, // mask_blur
	32, // padding
	64, // seams_fix_width
	0.35, // seams_fix_denoise
	32, // seams_fix_padding
	0, // upscaler_index
	true, // save_upscaled_image a.k.a Upscaled
	0, // redraw_mode
	false, // save_seams_fix_image a.k.a Seams fix
	8, // seams_fix_mask_blur
	0, // seams_fix_type
	0, // target_size_type
	2048, // custom_width
	2048, // custom_height
	2 // custom_scale
]
}
```

redraw_mode
| Value         |  |
|:-------------:| -----:|
| 0 | Linear |
| 1 | Chess |
| 2 | None |

seams_fix_type
| Value         |  |
|:-------------:| -----:|
| 0 | None |
| 1 | Band pass |
| 2 | Half tile offset pass |
| 3 | Half tile offset pass + intersections |

target_size_type
| Value         |  |
|:-------------:| -----:|
| 0 | From img2img2 settings |
| 1 | Custom size |
| 2 | Scale from image size |

## `scripts/ultimate-upscale.py` — modes (lines 12-21)

```python
class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2

class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3
```

## Redraw pass (`USDURedraw`, lines 152-247)

Grid tiles are drawn with a hard rectangular mask (full white on the current tile, full black
elsewhere) through the normal img2img inpainting path — the "seam" avoidance for the redraw pass
itself comes from `mask_blur` (a webui `StableDiffusionProcessing` parameter that Gaussian-blurs
the inpainting mask edge before compositing, feathering the tile boundary) plus tile `padding`
(extra context pulled in around the inpainted region via `inpaint_full_res_padding`), not from a
gradient mask:

```python
class USDURedraw():

    def init_draw(self, p, width, height):
        p.inpaint_full_res = True
        p.inpaint_full_res_padding = self.padding
        p.width = math.ceil((self.tile_width+self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height+self.padding) / 64) * 64
        mask = Image.new("L", (width, height), "black")
        draw = ImageDraw.Draw(mask)
        return mask, draw

    def calc_rectangle(self, xi, yi):
        x1 = xi * self.tile_width
        y1 = yi * self.tile_height
        x2 = xi * self.tile_width + self.tile_width
        y2 = yi * self.tile_height + self.tile_height

        return x1, y1, x2, y2

    def linear_process(self, p, image, rows, cols):
        mask, draw = self.init_draw(p, image.width, image.height)
        for yi in range(rows):
            for xi in range(cols):
                if state.interrupted:
                    break
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if (len(processed.images) > 0):
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p, image, rows, cols):
        self.initial_info = None
        if self.mode == USDUMode.LINEAR:
            return self.linear_process(p, image, rows, cols)
        if self.mode == USDUMode.CHESS:
            return self.chess_process(p, image, rows, cols)
```

`chess_process` (same file, lines 191-240) runs the same per-tile rectangle-mask inpaint, but
processes tiles in a checkerboard order (all tiles of one color first, then the other) so that,
by the time each second-pass tile is redrawn, all four of its orthogonal neighbors have already
been redrawn — reducing visible grid seams from the redraw pass before the dedicated seams-fix
pass runs at all.

## Seams-fix pass (`USDUSeamsFix`, lines 249-416)

This pass runs *after* the full grid has been redrawn, and repairs the boundaries directly with
gradient-mask compositing (this is the actual "mask blur + seam-fix modes" mechanism):

```python
class USDUSeamsFix():

    def init_draw(self, p):
        self.initial_info = None
        p.width = math.ceil((self.tile_width+self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height+self.padding) / 64) * 64

    def half_tile_process(self, p, image, rows, cols):

        self.init_draw(p)
        processed = None

        gradient = Image.linear_gradient("L")
        row_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        row_gradient.paste(gradient.resize(
            (self.tile_width, self.tile_height//2), resample=Image.BICUBIC), (0, 0))
        row_gradient.paste(gradient.rotate(180).resize(
                (self.tile_width, self.tile_height//2), resample=Image.BICUBIC),
                (0, self.tile_height//2))
        col_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        col_gradient.paste(gradient.rotate(90).resize(
            (self.tile_width//2, self.tile_height), resample=Image.BICUBIC), (0, 0))
        col_gradient.paste(gradient.rotate(270).resize(
            (self.tile_width//2, self.tile_height), resample=Image.BICUBIC), (self.tile_width//2, 0))

        p.denoising_strength = self.denoise
        p.mask_blur = self.mask_blur

        for yi in range(rows-1):
            for xi in range(cols):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(row_gradient, (xi*self.tile_width, yi*self.tile_height + self.tile_height//2))

                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                if (len(processed.images) > 0):
                    image = processed.images[0]

        for yi in range(rows):
            for xi in range(cols-1):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(col_gradient, (xi*self.tile_width+self.tile_width//2, yi*self.tile_height))

                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                if (len(processed.images) > 0):
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def half_tile_process_corners(self, p, image, rows, cols):
        fixed_image = self.half_tile_process(p, image, rows, cols)
        processed = None
        self.init_draw(p)
        gradient = Image.radial_gradient("L").resize(
            (self.tile_width, self.tile_height), resample=Image.BICUBIC)
        gradient = ImageOps.invert(gradient)
        p.denoising_strength = self.denoise
        #p.mask_blur = 0
        p.mask_blur = self.mask_blur

        for yi in range(rows-1):
            for xi in range(cols-1):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = 0
                mask = Image.new("L", (fixed_image.width, fixed_image.height), "black")
                mask.paste(gradient, (xi*self.tile_width + self.tile_width//2,
                                      yi*self.tile_height + self.tile_height//2))

                p.init_images = [fixed_image]
                p.image_mask = mask
                processed = processing.process_images(p)
                if (len(processed.images) > 0):
                    fixed_image = processed.images[0]

        p.width = fixed_image.width
        p.height = fixed_image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return fixed_image

    def band_pass_process(self, p, image, cols, rows):

        self.init_draw(p)
        processed = None

        p.denoising_strength = self.denoise
        p.mask_blur = 0

        gradient = Image.linear_gradient("L")
        mirror_gradient = Image.new("L", (256, 256), "black")
        mirror_gradient.paste(gradient.resize((256, 128), resample=Image.BICUBIC), (0, 0))
        mirror_gradient.paste(gradient.rotate(180).resize((256, 128), resample=Image.BICUBIC), (0, 128))

        row_gradient = mirror_gradient.resize((image.width, self.width), resample=Image.BICUBIC)
        col_gradient = mirror_gradient.rotate(90).resize((self.width, image.height), resample=Image.BICUBIC)

        for xi in range(1, rows):
            if state.interrupted:
                    break
            p.width = self.width + self.padding * 2
            p.height = image.height
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = self.padding
            mask = Image.new("L", (image.width, image.height), "black")
            mask.paste(col_gradient, (xi * self.tile_width - self.width // 2, 0))

            p.init_images = [image]
            p.image_mask = mask
            processed = processing.process_images(p)
            if (len(processed.images) > 0):
                image = processed.images[0]
        for yi in range(1, cols):
            if state.interrupted:
                    break
            p.width = image.width
            p.height = self.width + self.padding * 2
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = self.padding
            mask = Image.new("L", (image.width, image.height), "black")
            mask.paste(row_gradient, (0, yi * self.tile_height - self.width // 2))

            p.init_images = [image]
            p.image_mask = mask
            processed = processing.process_images(p)
            if (len(processed.images) > 0):
                image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p, image, rows, cols):
        if USDUSFMode(self.mode) == USDUSFMode.BAND_PASS:
            return self.band_pass_process(p, image, rows, cols)
        elif USDUSFMode(self.mode) == USDUSFMode.HALF_TILE:
            return self.half_tile_process(p, image, rows, cols)
        elif USDUSFMode(self.mode) == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            return self.half_tile_process_corners(p, image, rows, cols)
        else:
            return image
```

Mechanism summary:
- **HALF_TILE** (`half_tile_process`): builds a `tile_width × tile_height` gradient mask that is
  black at the tile centers and ramps to white at the seam (`Image.linear_gradient` mirrored
  top/bottom for horizontal seams, left/right for vertical seams via 90°/270° rotation), then
  inpaints a `tile_width × tile_height` window centered on each row-seam and each col-seam in turn
  — i.e. it re-diffuses a window straddling the boundary between two already-redrawn tiles, with
  the gradient mask feathering how much of that window gets changed vs. kept, at `self.denoise`
  strength and `self.mask_blur` blur.
- **HALF_TILE_PLUS_INTERSECTIONS** (`half_tile_process_corners`): runs `half_tile_process` first,
  then does one more pass centered on every four-tile *intersection* point using an inverted
  radial gradient (white at the center point, black at the tile edges) — fixing the corner where
  four tiles meet, which the row/col seam passes don't directly touch.
- **BAND_PASS** (`band_pass_process`): instead of a full `tile_width × tile_height` window, only
  a narrow `self.width`-pixel-wide strip straddling each seam is inpainted (`p.width/height` set
  to strip width, not tile size), using a mirrored linear gradient so the strip is white (fully
  denoised) at its center — right on the seam line — and fades to black at both edges; explicitly
  sets `p.mask_blur = 0` since the gradient mask already provides the feather.
- **`mask_blur`** (redraw pass) and **`seams_fix_mask_blur`** (seams-fix pass, only meaningful for
  HALF_TILE / HALF_TILE_PLUS_INTERSECTIONS per the README's `seams_fix_mask_blur` table) both flow
  into `p.mask_blur`, the webui inpainting parameter that Gaussian-blurs the composite mask edge —
  orthogonal to, and stackable with, the gradient masks above.
