Source: https://github.com/exx8/differential-diffusion (README.md, and SD2/diff_pipe.py lines 568-596 and 675-711)
Repo: exx8/differential-diffusion, branch `main` (official reference implementation for arXiv:2306.00950)
Retrieved: 2026-07-22

Note: this is the official code released alongside the Differential Diffusion paper. Captured here
is only the pipeline signature and the concrete threshold/release mechanism (the actual runnable
equivalent of the paper's Algorithm 1) from the Stable Diffusion 2 pipeline
(`SD2/diff_pipe.py`); install/usage instructions and the SDXL/Kandinsky/IF variants (same pattern)
are out of scope for this doc set.

## README abstract

> Diffusion models have revolutionized image generation and editing, producing state-of-the-art results in conditioned and unconditioned image synthesis. While current techniques enable user control over the degree of change in an image edit, the controllability is limited to global changes over an entire edited region. This paper introduces a novel framework that enables customization of the amount of change *per pixel* or *per image region*. Our framework can be integrated into any existing diffusion model, enhancing it with this capability. Such granular control on the quantity of change opens up a diverse array of new editing capabilities, such as control of the extent to which individual objects are modified, or the ability to introduce gradual spatial changes. Furthermore, we showcase the framework's effectiveness in soft-inpainting---the completion of portions of an image while subtly adjusting the surrounding areas to ensure seamless integration. Additionally, we introduce a new tool for exploring the effects of different change quantities. Our framework operates solely during inference, requiring no model training or fine-tuning. We demonstrate our method with the current open state-of-the-art models, and validate it via both quantitative and qualitative comparisons, and a user study.

## `SD2/diff_pipe.py` — pipeline signature (lines 568-586)

```python
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        strength: float = 1,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        map:torch.FloatTensor = None,
    ):
```

Docstring note for `strength`: "Repealed in favor of the map."

## `SD2/diff_pipe.py` — the threshold/release mechanism (lines 675-711)

```python
        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        map = torchvision.transforms.Resize(tuple(s // self.vae_scale_factor for s in image.shape[2:]),antialias=None)(map)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # prepartions
        original_with_noise = self.prepare_latents(
            image, timesteps, batch_size, num_images_per_prompt, prompt_embeds.dtype, device, generator
        )
        thresholds = torch.arange(len(timesteps), dtype=map.dtype) / len(timesteps)
        thresholds = thresholds.unsqueeze(1).unsqueeze(1).to(device)
        masks = map > thresholds
        # end diff diff preparations

        with self.progress_bar(total=num_inference_steps) as progress_bar:

            for i, t in enumerate(timesteps):
                # diff diff
                if i == 0:
                    latents = original_with_noise[:1]
                else:
                    mask = masks[i].unsqueeze(0)
                    # cast mask to the same type as latents etc
                    mask = mask.to(latents.dtype)
                    mask = mask.unsqueeze(1)  # fit shape
                    latents = original_with_noise[i] * mask + latents * (1 - mask)
                    # end diff diff
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
```

This is the concrete, runnable form of the paper's Algorithm 1: `map` is resized to latent resolution once; `thresholds` is the per-timestep-index fraction `i/len(timesteps)` broadcast over the spatial dims; `masks = map > thresholds` precomputes, for every timestep, the boolean per-pixel release mask (`map` here plays the role of the paper's down-sampled change map `μ_s`, and a pixel is "released" — driven from its own noised-original rather than the running denoise chain — once the map value exceeds the current step's threshold). Each loop iteration then does the same convex mix as Algorithm 1 line 9: `latents = original_with_noise[i] * mask + latents * (1 - mask)`.

Same pattern (resize map → build per-step `thresholds`/`masks` → convex-mix `original_with_noise` and the running `latents` every step) is repeated in `SDXL/diff_pipe.py`, `Kandinsky/diff_pipe.py`, and `IF/diff_pipe_I.py` / `IF/diff_pipe_II.py` in the same repo.
