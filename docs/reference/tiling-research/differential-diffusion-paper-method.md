Source: https://arxiv.org/abs/2306.00950 (HTML render via https://ar5iv.labs.arxiv.org/html/2306.00950)
Paper: "Differential Diffusion: Giving Each Pixel Its Strength" — Eran Levin, Ohad Fried (arXiv:2306.00950v2, cs.CV)
Retrieved: 2026-07-22

Note: this file captures only the Method section (the per-pixel threshold/release mechanism),
not the full paper (Related Work, Results/comparisons, Conclusion, References, and Appendix are
omitted as out of scope for this doc set). Math is reproduced from the paper's own LaTeX source
(embedded in the ar5iv HTML as TeX annotations); this is a verbatim capture with HTML→Markdown
formatting only, no wording changes. One symbol required a typesetting substitution — noted inline
where it occurs.

## Abstract

Text-based image editing has advanced significantly in recent years. With the rise of diffusion models, image editing via textual instructions has become ubiquitous. Unfortunately, current models lack the ability to customize the *quantity* of the change per pixel or *per image fragment*, resorting to changing the entire image in an equal amount, or editing a specific region using a binary mask.
In this paper, we suggest a new framework which enables the user to customize the quantity of change for each image fragment,
thereby enhancing the flexibility and verbosity of modern diffusion models. Our framework does not require model training or fine-tuning, but instead performs everything at inference time, making it easily applicable to an existing model.
We show both qualitatively and quantitatively that our method allows better controllability and can produce results which are unattainable by existing models. Our code is available at: https://github.com/exx8/differential-diffusion.

## 3. Method

Given an image, a mono-channel change map representing the strength of editing in each pixel, and a text prompt to guide the edit, we aim to edit the image.
We want the generated result to satisfy the strength constraints, be photo-realistic, and adhere to the prompt.

### 3.1. Overview

Diffusion models are deep machine learning models that have been inspired by thermodynamics (Sohl-Dickstein et al., 2015).
In computer vision context, they are usually trained gradually to denoise images, that have been corrupted by a random Gaussian noise.
Usually the image-to-image translation process
(**"the inference process"**)
begins with an image with added Gaussian noise,
then in a repetitive process the noise is gradually removed.
This inference process creates a series of images, where each is the result of the denoising process of the previous one (**"the inference chain"**).

The **prompt** is a text that describes the content of the generated segments.
**Strength** is a parameter that guides the amount of change in an image. See: Figure 5.
For more details, see Saharia et al. (2022).
Usually there is a single strength parameter for the entire image.
In this work we allow different strengths for each fragment of the image, via the **change map**, or simply "the map" — a tensor with the height and width of the original image, with values between 0 and 1, representing the strength to apply to each pixel.
In contrast to previous works (Avrahami et al., 2022b, a), we denote a complete change as "black" (0) and not "white" (1).

In Latent Diffusion Models (Rombach et al., 2022b), the **latent encoder** is a neural network that compresses an image into a smaller latent space, in which we apply the diffusion process.
At the end of the process, a **latent decoder** decompress the latent output into an image.

In this work, we present *Differential Diffusion* — an enhancement of image-to-image diffusion models that adds the ability to
control the amount of change applied to each image fragment via a change map (Algorithm 1).
The method only changes the inference process.

```
Algorithm 1  Differential Image to Image Diffusion

Input:  x (image to edit), k (number of steps), μ (change map between 0 and 1), p (prompt)
Output: x̂

1:  procedure INFERENCE(x, k, μ, p)
2:      z_init = ldm_encode(x)
3:      μ_s = down_sample(μ)
4:      z'_k = add_noise(z_init, k)
5:      z_k = denoise(z'_k, p, k)
6:      for t = k-1 to 0 do
7:          z'_t = add_noise(z_init, t)
8:          mask = μ_s ≥ (t / k)
9:          z_t^mix = z_{t+1} ⊙ mask + z'_t ⊙ (1 - mask)
10:         z_t = denoise(z_t^mix, p, t)
11:     end for
12:     x̂ = ldm_decode(z_0)
13:     return x̂
14: end procedure
```

> Typesetting note: in the source, the comparison operator on line 8 is rendered as a custom
> circled-≥ glyph (a LaTeX macro drawing "≥" inside a circle, parallel to how ⊙ circles the
> multiplication dot). It is typeset above as plain ≥; the paper's own description of the operator
> (below) is reproduced verbatim.

Whereas ≥, ⊙ are element-wise larger-than and element-wise multiplication, respectively. ≥ returns a tensor of 1s and 0s.

Figure 2. In this figure we show the intermediate steps of Differential Diffusion inference process which uses "injection" in every time-step. As one can see, the injected content is indistinguishable from non-injected fragments as expected.
Here the maximal strength is 30% and the lowest is 10%. The prompt is "statues of men". We used the technique described in Section 3.5 to expand the prompt.

Figure 3.
On this figure, you can see the decomposition of the inference process of Differential Diffusion:
Top:
$z_{t+1}\odot mask$, which corresponds to the injected fragments at each time-step.
bottom:
$z_{t}^{\prime}\odot(1-mask)$ which corresponds to the fragments which are not injected at this time-step.
Zero values are indicated in brown.
As you can see, the low-strength fragments (the light ones) are injected across many time-steps, in contrast to the high-strength (dark) ones who are injected only in early and limited time-steps.

### 3.2. Observations

For designing the algorithm we made three key-observations:

(1) Given two series of indices of time-steps $(100,95,...)$, of two inference process, A and B, created by the same parameters, up to the strength parameter; without loss of generality, let the inference process of A be with a higher strength than B. Then $B\subseteq A$.
Furthermore $\exists i\geq 0(\forall j\geq 0(B[j]=A[i+j]))$. Meaning, B is a truncated series of A.

(2) One can insert a fragment of a picture, which is encoded to the latent space, into an intermediate step $t$ of the chain, without breaking the inference process, by inserting the fragment with noise that corresponds to $t$. In Figure 2 we show the effects of such insertion on intermediate steps via our Algorithm 1. We will call this an injection.

(3) The latent encoder that we use (Stable Diffusion (Rombach et al., 2022a)) generally encodes pixels to the same relative positions, meaning for non-miniscule shapes in the picture, one can estimate which position they will be in the latent-tensor, just by calculating their relative positions. We call this property locality.

Our algorithm is composed of the following techniques:

### 3.3. Change Map Down-Sampling

The map is down-sampled to match the size of the latent-space encoding, by height and width (the same map is applied to all the channels), as the diffusion process is preformed in the latent-space, and not in the pixel space. Due to locality, the down-sampled map matches the position of the latent pixel in the latent tensor.

### 3.4. Fragment Injection

We define a new operation called injection — the process of replacing fragments in the latent space.
As we replace fragments in the noised latent space, we must inject segments that are encoded and noised according to the current time-step.

#### 3.4.1. Gradual Injection

We inject the fragments in a time-step and noise that match their user-specified change map.
Fragments with the smallest strength value are added last,
and with the least amount of noise.
By doing this, we utilize the gradual image generation process:

(1) The later the final injection, the greater similarity of this fragment on the output to the input. As expected, because the fragment is changed by fewer iterations, the fewer changes are applied to this fragment.

(2) The amount of noise added to the injected fragment is proportional to the number of steps that are to be applied to this fragment. The more steps a fragment will take part in,
the more noise is added. This is a key change from the regular diffusion inference process, in which all the fragments are noised at the same time, and therefore should not be noised differently.

#### 3.4.2. Future Hinting

It is tempting to assume, that it is enough to inject each fragment once, according to its designated strength. This might look reasonable as in most image-to-image translation procedures all the fragments are inserted on the first time-step.
It also coincides with the observations made in Section 3.4.1, that lower-strength-fragments will be injected later.
However, Throughout the new inference process, we re-inject some of the fragments repeatedly.
On any time step $k$, we inject the fragments that correspond to k and the fragments that correspond to the future time-steps (Figure 3).
We called this behavior future hinting. This serves a few goals:

(1) We give the diffusion model advance knowledge of some of the upcoming visual data, thereby allowing it to "plan" more complex objects.
Meaning, we use that the model make decisions according the content of the entire picture.

(2) Usually, diffusion models are trained on pictures that mostly contain no holes. Therefore, they are expected to be less robust to diffusion processes with blank pixels in their intermediate diffusion steps.
Future hinting is a natural way to fill such holes with appropriate data.

We conducted a comparison experiment
between our full method and a version without future hinting, meaning that
for each time-step, only the fragments that match the exact time-step were injected (change 8 in Algorithm 1 to `mask = (t+1)/k > μ_s ≥ t/k`).
The experiment demonstrates that future hinting improves result quality (Figure 6).

> Typesetting note: as in Algorithm 1, the source renders both comparison operators here as
> custom circled glyphs (circled-">" and circled-"≥"); typeset above as plain `>` and `≥`. The
> ablation restricts injection to pixels whose down-sampled change-map value falls in the
> half-open bin `[t/k, (t+1)/k)` for the current step only, instead of the full algorithm's
> cumulative `μ_s ≥ t/k` (which keeps injecting every fragment due at this step or later —
> the "future hinting" behavior).
