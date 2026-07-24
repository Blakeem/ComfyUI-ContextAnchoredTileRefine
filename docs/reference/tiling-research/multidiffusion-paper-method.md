Source: https://arxiv.org/abs/2302.08113 (HTML render via https://ar5iv.labs.arxiv.org/html/2302.08113)
Paper: "MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation" (ICML 2023) — Omer Bar-Tal, Lior Yariv, Yaron Lipman, Tali Dekel (arXiv:2302.08113)
Retrieved: 2026-07-22

Note: this file captures only the Method section and the Panorama / Region-based-generation
applications (the seam-blending mechanism: the least-squares fusion objective and its closed-form
weighted-average solution), not the full paper (Introduction, Related Work, Results/comparisons,
Discussion, Acknowledgments, References, and Appendices are omitted as out of scope for this doc
set). Math is reproduced from the paper's own LaTeX source (embedded in the ar5iv HTML as TeX
annotations); this is a verbatim capture with HTML→Markdown formatting only, no wording changes.

## 3 Method

We consider a pre-trained diffusion model, which serves as a reference model:

$$\Phi:{\mathcal{I}}\times{\mathcal{Y}}\rightarrow{\mathcal{I}}$$

working in image space ${\mathcal{I}}=\mathbb{R}^{H\times W\times C}$ and condition space ${\mathcal{Y}}$, e.g., $y\in{\mathcal{Y}}$ is a text prompt. Initializing $I_{T}\sim P_{\mathcal{I}}$, where $P_{\mathcal{I}}$ represents the distribution of Gaussian i.i.d. pixel values, and setting a condition $y\in{\mathcal{Y}}$, the diffusion model builds a sequence of images,

$$I_{T},I_{T-1},\ldots,I_{0}\quad\text{s.t.}\quad I_{t-1}=\Phi(I_{t}|y)\tag{1}$$

gradually transforming the noisy image $I_{T}$ into a clean image $I_{0}$.

**MultiDiffusion.** Our goal is to leverage $\Phi$ to generate images in a potentially different image space ${\mathcal{J}}=\mathbb{R}^{H^{\prime}\times W^{\prime}\times C}$ and condition space ${\mathcal{Z}}$, without any training or finetuning. To do so, we define a MultiDiffusion process, defined by a function, called MultiDiffuser,

$$\Psi:{\mathcal{J}}\times{\mathcal{Z}}\rightarrow{\mathcal{J}}$$

The MultiDiffusion, similarly to a diffusion process, starts with some initial noisy input $J_{T}\sim P_{\mathcal{J}}$, where $P_{\mathcal{J}}$ is a noise distribution over ${\mathcal{J}}$, and produces a series of images

$$J_{T},J_{T-1},\ldots,J_{0}\quad\text{s.t.}\quad J_{t-1}=\Psi(J_{t}|z)\tag{2}$$

Our key idea is to define $\Psi$ to be as-consistent-as-possible with $\Phi$. More specifically, we define a set of mappings between the target and reference image spaces $F_{i}:{\mathcal{J}}\rightarrow{\mathcal{I}}$, and a corresponding set of mappings between the condition spaces: $\lambda_{i}:{\mathcal{Z}}\rightarrow{\mathcal{Y}}$ where $i\in[n]=\{1,\ldots,n\}$. These mappings are application depended, as will be described later in Sec. 4. Our goal is to make every MultiDiffuser step $J_{t-1}=\Psi(J_{t}|z)$ follow as closely as possible $\Phi(I^{i}_{t}|y_{i})$, $i\in[n]$, i.e., the denoising steps of $\Phi$ when applied to the images and conditions:

$$I^{i}_{t}=F_{i}(J_{t}),\quad y_{i}=\lambda_{i}(z)$$

Formally, our new process is given by solving the following optimization problem:

$$\Psi(J_{t}|z)=\operatorname*{arg\,min}_{J\in{\mathcal{J}}}\ \ {\mathcal{L}}_{\text{FTD}}(J|J_{t},z)\tag{3}$$

$${\mathcal{L}}_{\text{FTD}}(J|J_{t},z)=\sum_{i=1}^{n}\Big{\|}W_{i}\otimes\Big{[}F_{i}(J)-\Phi(I^{i}_{t}|y_{i})\Big{]}\Big{\|}^{2}\tag{4}$$

where $W_{i}\in\mathbb{R}^{H\times W}_{+}$ are per pixel weights and $\otimes$ is the Hadamard product.
Intuitively, the FTD loss reconciles, in the least-squares sense, the different denoising sampling steps, $\Phi(I^{i}_{t}|y_{i})$, suggested on different regions, $F_{i}(J_{t})$, of the generated image $J_{t}$. Fig. 2 illustrates one step of the MultiDiffuser; Algorithm 2 recaps the MultiDiffusion sampling process.

**Closed-form formula.** In the applications demonstrated in this paper $F_{i}$ consist of direct pixel samples (e.g., taking a crop out of image $J_{t}$). In this case, Eq. 4 is a quadratic Least-Squares (LS) where each pixel of the minimizer $J$ is a weighted average of all its diffusion sample updates, i.e.,

$$\Psi(J_{t}|z)=\sum_{i=1}^{n}\frac{F_{i}^{-1}(W_{i})}{\sum_{j=1}^{n}F_{j}^{-1}(W_{j})}\otimes F_{i}^{-1}(\Phi(I^{i}_{t}|y_{i}))\tag{5}$$

**Properties of MultiDiffusion.** The main motivation for the definition of $\Psi$ in Eq. 3 comes from the following observation: If we choose a probability distribution $P_{\mathcal{J}}$ such that

$$F_{i}(J_{T})\sim P_{\mathcal{I}},\qquad\forall i\in[n]\tag{6}$$

and compute $J_{t-1}=\Psi(J_{t}|z)$, as defined in Eq. 3, where we reach a zero FTC loss, ${\mathcal{L}}_{\text{FTD}}(J_{t-1}|J_{t},z)=0$, then:

$$I^{i}_{t-1}=F_{i}(J_{t})=\Phi(I^{i}_{t}|y_{i})$$

That is, $I^{i}_{t}$, for all $i\in[n]$, is a diffusion sequence and thus $I^{i}_{0}$ is distributed according to the distribution defined by $\Phi$ over the image space ${\mathcal{I}}$. We summarize

**Proposition 3.1.** If $P_{\mathcal{J}}$ is a distribution over ${\mathcal{J}}$ satisfying Eq. 6,
and the FTD cost (Eq. 4) is minimized to zero in Eq. 3 for all steps $T,T-1,\ldots,0$, then the images $I^{i}_{t}=F_{i}(J_{t})$ reproduce a $\Phi$ diffusion path. In particular $F_{i}(J_{0})$, $i\in[n]$ are distributed identically to samples from the reference diffusion model $\Phi$.

The implications of this proposition are far reaching: using a single reference diffusion process we can flexibly adapt to different image generation scenarios without the need to retrain the model, while still being consistent with the reference diffusion model. Next, we instantiate this framework outlining several application of the Follow-the-Diffusion-Paths approach.

**Algorithm 1 MultiDiffusion sampling.**

```
Input:
  Φ            ▷ pre-trained Diffusion Model
  {F_i}_{i=1}^n   ▷ image space mappings
  {y_i}_{i=1}^n    ▷ text-prompts conditioning
  {W_i}_{i=1}^n  ▷ per-pixel weights

J_T ~ P_J    ▷ noise initialization
for t = T, ..., 1 do
      I_{t-1}^i ← Φ(F_i(J_t), y_i)  ∀i ∈ [n]    ▷ diffusion updates
      J_{t-1} ← MultiDiffuser({I_{t-1}^i}_{i=1}^n)           ▷ Eq. 5
Output: J_0
```

## 4 Applications

### 4.1 Panorama

As a first instantiation we use our framework to define a diffusion model in an image space ${\mathcal{J}}$ with $H^{\prime}\geq H$, $W^{\prime}\geq H$ directly from a trained model $\Phi$ working in image space ${\mathcal{I}}$. Let ${\mathcal{Z}}={\mathcal{Y}}$ (namely, generating a panoramic image for a given text-prompt), $F_{i}(J)\in{\mathcal{I}}$ is an $H\times W$ crop of image $J$, and $z=\lambda_{i}(z)$. We consider $n$ such crops that cover the original images $J$. Setting $W_{i}=\mathbf{1}$, we get

$$\Psi(J_{t},z)=\operatorname*{arg\,min}_{J\in{\mathcal{J}}}\ \ \sum_{i=1}^{n}\left\|F_{i}(J)-\Phi(F_{i}(J),z)\right\|^{2}\tag{7}$$

that is a least-squares problem, the solution of which is calculated analytically according to Eq. 5. See the Appendix B.1 for implementation details.

As discussed in Sec. 3, MultiDiffusion reconciles multiple diffusion paths provided by the reference model $\Phi$. We illustrate this property in Fig. 3, where we consider a panorama of $H\times 4W$. Fig. 3(a) shows the generation result when independently applying $\Phi$ on four non-overlapping crops. As expected, there is no coherency between the crops since this amounts to four random samples from the model. Starting from the same initial noise, our generation process (Eq. 7), allows us to fuse these initially-unrelated diffusion paths, and steer the generation into a high-quality, coherent panorama (b).

### 4.2 Region-based text-to-image-generation

Given a set of region-masks $\{M_{i}\}_{i=1}^{n}\subset\{0,1\}^{H\times W}$ and a corresponding set of text-prompts $\{y_{i}\}_{i=1}^{n}\subset{\mathcal{Y}}^{n}$, our goal is to generate a high-quality image $I\in{\mathcal{I}}$ that depicts the desired content in each region. That is, the image segment $I\otimes M_{i}$ should manifest $y_{i}$. Going back to our formulation (Eq. 2), the MultiDiffusion process is defined over the condition space ${\mathcal{Z}}={\mathcal{Y}}^{n}$, i.e., $z=(y_{1},\ldots,y_{n})$, and the target image space ${\mathcal{J}}={\mathcal{I}}$ is identical to the reference one:

$$\Psi:{\mathcal{I}}\times{\mathcal{Y}}^{n}\rightarrow{\mathcal{I}}$$

Furthermore, the region selection maps are defined as $F_{i}(I)=I$, the pixel weights are set according to the masks, $W_{i}=M_{i}$, and the $\Psi$ step is defined as the solution to the least-squares problem:

$$\Psi(J_{t},z)=\operatorname*{arg\,min}_{J\in{\mathcal{I}}}\ \ \sum_{i=1}^{n}\Big{\|}M_{i}\otimes\Big{[}J-\Phi(J_{t}|y_{i})\Big{]}\Big{\|}^{2}\tag{8}$$

The solution to this LS problem is calculated analytically. At each step we apply the pretrained diffusion w.r.t. each of the given prompts, resulting in multiple diffusion directions $\Phi(J_{t}|y_{i})$. We encourage each pixel in $J_{t}$ to follow the (averaged) directions associated with the regions $M_{i}$ containing it (Eq. 5).

**Fidelity to tight masks.** We further support obtaining high-fidelity to tight masks if provided by the user (see Fig. 5).
We noticed that the layout is being determined early on in the diffusion process, and thus we strive to encourage $\Phi(J_{t}|y_{i})$ to focus on the region $M_{i}$ early on in the process in order to match the desired layout, and to consider the full context in the image next, to achieve an harmonized result. We integrate time dependency in the maps $F_{i}$, introducing a bootstrapping phase. That is,

$$F_{i}(J_{t},t)=\begin{cases}J_{t},&\text{if }t\leq T_{init}\\ M_{i}\otimes J_{t}+(1-M_{i})\otimes S_{t},&\text{otherwise}\end{cases}\tag{9}$$

Where $T_{init}$ is the bootstrapping stopping step parameter, and $S_{t}$ is a random image with a constant color, which serves as background (see Appendix B.2 for implementation details).

We demonstrate the efficiency of our bootstrapping approach in Sec. 5.2. We set $T_{init}$ to be $20\%$ of the generation process (i.e., $T_{init}=800$).
