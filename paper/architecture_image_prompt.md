# OCT-SSRet — Hybrid ResNet-18 + Mamba-Style Selective State-Space Bottleneck for OCT Disease Classification

## Image-generation prompt (paste verbatim into DALL-E 3 / Midjourney / ChatGPT-Image)

> A clean, publication-quality neural-network architecture diagram for a top-tier medical-imaging journal (target style: IEEE Transactions on Medical Imaging / Medical Image Analysis). White background, flat-color rounded-rectangle blocks with thin 1px dark-grey borders, no shadows, no gradients, no 3D. Sans-serif labels (DejaVu Sans). Strict left-to-right flow in five vertical bands. Avoid crossing arrows.
>
> **Title at top, centered, bold (14 pt):** "OCT-SSRet: ResNet-18 + Mamba-Style Bidirectional Selective State-Space Bottleneck for OCT Retinal-Disease Classification"
>
> **Band 1 — Input (far left):** Small grayscale OCT B-scan thumbnail (the layered horizontal banding of the retinal cross-section visible). Caption below: "OCT B-scan x ∈ ℝ^(B×3×256×256)". Below that, a faint pre-processing label: "FOV crop + letterbox-resize + light CLAHE on green channel".
>
> **Band 2 — ResNet-18 backbone:**
> - Light-blue block "ResNet-18 Encoder (ImageNet-1K pretrained, 11.18M params)" with five horizontal stripes labelled stem (/2, 64ch), layer1 (/4, 64ch), layer2 (/8, 128ch), layer3 (/16, 256ch), layer4 (/32, 512ch).
> - Output tensor labelled: "feat_map ∈ ℝ^(B×512×8×8)". An arrow drops down 'flatten + transpose' producing "tokens ∈ ℝ^(B×64×512)" — the input to the bottleneck.
>
> **Band 3 — Mamba-style Bidirectional Selective SSM Bottleneck (the headline contribution):**
> - Large green block titled "**Mamba-style Bidirectional Selective SSM (pure-PyTorch, 2 layers, 3.9M params)**" with a tiny inset diagram showing two arrows: one labelled "→ forward scan", one labelled "← reverse scan", merging via a "+" symbol gated by a SiLU branch.
> - Inside the block, three small annotations stacked (matching the methodology section):
>   - "Δ, B, C ← f(input)   (input-dependent SSM parameters)"
>   - "h_t = exp(Δ A) h_{t-1} + Δ B · u_t"
>   - "y_t = C · h_t + D · u_t"
> - Two input arrows enter the SSM block: solid arrow from "tokens" labelled "raster sequence (length 64)", and a second arrow labelled "tokens (reversed)" implied by the bidirectional design.
> - Output tensor labelled "tokens_ssm ∈ ℝ^(B×64×512)" — bidirectional state-space-aware tokens.
>
> **Band 4 — Pool + Dropout + Classification head:**
> - Light-grey block "Mean-Pool + LayerNorm" producing "pooled ∈ ℝ^(B×512)".
> - Light-grey "Dropout(p=0.3)" block.
> - Light-blue block "Linear(512 → 4) + softmax" producing "p_cls ∈ ℝ^(B×4)".
>
> **Band 5 — Loss + output (far right):**
> - Light-red loss box: "L = class-balanced focal cross-entropy (β=0.999, γ=2.0)" — solid arrow from p_cls.
> - Below, an amber output box: "Predicted disease class ∈ {CNV, DME, DRUSEN, NORMAL}".
> - To the right of the output box, a small tan annotation strip:
>   "**Empirical operating profile:** in-domain neutral on Kermany 2017 (matches ResNet-18 baseline within McNemar p=0.12); +0.035 cross-cohort accuracy on the hard-mapped OCTID subset; +0.033–0.041 OOD-detection AUROC across MSP/entropy/energy; lower ECE, NLL, Brier."
>
> **Bottom legend (small text, light grey):**
> - Solid dark-grey arrow → "forward + backward (gradient flows)"
> - **Green box = NEW Mamba-style bidirectional selective state-space bottleneck (the headline contribution)**
> - Light-blue box = pretrained ResNet-18 backbone / classification head
> - Light-red box = loss; Amber box = output
> - Annotation strip beneath the diagram: "Pure-PyTorch SSM implementation (no custom CUDA), portable across GPUs including Blackwell sm_120 where the official mamba-ssm kernels are not yet shipped."
>
> **Color palette (muted pastels, fixed):** ResNet/classifier #dbeafe, **SSM #86efac (slightly bolder green to highlight as the new contribution)**, pool/dropout #cbd5e1, loss #fecaca, output #fcd34d, annotation #fde6c8. Block titles 12 pt bold; tensor shapes 9 pt regular monospace. Aspect ratio 16:9, sized for a journal double-column figure spanning full text width.

---

## Refinement follow-ups

- "the green Mamba SSM block must be visibly the largest/most prominent block — it is the new contribution"
- "show two arrows entering the SSM block: forward scan and backward scan, both from the tokens tensor"
- "the ResNet-18 backbone must show five horizontal stages with output shapes visible"
- "remove all shadows, gradients, and 3D effects — flat vector style only"
- "keep the architecture readable at the size of a journal column figure"

---

## Drop-in instructions

Save the generated PNG/PDF as `paper/paper_latex/figures/fig_architecture.pdf` (overwriting the matplotlib backup version). The paper picks it up via `\includegraphics[width=0.95\textwidth]{fig_architecture.pdf}`.
