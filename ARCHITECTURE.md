# Diffusion OCR Architecture Draft

## Goal

Build a document-scan-to-text model that combines:

1. A high-compression visual encoder for page images.
2. A text decoder that can run in both autoregressive (AR) mode and masked diffusion mode.
3. Attention masks that allow rich interaction inside one diffusion canvas, and between canvas tokens and visual tokens, while preventing attention between different diffusion canvases.
4. A practical inference path where diffusion drafts text blocks and AR decoding verifies, repairs, or continues the result.

The target use case is OCR and document parsing, not generic captioning. The model should preserve literal text, reading order, layout-sensitive structures, tables, formulas, and low-frequency tokens such as IDs, names, and numbers.

## Background Assumptions

DeepSeek-OCR is a useful reference point because it shows that aggressive visual-token compression can work for OCR-like decoding. Its reported result is strong below roughly 10 text tokens per visual token, but quality drops at higher compression. A later analysis argues that very compressed visual tokens can increase reliance on language priors, which is especially dangerous for OCR because hallucinating plausible text is worse than producing uncertain text.

Gemma-style discrete diffusion language models are relevant because they predict masked/noisy text spans with bidirectional attention. However, a pure diffusion decoder is not automatically a drop-in replacement for AR OCR: it may commit tokens in non-obvious orders, and exact string fidelity can suffer if the model leans on priors.

Nemotron-TwoTower is the closest architectural reference for the dual-mode idea. It uses an AR context tower for clean causal context and a diffusion denoiser tower for noisy blocks. For this project, the visual encoder can play part of the "context" role, and the text decoder should optionally run as an AR verifier/refiner.

References:

- DeepSeek-OCR: Contexts Optical Compression: https://arxiv.org/abs/2510.18234
- Visual Merit or Linguistic Crutch? A Close Look at DeepSeek-OCR: https://arxiv.org/abs/2601.03714
- Nemotron-TwoTower: Diffusion Language Modeling with Pretrained Autoregressive Context: https://arxiv.org/abs/2606.26493
- DiffusionVL: Translating Any Autoregressive Models into Diffusion Vision Language Models: https://arxiv.org/abs/2512.15713
- SpecDiff-2: Scaling Diffusion Drafter Alignment For Faster Speculative Decoding: https://arxiv.org/abs/2511.00606

## Proposed High-Level Architecture

```text
page image
  |
  v
visual preprocessor
  - deskew, crop, normalize
  - optional layout patches / tiles
  |
  v
compressed visual encoder
  - local high-res stem
  - token compressor / resampler
  - optional multi-scale tokens
  |
  v
visual memory V = [v_1 ... v_M]
  |
  +-----------------------+
                          |
                          v
dual-mode text decoder: shared or partially shared weights
  - AR mode: causal self-attention over generated text + cross-attention to V
  - diffusion mode: bidirectional attention inside one canvas + cross-attention to V
  - no attention between different canvases
  |
  v
text, layout tags, confidence, optional bounding links
```

The recommended first experiment is not a fully novel foundation model. Start with a small AR language decoder that already tokenizes your target scripts well, then adapt it with a diffusion denoising objective and visual cross-attention. This gives a better chance of preserving language competence while testing the attention and cache idea.

## Visual Encoder

### Requirements

The visual encoder should compress a page into a small number of tokens while retaining OCR-critical details:

- character shapes,
- line and paragraph structure,
- table/grid structure,
- reading order cues,
- small punctuation,
- numeric strings,
- low-contrast or degraded text.

For a first prototype, target these token budgets:

| Page type | Visual tokens per page | Notes |
| --- | ---: | --- |
| simple printed page | 256 to 512 | useful baseline |
| dense page | 512 to 1024 | safer for OCR fidelity |
| tables/forms | 1024 to 2048 | structure needs more tokens |
| experimental extreme compression | 100 to 256 | likely hallucination-prone |

Avoid optimizing only for average edit distance. Track numeric accuracy, rare-word accuracy, table-cell accuracy, and hallucination rate separately.

### Candidate Encoder Design

Use a multi-stage encoder:

1. High-resolution patch stem:
   - Input resolution: page resized to a bounded long side, plus optional tiled crops.
   - Patch size: 8x8 or 16x16.
   - Local conv or ViT/Swin-style early layers to preserve fine glyph detail.

2. Layout-aware intermediate tokens:
   - Keep a 2D token grid before compression.
   - Add 2D position embeddings.
   - Optionally add scale/page-region embeddings.

3. Token compressor:
   - Perceiver resampler, cross-attention pooling, token merging, or learned queries.
   - Output fixed or semi-fixed `M` visual memory tokens.
   - Recommended: dynamic token budget based on text density, with an upper cap.

4. Optional OCR-specialized auxiliary heads:
   - text density map,
   - line segmentation,
   - reading-order graph,
   - token saliency,
   - character/word-level CTC head for regularization.

The auxiliary heads are not required at inference, but they may reduce the decoder's temptation to invent plausible text.

## Text Decoder Modes

The decoder should support two execution modes with a shared backbone where possible.

### AR Mode

AR mode is the reliability path.

Inputs:

- already generated clean text tokens,
- optional prompt/control tokens,
- visual memory tokens.

Mask:

- causal self-attention over text,
- full cross-attention from text tokens to visual tokens,
- no causal restriction on visual tokens because they are encoder outputs.

Use cases:

- final verification/refinement,
- long output continuation beyond a canvas,
- fallback on uncertain diffusion spans,
- teacher model during training.

### Diffusion Mode

Diffusion mode generates or repairs a fixed-length text canvas in parallel/iteratively.

Inputs:

- a text canvas of length `B`,
- some clean tokens,
- some masked/noisy tokens,
- timestep/noise embedding,
- optional AR prefix context,
- visual memory tokens.

Mask:

- bidirectional self-attention inside the same canvas,
- cross-attention to visual memory,
- optional cross-attention to AR prefix context,
- no attention to other active canvases.

Use cases:

- draft line, paragraph, region, or page chunks,
- fill uncertain spans,
- speculative drafting before AR verification,
- iterative repair of layout-sensitive regions.

## Canvas Attention Design

Let the model process multiple canvases in one batch:

```text
visual tokens: V = [v_1 ... v_M]

canvas 1: C1 = [c_1 ... c_B]
canvas 2: C2 = [c_1 ... c_B]
canvas 3: C3 = [c_1 ... c_B]
```

Desired diffusion attention:

```text
allowed:
  C_i token -> tokens inside C_i
  C_i token -> all or selected visual tokens V
  C_i token -> optional clean AR prefix/context

blocked:
  C_i token -> C_j token, where i != j
```

This is a block-diagonal text attention mask plus shared visual cross-attention:

```text
             V      C1     C2     C3
C1 queries   yes    yes    no     no
C2 queries   yes    no     yes    no
C3 queries   yes    no     no     yes
```

Implementation options:

1. Separate cross-attention layers:
   - Self-attention is block diagonal over canvases.
   - Cross-attention uses shared visual K/V.
   - Easiest to reason about and cache.

2. Unified attention sequence:
   - Concatenate visual tokens and all canvas tokens.
   - Use a custom block mask.
   - More flexible, but harder to optimize.

Recommendation: use separate self-attention and cross-attention first. It makes the constant-KV argument cleaner and avoids accidental canvas-to-canvas leakage.

## KV Cache and Complexity

The user's core hypothesis is that omitting attention between different diffusion canvases keeps KV size approximately constant.

This is partly true, with an important caveat:

- The reusable visual K/V cache is constant per page: `O(M * L * H * D)`.
- Each canvas still needs temporary K/V for its own block: `O(B * L * H * D)`.
- With `N` canvases processed concurrently, raw compute and temporary activation memory can still scale with `N`, even if there is no cross-canvas K/V dependency.

The useful property is not that all memory is constant. The useful property is that persistent context cache does not grow with the number of canvases, and canvases can be processed independently, streamed, or microbatched.

Approximate per-layer attention cost:

```text
self attention inside canvases:    O(N * B^2)
cross attention to visual tokens:  O(N * B * M)
visual cache storage:              O(M)
temporary canvas K/V:              O(N * B), or O(B) if streamed one canvas at a time
```

If `B` is fixed and canvases are streamed, runtime scales with page length but persistent KV remains bounded by the visual memory and current canvas.

## Decoder Architecture Choices

### Option A: Single Backbone, Two Masks

One transformer decoder supports both:

- causal mask for AR,
- bidirectional block mask for diffusion.

Pros:

- parameter efficient,
- easiest to keep AR and diffusion vocab/logits aligned,
- direct conversion from an AR checkpoint is possible.

Cons:

- objectives can interfere,
- diffusion wants bidirectional denoising behavior, while AR wants strict next-token prediction,
- timestep conditioning must be added carefully.

This is the recommended initial route.

### Option B: Two-Tower Decoder

Use:

- AR context tower: frozen or mostly frozen, causal.
- diffusion denoiser tower: trainable, bidirectional inside canvas, cross-attends to visual memory and AR context.

Pros:

- closer to Nemotron-TwoTower,
- cleaner separation of context and denoising roles,
- AR competence is preserved.

Cons:

- more memory and parameters,
- more engineering complexity,
- less direct weight sharing.

This is attractive if the single-backbone version shows objective conflict.

### Option C: Diffusion Drafter Plus Independent AR Verifier

Use two models:

- a smaller diffusion OCR drafter,
- a stronger AR OCR verifier/refiner.

Pros:

- easiest to prototype,
- verifier can reject bad drafts,
- compatible with speculative decoding style experiments.

Cons:

- not a single decoder,
- alignment between drafter and verifier can be hard,
- speedup depends on high draft acceptance.

This is a good baseline even if the long-term goal is one dual-mode decoder.

## Training Objectives

Train with a mixture of objectives.

### 1. Visual-to-Text AR Loss

Standard teacher-forced OCR:

```text
P(y_t | y_<t, V)
```

Purpose:

- preserves exact text modeling,
- gives a reliable decoding path,
- creates a verifier/refiner.

### 2. Masked Diffusion Denoising Loss

Sample a canvas span and corrupt tokens with a noise schedule:

```text
x_t = corrupt(y_span, timestep=t)
model predicts y_span from x_t, t, V, optional prefix/suffix
```

Use discrete diffusion with `[MASK]` or absorbing-state corruption first. Continuous embedding diffusion is possible later but complicates decoding.

Recommended schedules:

- random mask ratio from low to high,
- span-aware masking for OCR words/lines,
- structure-aware masking for table cells,
- high corruption for draft generation,
- low corruption for repair/refinement.

### 3. AR Refinement Loss

Feed imperfect diffusion drafts to AR mode and train it to produce corrected text:

```text
draft = noisy_or_model_generated_text
target = clean_text
```

The AR decoder can be conditioned with special tokens such as:

```text
<ocr_verify>
<draft>
...
</draft>
<final>
```

### 4. Consistency Loss Between Modes

Encourage AR and diffusion logits to agree on clean or lightly corrupted positions:

```text
KL(logits_diffusion || logits_AR)
```

Do not over-weight this. The two modes have different conditioning and should not be forced to be identical everywhere.

### 5. Visual Grounding Losses

Optional but recommended:

- align text tokens to visual regions,
- predict bounding boxes for words/lines,
- contrastive loss between text spans and visual patches,
- CTC auxiliary loss on line crops.

These losses directly attack the main risk: language-prior hallucination.

## Inference Recipes

### Recipe 1: AR-Only Baseline

Use the visual encoder and AR decoder only.

Purpose:

- measure maximum reliable OCR quality,
- establish a comparison for diffusion speed and quality.

### Recipe 2: Diffusion-Only Canvas Decoding

Split output into canvases:

- fixed token blocks,
- layout regions,
- lines,
- paragraphs,
- table cells.

For each canvas:

1. initialize with `[MASK]` tokens,
2. denoise for `K` steps,
3. commit high-confidence tokens,
4. repeat or stop early.

This mode should expose whether the visual tokens alone are sufficient.

### Recipe 3: Diffusion Draft + AR Refinement

1. Visual encoder produces `V`.
2. Diffusion decoder drafts canvases independently.
3. AR decoder verifies left-to-right.
4. AR decoder accepts, edits, or regenerates uncertain spans.

This is the strongest practical path.

Confidence signals for AR refinement:

- low max probability,
- high entropy,
- disagreement across diffusion samples,
- weak visual attention mass,
- suspicious language-model-only confidence,
- mismatch with optional CTC/line recognizer.

### Recipe 4: Speculative OCR

The diffusion model proposes `B` tokens. The AR model verifies them.

For exact lossless speculative decoding, the verifier must be able to compute acceptance probabilities under the same target distribution. That may be difficult in multimodal OCR with separate diffusion proposals. A practical approximation is still useful, but should be labeled as approximate refinement rather than lossless speculative decoding.

## Document Structure Handling

Output should not be plain text only. Use a structured target format during training:

```text
<page>
<block bbox="...">
<line bbox="..."> text </line>
<table>
<row><cell>...</cell></row>
</table>
</block>
</page>
```

For the first prototype, avoid verbose XML if token budget is tight. A compact markdown-like format may be enough:

```text
[block x1 y1 x2 y2]
line text

| col 1 | col 2 |
| ...   | ...   |
```

The canvas unit should match document structure when possible:

- line canvas for dense OCR,
- table-cell canvas for tables,
- paragraph canvas for prose,
- region canvas for forms.

This reduces the chance that independent canvases produce inconsistent reading order.

## Proposed Experiment Plan

### Phase 0: Baselines

- Run an existing OCR system on the dataset.
- Run a standard VLM/OCR model if available.
- Measure CER, WER, table metrics, numeric exact match, and hallucination rate.

### Phase 1: Visual Encoder + AR Decoder

- Train or fine-tune visual encoder plus AR decoder.
- Use moderate visual token counts: 512 to 1024 per page.
- Add CTC or line-recognition auxiliary loss if possible.

Exit criteria:

- competitive OCR quality,
- acceptable rare-token and numeric accuracy,
- no severe hallucination under semantic perturbation.

### Phase 2: Add Diffusion Mode

- Add timestep embeddings and bidirectional canvas masks.
- Train denoising on ground-truth spans.
- Keep AR objective active.
- Compare single-backbone and two-tower variants if resources allow.

Exit criteria:

- diffusion drafts are better than an AR-free language prior,
- quality improves with visual tokens and degrades when visual tokens are shuffled/occluded,
- canvas isolation does not damage layout consistency too much.

### Phase 3: Draft + Refine

- Generate diffusion drafts.
- Train AR refinement on generated drafts, not only synthetic noise.
- Add uncertainty-triggered repair.

Exit criteria:

- same or better quality than AR baseline,
- lower latency or higher throughput,
- bounded persistent KV cache,
- measurable acceptance/repair rate.

### Phase 4: Compression Stress Test

Vary visual tokens:

```text
M = 128, 256, 512, 1024, 2048
```

Track:

- CER/WER,
- numeric exact match,
- rare-token exact match,
- table structure F1,
- hallucination under corrupted/non-word documents,
- attention concentration and visual-token usage,
- speed and memory.

The key question is not "can it decode compressed pages?" but "when does compression turn OCR into plausible reconstruction?"

## Evaluation

Use multiple evaluation slices:

- clean printed pages,
- scans with blur/noise,
- handwriting if in scope,
- receipts/forms,
- tables,
- math,
- multilingual text,
- rare names and IDs,
- synthetic random strings,
- semantic nonsense text,
- adversarial fonts.

Metrics:

- character error rate,
- word error rate,
- normalized edit distance,
- exact match for numbers/dates/IDs,
- table TEDS or cell-level F1,
- reading order accuracy,
- hallucinated token rate,
- omission rate,
- confidence calibration,
- latency per page,
- peak memory,
- visual tokens per output token.

A crucial ablation is semantic nonsense text. If the model performs well on normal prose but collapses on random strings or corrupted sentences, it is relying too much on language priors.

## Critique and Risks

### Risk 1: Compression Can Hide OCR Evidence

Very small visual-token budgets may not preserve enough glyph-level evidence. A decoder can still output fluent text, but that may be reconstruction, not OCR.

Mitigation:

- use dynamic token budgets,
- keep high-resolution local tokens for dense regions,
- add auxiliary visual grounding losses,
- evaluate on random strings and semantically corrupted documents.

### Risk 2: Canvas Independence Can Break Global Consistency

Blocking attention between canvases helps cache behavior, but pages have global dependencies:

- reading order,
- repeated headers,
- table alignment,
- footnotes,
- hyphenation across lines,
- multi-line formulas.

Mitigation:

- allow a small global context: layout tokens, AR prefix summaries, or document-plan tokens,
- use AR refinement after diffusion,
- choose canvases by layout region rather than arbitrary fixed blocks.

### Risk 3: Diffusion May Not Improve OCR Quality

Diffusion can draft many tokens in parallel, but OCR is often bottlenecked by visual evidence and exact copying, not only text decoding speed.

Mitigation:

- treat diffusion as a speed/throughput experiment,
- keep AR-only baseline strong,
- measure quality at equal latency, not only raw decoding steps.

### Risk 4: Dual-Mode Training Conflict

One backbone may struggle to serve both causal next-token prediction and bidirectional denoising.

Mitigation:

- use mode-specific adapters or LoRA,
- add separate attention norms,
- freeze early layers during diffusion adaptation,
- move to two-tower design if interference is visible.

### Risk 5: Approximate Speculative Refinement Is Not Lossless

If diffusion drafts are verified by an AR decoder, this is not automatically equivalent to exact speculative decoding.

Mitigation:

- be explicit whether refinement is exact or approximate,
- report accepted-token rate and correction rate,
- compare final output to AR-only quality.

## Recommended Initial Configuration

For the first serious prototype:

- Visual encoder: high-resolution ViT or hybrid Conv/ViT with Perceiver-style resampler.
- Visual tokens: start with 768 or 1024 per page, then compress later.
- Decoder: 1B to 3B AR language model adapted for OCR.
- Attention: separate text self-attention and visual cross-attention.
- Diffusion: absorbing-mask discrete diffusion over fixed canvases of 128 or 256 tokens.
- Canvas mask: block diagonal across canvases.
- Modes:
  - AR OCR baseline,
  - diffusion draft,
  - diffusion draft plus AR refinement.
- Training:
  - AR OCR loss,
  - diffusion denoising loss,
  - generated-draft refinement loss,
  - optional CTC/layout auxiliary losses.

## Open Research Questions

1. What is the best canvas unit: fixed token block, line, paragraph, table cell, or layout region?
2. How many visual tokens are needed before OCR stops relying on language priors?
3. Can visual cross-attention be sparse or routed per canvas without losing fidelity?
4. Does a single dual-mode decoder work, or is a two-tower split necessary?
5. Can AR refinement be made efficient enough that diffusion drafting improves end-to-end latency?
6. How should uncertainty from diffusion steps be calibrated for OCR-specific errors?
7. Can the model output alignments/bounding boxes without sacrificing text accuracy?

## Bottom Line

The idea is plausible and worth testing, especially as a diffusion-drafter plus AR-refiner system. The constant-KV claim should be stated carefully: visual context cache can remain constant, and canvas-local temporary state can be bounded by streaming, but total compute still scales with the number of canvases.

The biggest technical risk is not attention masking. It is whether high visual compression preserves enough evidence for exact OCR. The experiment should therefore be designed around falsifying language-prior hallucination early, before investing heavily in larger diffusion decoders.
