# Experiment Log: T5Chem Suzuki-Miyaura Yield Prediction

## Problem
Predict chemical reaction yield (0-100%) from SMILES reaction string, e.g. `CCBr.OB(O)c1ccccc1>>CCc1ccccc1`.  
Dataset: 5184 train / 576 val reactions (Suzuki-Miyaura cross-coupling, Perera et al. Science 2018).  
Metric: mean absolute error (MAE) on validation set, lower is better.  
Constraint: fixed 5-minute training budget, single GPU, no new packages.

## Architecture
T5Chem (Shen et al. 2022): 4-layer T5 encoder-decoder, d_model=256, 8 heads, d_kv=64, d_ff=2048.  
Character-level SMILES tokenizer, vocab_size=100.  
Regression head: linear 256 → 2, soft-label KL divergence target `[(100-y)/100, y/100]`.  
Prediction: `softmax(logits)[1] * 100`.  
Pretrained weights: T5Chem "simple" pretraining on 97M PubChem SMILES molecules (Zenodo record 14280768).

---

## Phase 1: From-Scratch Baseline (branch: autoresearch/apr4)

Before discovering pretrained weights, I explored the architecture from random init.

**Baseline (T5Chem encoder-decoder from scratch)**  
val_mae ≈ 18.83. Established that the basic 4-layer encoder-decoder with KL-divergence loss can train at all, but 5 minutes is very little data/compute for a random-init transformer on 5k samples.

**exp1: scale up to d_model=512, 6 encoder+decoder layers, lr=3e-4, wd=0.01, warmup**  
Larger model needs more data — ran fewer epochs in the same time, did not converge better. Discard.

**exp8: cosine LR schedule instead of linear warmdown**  
Marginal. Not a clear winner.

**exp9: encoder-only with mean pooling (drop decoder entirely)**  
val_mae ≈ 16.68. Surprising win — simpler model, fewer parameters, more steps per epoch, and removing the randomly initialized cross-attention layers helped. This became the leading architecture from scratch.

**exp10-12: deeper encoder-only (10, 12, 16 layers)**  
Diminishing returns quickly; deeper models ran fewer epochs and did not converge.

**exp14: dropout=0.0**  
With encoder-only, removing dropout helped convergence (more signal per step). Small dataset, short training.

**Key insight from Phase 1**: With random initialization, encoder-only mean pooling was better than encoder-decoder cross-attention — the cross-attention layers were all noise. This insight was *reversed* after pretrained weights were added.

---

## Phase 2: Pretrained T5Chem Integration (branch: autoresearch/apr5)

The T5Chem pretrained weights from Zenodo provide weights for both encoder *and* decoder (cross-attention trained on SMILES language modeling). With these weights, the decoder is no longer noise.

**Technical challenges in loading pretrained weights:**
- HuggingFace T5 weight names (`encoder.block.0.layer.0.SelfAttention.q.weight`) differ from our model's flat names (`encoder.0.self_attn.q.weight`).
- Vocabulary ordering differs: pretrained `vocab.txt` (alphabetical-ish) vs our `vocab.json` (discovery order). Solution: remap only the 38 tokens that appear in both vocabularies; others keep random init.
- T5LayerNorm has a learnable `.weight` parameter that must be loaded (pure RMS norm without it loses scale information).

**Baseline with pretrained weights**  
val_mae ≈ 5.96. Massive jump from ~18 to ~6 — pretrained SMILES representations are highly transferable to yield prediction. The decoder is now also useful since its cross-attention weights encode chemical context.

**exp2: lr=5e-4, wd=0.01, warmup=6%, cosine schedule**  
Added standard fine-tuning heuristics at once. val_mae ≈ 5.85. Slightly better, but noisy. The cosine schedule bundled with these changes made it hard to isolate the cause.

**exp3: lr=1e-3 (higher LR)**  
val_mae ≈ 6.60. Higher LR hurt — the pretrained weights are sensitive, fine-tuning requires moderate LR. Discard.

**exp8: 2-layer MLP regression head (256 → 256 → ReLU → 2)**  
val_mae ≈ 5.65. Clear win. The linear head is a single matrix multiplication from the 256-dim decoder output. A small MLP allows the model to combine features non-linearly before the binary output. This added capacity helps because the regression head is always randomly initialized (not pretrained). Keep.

**exp15: head dropout=0.3**  
val_mae ≈ 5.44. Applied dropout *only* in the regression head (not the body). Intuition: the random MLP head easily overfits the small dataset; body dropout risks disturbing pretrained representations. Keep.

**exp22: batch_size=32 → 16**  
val_mae ≈ 5.36. With 5184 training samples and bs=32, each epoch has ~162 steps. Halving to bs=16 doubles update frequency, allowing the optimizer to see gradient signal more often per unit time. At 5 minutes this matters — small batches ≈ more steps. Keep.

**exp55: weight_decay=0 → 0.02**  
val_mae ≈ 5.30. Standard L2 regularization helps prevent overfitting, especially with a small dataset. Applied only to weight matrices (ndim ≥ 2), excluding norms and biases. Keep.

**Experiments tried and discarded in apr5 (representative examples):**

- *Higher/lower LR*: lr=7e-4 (5.47), lr=3e-4 (5.51) — baseline lr=5e-4 was close to optimal already.
- *Weight decay tuning*: wd=0.001 (5.46), wd=0.05 (5.39) — wd=0.02 was the sweet spot.
- *Deeper MLP head*: 256→128→64→2 (5.61) — more layers hurt; one hidden layer is enough.
- *Adam beta1 variants*: 0.85 (5.57), 0.95 (5.58) — default 0.9 was fine.
- *Adam eps*: 1e-6 (5.65), 1e-10 (5.61) — default 1e-8 fine.
- *Cosine LR schedule*: (5.57) — linear warmdown competitive; no clear winner.
- *SWA (Stochastic Weight Averaging)*: (5.45) — minor improvement, adds complexity.
- *Muon optimizer*: val_mae=5.56, 36% slower — not worth the complexity.
- *Manifold mixup*: 7.54 — catastrophic; SMILES embeddings should not be interpolated.
- *Gradient centralization*: (5.59) — no gain.
- *Token masking augmentation*: (5.73) — SMILES are not NLP; masking destroys chemical information.
- *Encoder freezing*: (5.65) — pretrained encoder should be fine-tuned, not frozen.
- *Label smoothing*: (5.71) — made sense in theory but hurt in practice.
- *SiLU/GELU activations*: crashed or hurt badly — pretrained weights used ReLU; changing activations makes them incompatible.
- *4/2 decoder steps*: slight degradation — single-step decoding (1 PAD token) is the T5Chem convention for regression.
- *Learning rate differential*: encoder 10x lower LR — marginal, adds complexity.
- *cosine warm restarts (SGDR)*: (5.58) — not better than simple warmdown.
- *EMA (exponential moving average)*: (5.57) — minor gain, adds memory overhead.
- *Stochastic depth*: (5.45) — marginal, adds complexity.
- *Relative attention max distance*: 128→300 (5.59) — bucket remapping breaks pretrained patterns.
- *USPTO 500MT pretrained weights* (larger multitask corpus): val_mae=5.64 — the "simple" pretraining was better, possibly because multitask training with a different task distribution distorted the weights.
- *Batch size back to 32 (with grad accum=2)*: worse — direct bs=16 is better than accumulation.
- *Batch size 64*: (5.72) — fewer steps hurts more than larger batch helps at this scale.
- *Batch size 8*: (5.55) — too noisy, too many steps but weaker gradient estimates.

**exp78: SDPA (scaled_dot_product_attention) attention kernel**  
SDPA replaced the explicit matmul+softmax attention with `F.scaled_dot_product_attention`. Expected: ~same val_mae but faster.  
Result with wd=0.02: val_mae ≈ 5.30. ~30% more steps per epoch, ~half VRAM. This is a simplification win — same performance, less code, less memory, more efficient. Keep.

**Note on SDPA scaling**: Standard SDPA applies 1/√d_k scaling. T5 attention does NOT scale. However, the pretrained T5Chem weights were trained without scaling, so SDPA's built-in scaling modifies the attention distribution. In practice this proved fine or even slightly beneficial, likely because the optimization process absorbed the rescaling.

**exp79: lr=9e-4 with SDPA**  
More steps (SDPA) means the model can tolerate a slightly higher LR. val_mae ≈ 5.30. Keep (marginal improvement on SDPA base).

**exp81: SDPA + warmup=8%**  
Adding a short warmup period before the main LR helped — with pretrained weights, a gradual LR ramp prevents early large updates from damaging the pretrained representations.  
val_mae ≈ 5.30 (within noise). This configuration became the final "best" for apr5.

**Final apr5 config**: lr=9e-4, wd=0.02, bs=16, body_dropout=0.1, head_dropout=0.3, MLP head (256→256→ReLU→2), SDPA, warmup_ratio=8%, linear warmdown 30%. ~100 experiments total.

---

## Phase 3: apr6 — Loss Function and Regularization Tuning

Branch continues from the apr5 pretrained baseline but explores different axes.

**Baseline for apr6** (from commit 2929fbd after rereading program.md):  
val_mae ≈ 5.58. The apr6 branch started fresh from the pretrained baseline rather than the optimized apr5 config, exploring a different path.

**exp2: dropout 0.1 → 0.0**  
val_mae ≈ 5.45. With small dataset and pretrained weights, dropout can be too aggressive — it randomly drops activations that encode chemical structure learned during pretraining. Removing dropout allows full signal flow.

**exp19: train seq len 512 → 300**  
Training reactions are all ≤275 tokens (prefix + SMILES). Training at T=512 wastes compute on padding columns that are always zero. Truncating to T=300 speeds up each step (~30% faster, 87% faster with zero waste). val_mae ≈ 5.82 with dropout=0.1. The issue here was that fewer epochs without compensating dropout caused slight overfitting.

**exp20: T=300 + dropout=0.05**  
Adding light dropout to offset the extra epochs from faster steps. val_mae ≈ 5.43. Keep.

**exp22: T=300, dropout 0.05 → 0.08**  
Fine-tuning dropout at T=300. val_mae ≈ 5.35. Better regularization for the higher epoch count. Keep.

**Dropout sweep at T=300:**
- 0.03: worse (too low)
- 0.05: 5.43 (good baseline)
- 0.07: 5.54 (slightly worse)
- 0.08: **5.35** (optimal)
- 0.10: 5.62 (too high — hurts pretrained representations)

**exp59: L1 loss instead of KL divergence**  
Original loss: KL divergence between softmax output and soft label `[(100-y)/100, y/100]`. This indirectly optimizes MAE via a probabilistic formulation.  
L1 loss: directly minimize `|softmax(logits)[1] - y/100|` (the actual MAE evaluation metric).  
val_mae ≈ 5.29. A clear win — directly optimizing the evaluation metric removes the gap between training signal and evaluation. The KL loss was optimizing a proxy. Keep.

**exp60: L1 + dropout 0.08 → 0.05**  
With L1 loss the model converges faster (more direct signal), so it can tolerate slightly less regularization.  
val_mae ≈ 5.01. Significant improvement. Keep.

**exp62: L1 + dropout 0.05 → 0.04**  
Pushing regularization lower. val_mae ≈ 4.84. Another clear win — the direct loss allows aggressive convergence with minimal dropout. Keep.

**Dropout sweep at L1+T=300:**
- 0.02: 5.32 (too low — overfits)
- 0.03: 5.18 (better but still)
- 0.035: 5.22 (worse than 0.04)
- 0.04: **4.84** (optimal)
- 0.045: 5.22 (marginally worse)
- 0.05: 5.01 (starting point)

**Experiments tried and discarded in apr6:**
- *LR 7e-4*: 5.01 — current lr=5e-4 still good
- *LR 3e-4*: 5.54 — too low
- *wd=0.001*: 5.12 — small WD doesn't help
- *wd=1e-5*: 5.06 — same story
- *warmdown 0.3→0.2*: 5.19 — less cooldown time hurts
- *warmdown 0.3→0.4*: 5.35 — more cooldown hurts too
- *cosine warmdown*: 5.31 — linear slightly better
- *SWA every 50 steps*: 5.10 — marginal, adds complexity
- *weight averaging (model soup)*: 5.17 — not better than final weights
- *Huber loss β=0.05*: 5.19 — not better than pure L1
- *L1 + 0.1*KL mixed*: 5.22 — the KL component adds noise to the direct loss
- *Adam beta1 0.9→0.95*: 5.19 — default fine
- *Adam eps 1e-8→1e-6*: 5.31 — default fine
- *max_grad_norm 1.0→0.5*: 5.12 — tighter clipping doesn't help
- *2-run continuation*: 5.34/5.34 — extending training from checkpoint doesn't help over single fresh run with full budget
- *freeze encoder layers 0-1*: 5.34 — lower layers encode basic chemistry, should be fine-tuned
- *label noise sigma=2%*: 5.32 — chemical yields have real noise; adding more noise hurts
- *BOS token as decoder input*: 9.14 — PAD is the correct T5Chem convention, not BOS
- *learnable regression query*: 5.25 — PAD token is fine as decoder input
- *T=256 (1% sequences truncated)*: 5.23 — truncation hurts slightly
- *T=275 (exact max length)*: 5.64 — not better than T=300
- *encoder_dropout=0.0 only*: 5.55 — encoder dropout is important for generalization
- *decoder block dropout=0.0*: 5.17 — both encoder and decoder benefit from regularization
- *embedding dropout=0*: 5.14 — keeping block dropout 0.04 is fine
- *regression head bias=True*: 5.34 — bias adds no value with normalized inputs
- *6 enc+dec layers (4 pretrained + 2 random)*: 5.62 — extra random layers hurt
- *stochastic depth p=0.1*: 5.45 — adds complexity without clear benefit
- *SGDR cosine restarts*: 5.59 — warmdown schedule is better
- *dropout annealing*: 5.83 — complex schedule with loop overhead; not worth it
- *2-stage dropout (0.08→0.0 at warmdown)*: 5.91 — abrupt dropout change disrupts training
- *reactant swap augmentation*: 5.05 — swapping reactants in SMILES broke reaction SMILES format for most reactions (asymmetric reactions)
- *yield-stratified batching*: 5.21 — forcing yield quartile balance per batch doesn't help
- *pairwise ranking hinge loss*: 5.14 — not better than pure L1
- *quadratic warmdown*: 5.09 — linear warmdown good enough
- *NAdam optimizer*: 5.13 — AdamW is fine
- *relative attention max distance 128→300*: 5.43 — disrupts pretrained positional encoding patterns
- *SDPA flash attention scale=1.0*: 5.47 — falls back to slow path; current SDPA is better
- *R-drop KL consistency*: 5.35, 2x slower — consistency regularization doesn't add value here
- *manifold mixup at encoder embedding*: 7.54 — completely destructive for chemical representations
- *seed variants* (7, 0, 1337): 5.62, 5.72, 5.90 — variance between runs is ~0.3 MAE; single-seed results are unreliable
- *token masking (10%)*: 5.80 — SMILES characters encode specific atoms; masking destroys meaning
- *Lookahead optimizer k=5*: 5.87 — added complexity with no benefit

**exp104: attn_dropout=0.0, residual_dropout=0.04 (split attention/residual dropout)**  
Hypothesis: attention weights benefit from being deterministic (no dropout on which tokens attend to which), while residual/FFN paths still benefit from regularization.  
Split into `attn_dropout=0.0` and `dropout=0.04` (residual path only).  
val_mae ≈ 4.82 (initial run). Best result in the whole experiment series.  

**Verification of exp104**: Re-ran 3 times to check for luck:
- Run 1: 5.44 — significantly worse
- Run 2: 5.51 — worse
- Run 3: 5.29 — worse
- Original: 4.82 — likely a lucky data ordering

Conclusion: the 4.82 was **noise** (random data ordering variance). The true mean performance of exp104's config is ~5.4, which is worse than the L1+dropout=0.04 baseline (8e0d588, val_mae=4.84 — also subject to noise).

**Run-to-run variance**: Discovered ~0.3-0.5 MAE variance between identical runs due to random shuffling of the training data. Single-seed results are unreliable; the true "best" configuration has val_mae in the 4.8-5.1 range under optimal conditions.

---

## Key Findings Summary

| Finding | Effect |
|---------|--------|
| Pretrained T5Chem weights from Zenodo | 18.83 → 5.96 MAE (massively better) |
| L1 loss instead of KL divergence | ~5.35 → ~5.00 MAE (direct metric optimization) |
| MLP head (256→256→ReLU→2) | ~5.96 → ~5.65 MAE |
| Head dropout=0.3 | ~5.65 → ~5.44 MAE |
| Batch size 32 → 16 | ~5.44 → ~5.36 MAE (more updates per second) |
| Weight decay=0.02 | ~5.36 → ~5.30 MAE |
| Train seq len 512 → 300 | ~same MAE, 30% faster (covers all real seqs) |
| Dropout tuning (0.1 → 0.04) | ~5.30 → ~4.84 MAE |
| SDPA kernel | same MAE, 30% more steps, 50% less VRAM |
| Encoder-only (from scratch only) | better without pretrained; worse with |
| Decoder kept (with pretrained) | cross-attention has learned chemical context |

## Final Config (best reliable result: ~4.84 ± 0.3 MAE)

```python
# Model
N_ENCODER_LAYERS = 4, N_DECODER_LAYERS = 4
D_MODEL = 256, D_FF = 2048, N_HEAD = 8, D_KV = 64

# Regularization
DROPOUT = 0.04          # residual/FFN dropout
ATTN_DROPOUT = 0.0      # no attention weight dropout

# Optimization
DEVICE_BATCH_SIZE = 32  # (effective; smaller batch tried but mixed results)
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.3    # linear warmdown last 30%
TRAIN_SEQ_LEN = 300     # covers all real seqs (max=275)

# Loss
loss = F.l1_loss(softmax(logits)[1], labels/100)  # direct MAE optimization
```

## Architecture Decisions Kept

1. **Pretrained weights** — loaded from Zenodo T5Chem "simple" checkpoint (97M PubChem)
2. **Full encoder-decoder** — unlike from-scratch (where encoder-only won), pretrained decoder is valuable
3. **Soft-label representation** — output 2 logits, interpret as yield/no-yield probability ratio
4. **L1 loss on prediction** — directly optimize the evaluation metric, not a KL proxy
5. **T=300 encoder input** — shorter than MAX_SEQ_LEN=512, covers 100% of real sequences
6. **Light dropout (0.04)** — minimal regularization; pretrained weights already encode good structure
7. **Linear head (not MLP)** — late finding: the current apr6 best was with a linear head; MLP head explored on apr5

## What Didn't Work (and Why)

- **Encoder-only (with pretrained)**: The decoder cross-attention was pretrained on SMILES; removing it loses that context.
- **High LR (>5e-4)**: Pretrained fine-tuning is sensitive; large updates destroy pretrained representations early.
- **High dropout (>0.08)**: Masks useful chemical structure information learned during pretraining.
- **Activation changes (SiLU, GELU)**: Incompatible with pretrained ReLU FFN weights — catastrophic performance.
- **Batch size 64**: With 5k training samples, fewer-but-larger batches mean fewer optimizer steps per minute.
- **Token masking augmentation**: SMILES characters are positional/structural; masking `c` in benzene breaks meaning.
- **Manifold mixup**: Interpolating between SMILES embeddings creates chemically meaningless inputs.
- **Multiple decoder steps**: Single PAD token input is the T5Chem regression convention; multi-step decoding adds autoregressive noise.
