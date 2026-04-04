train_t5.py — T5Chem architecture replacing GPT
Model changes:

GPT (decoder-only, causal LM) → T5 encoder-decoder with regression head
T5Config: d_model=256, d_ff=2048, 8 heads, d_kv=64, 4 encoder + 4 decoder layers
T5-style relative position bias (learned, bucketed log-linear) instead of RoPE
No 1/sqrt(d_k) attention scaling (matching HuggingFace T5)
Regression head: Linear(d_model, 2) with KL divergence soft-label loss — yield y maps to [(100-y)/100, y/100]
Prediction: softmax(logits)[1] * 100
Optimizer changes:

MuonAdamW → standard AdamW (lr=5e-4, betas=(0.9, 0.999))
Added gradient clipping (max_norm=1.0)
Training loop: Same time-budgeted structure, logs MAE alongside loss.

prepare.py — Suzuki-Miyaura dataset replacing ClimbMix
Data changes:

HuggingFace parquet shards → Suzuki-Miyaura CSV download with auto-split (90/10)
Auto-detects common CSV column names (rxn/reaction/smiles, yield/y)
Tokenizer changes:

BPE (rustbpe/tiktoken) → character-level SMILES tokenizer (T5Chem SimpleTokenizer style)
Vocab: 5 special tokens + Yield: prefix + SMILES chars, padded to 100
Dataloader changes:

Language model packing (x, y_shifted) → regression (input_ids, attention_mask, labels)
Pinned-memory async GPU transfer preserved
Evaluation:

evaluate_bpb → evaluate_mae (mean absolute error on yield 0-100)
