"""
Autoresearch T5Chem training script. Single-GPU, single-file.
T5 encoder-decoder for Suzuki-Miyaura yield prediction.
Usage: uv run train_t5.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_mae

# ---------------------------------------------------------------------------
# T5 Model (T5Chem: 4-layer encoder-decoder with regression head)
# ---------------------------------------------------------------------------

@dataclass
class T5Config:
    vocab_size: int = 100
    d_model: int = 256
    d_ff: int = 2048
    n_head: int = 8
    d_kv: int = 64
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    dropout: float = 0.1
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class T5RelativePositionBias(nn.Module):
    """T5-style learned relative position bias (bucketed log-linear scheme)."""
    def __init__(self, n_head, is_decoder=False, num_buckets=32, max_distance=128):
        super().__init__()
        self.is_decoder = is_decoder
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, n_head)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional, num_buckets, max_distance):
        ret = torch.zeros_like(relative_position, dtype=torch.long)
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.clamp(n, min=0)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.clamp(val_if_large, max=num_buckets - 1)
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qlen, klen, device):
        context_position = torch.arange(qlen, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        rp_bucket = self._relative_position_bucket(
            relative_position, bidirectional=not self.is_decoder,
            num_buckets=self.num_buckets, max_distance=self.max_distance,
        )
        values = self.relative_attention_bias(rp_bucket)  # (qlen, klen, n_head)
        return values.permute(2, 0, 1).unsqueeze(0)  # (1, n_head, qlen, klen)


class T5Attention(nn.Module):
    """T5 multi-head attention (no 1/sqrt(d_k) scaling, matching HuggingFace T5)."""
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.d_kv = config.d_kv
        self.inner_dim = self.n_head * self.d_kv
        self.q = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(config.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, kv=None, mask=None, position_bias=None):
        B, T, _ = x.size()
        q = self.q(x).view(B, T, self.n_head, self.d_kv).transpose(1, 2)
        kv_in = kv if kv is not None else x
        S = kv_in.size(1)
        k = self.k(kv_in).view(B, S, self.n_head, self.d_kv).transpose(1, 2)
        v = self.v(kv_in).view(B, S, self.n_head, self.d_kv).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, T, S)
        if position_bias is not None:
            scores = scores + position_bias
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
        attn = self.dropout(attn)
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, self.inner_dim)
        return self.o(y)


class T5FFN(nn.Module):
    """T5 feed-forward: dense-relu-dense (original T5, not gated-gelu T5v1.1)."""
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.wo(self.dropout(F.relu(self.wi(x))))


class T5EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = T5Attention(config)
        self.ffn = T5FFN(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None, position_bias=None):
        x = x + self.dropout(self.self_attn(norm(x), mask=mask, position_bias=position_bias))
        x = x + self.dropout(self.ffn(norm(x)))
        return x


class T5DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = T5Attention(config)
        self.cross_attn = T5Attention(config)
        self.ffn = T5FFN(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, enc_out, self_mask=None, cross_mask=None,
                self_pos_bias=None, cross_pos_bias=None):
        x = x + self.dropout(self.self_attn(norm(x), mask=self_mask, position_bias=self_pos_bias))
        x = x + self.dropout(self.cross_attn(norm(x), kv=enc_out, mask=cross_mask, position_bias=cross_pos_bias))
        x = x + self.dropout(self.ffn(norm(x)))
        return x


class T5ForRegression(nn.Module):
    """T5Chem encoder-decoder with 2-output regression head for yield prediction.

    Regression uses soft-label KL divergence: yield y in [0,100] is mapped to
    target distribution [(100-y)/100, y/100], and the model outputs 2 logits
    trained with KLDivLoss. Predicted yield = softmax(logits)[1] * 100.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Shared token embedding (T5 ties encoder/decoder embeddings)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        # Encoder
        self.encoder = nn.ModuleList([T5EncoderBlock(config) for _ in range(config.n_encoder_layers)])
        self.enc_pos_bias = T5RelativePositionBias(
            config.n_head, is_decoder=False,
            num_buckets=config.relative_attention_num_buckets,
            max_distance=config.relative_attention_max_distance)
        # Decoder
        self.decoder = nn.ModuleList([T5DecoderBlock(config) for _ in range(config.n_decoder_layers)])
        self.dec_self_pos_bias = T5RelativePositionBias(
            config.n_head, is_decoder=True,
            num_buckets=config.relative_attention_num_buckets,
            max_distance=config.relative_attention_max_distance)
        self.dec_cross_pos_bias = T5RelativePositionBias(
            config.n_head, is_decoder=False,
            num_buckets=config.relative_attention_num_buckets,
            max_distance=config.relative_attention_max_distance)
        # Regression head: 2 outputs for soft-label KL divergence (T5Chem)
        self.regression_head = nn.Linear(config.d_model, 2, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.pad_token_id = 0  # set from tokenizer before use

    @torch.no_grad()
    def init_weights(self):
        d = self.config.d_model
        inner = self.config.n_head * self.config.d_kv
        d_ff = self.config.d_ff
        # Shared embedding
        nn.init.normal_(self.shared.weight, std=d ** -0.5)
        # Encoder blocks
        for block in self.encoder:
            for linear in [block.self_attn.q, block.self_attn.k, block.self_attn.v]:
                nn.init.normal_(linear.weight, std=d ** -0.5)
            nn.init.normal_(block.self_attn.o.weight, std=inner ** -0.5)
            nn.init.normal_(block.ffn.wi.weight, std=d ** -0.5)
            nn.init.normal_(block.ffn.wo.weight, std=d_ff ** -0.5)
        # Decoder blocks
        for block in self.decoder:
            for attn in [block.self_attn, block.cross_attn]:
                for linear in [attn.q, attn.k, attn.v]:
                    nn.init.normal_(linear.weight, std=d ** -0.5)
                nn.init.normal_(attn.o.weight, std=inner ** -0.5)
            nn.init.normal_(block.ffn.wi.weight, std=d ** -0.5)
            nn.init.normal_(block.ffn.wo.weight, std=d_ff ** -0.5)
        # Regression head + position biases
        nn.init.normal_(self.regression_head.weight, std=d ** -0.5)
        for bias_mod in [self.enc_pos_bias, self.dec_self_pos_bias, self.dec_cross_pos_bias]:
            nn.init.normal_(bias_mod.relative_attention_bias.weight, std=d ** -0.5)

    def estimate_flops(self):
        """Estimated FLOPs per sample (forward + backward)."""
        return 6 * sum(p.numel() for p in self.parameters())

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def setup_optimizer(self, lr=5e-4, weight_decay=0.0, betas=(0.9, 0.999)):
        """Standard AdamW optimizer (matching T5Chem training setup)."""
        decay = [p for n, p in self.named_parameters() if p.ndim >= 2]
        no_decay = [p for n, p in self.named_parameters() if p.ndim < 2]
        return torch.optim.AdamW([
            dict(params=decay, weight_decay=weight_decay),
            dict(params=no_decay, weight_decay=0.0),
        ], lr=lr, betas=betas, eps=1e-8)

    def forward(self, input_ids, attention_mask, decoder_input_ids, labels=None):
        """
        input_ids:         (B, T_enc) encoder input (Yield: prefix + SMILES chars)
        attention_mask:    (B, T_enc) 1=real token, 0=padding
        decoder_input_ids: (B, T_dec) decoder input (pad token, length 1)
        labels:            (B,) yield values 0-100 (optional)
        """
        B, T_enc = input_ids.size()
        T_dec = decoder_input_ids.size(1)

        # --- Encoder ---
        x = self.dropout(self.shared(input_ids))
        enc_mask = (1.0 - attention_mask[:, None, None, :].float()) * torch.finfo(x.dtype).min
        enc_pos = self.enc_pos_bias(T_enc, T_enc, x.device)
        for block in self.encoder:
            x = block(x, mask=enc_mask, position_bias=enc_pos)
        enc_out = norm(x)

        # --- Decoder ---
        y = self.dropout(self.shared(decoder_input_ids))
        causal = torch.triu(torch.ones(T_dec, T_dec, device=y.device), diagonal=1)
        self_mask = causal.float() * torch.finfo(y.dtype).min
        self_mask = self_mask[None, None, :, :]
        cross_mask = (1.0 - attention_mask[:, None, None, :].float()) * torch.finfo(y.dtype).min
        self_pos = self.dec_self_pos_bias(T_dec, T_dec, y.device)
        cross_pos = self.dec_cross_pos_bias(T_dec, T_enc, y.device)
        for block in self.decoder:
            y = block(y, enc_out, self_mask=self_mask, cross_mask=cross_mask,
                     self_pos_bias=self_pos, cross_pos_bias=cross_pos)
        dec_out = norm(y)

        # --- Regression head ---
        logits = self.regression_head(dec_out[:, -1, :])  # (B, 2)

        if labels is not None:
            # T5Chem soft-label regression: yield y -> [(100-y)/100, y/100]
            smooth = torch.stack([(100 - labels) / 100, labels / 100], dim=1)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            loss = F.kl_div(log_probs, smooth.float(), reduction='batchmean')
            return loss, logits

        return logits

    def predict_yield(self, logits):
        """Convert 2-class logits to yield prediction (0-100)."""
        return F.softmax(logits.float(), dim=-1)[:, 1] * 100

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
D_MODEL = 512           # hidden dimension
D_FF = 2048             # feed-forward intermediate dimension
N_HEAD = 8              # number of attention heads
D_KV = 64              # per-head key/value dimension (inner_dim = n_head * d_kv = 512)
N_ENCODER_LAYERS = 6    # encoder depth
N_DECODER_LAYERS = 6    # decoder depth
DROPOUT = 0.1           # dropout rate

# Optimization
DEVICE_BATCH_SIZE = 32   # per-device batch size
LEARNING_RATE = 3e-4     # initial learning rate (AdamW)
WEIGHT_DECAY = 0.01      # weight decay
ADAM_BETAS = (0.9, 0.999) # Adam betas
MAX_GRAD_NORM = 1.0      # gradient clipping max norm
WARMUP_RATIO = 0.06      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.3     # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0      # final LR as fraction of initial

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

config = T5Config(
    vocab_size=vocab_size, d_model=D_MODEL, d_ff=D_FF,
    n_head=N_HEAD, d_kv=D_KV,
    n_encoder_layers=N_ENCODER_LAYERS, n_decoder_layers=N_DECODER_LAYERS,
    dropout=DROPOUT,
)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = T5ForRegression(config)
model.pad_token_id = tokenizer.get_pad_token_id()
model.to_empty(device=device)
model.init_weights()

num_params = model.num_params()
num_flops_per_sample = model.estimate_flops()
print(f"Total parameters: {num_params:,}")
print(f"Estimated FLOPs per sample: {num_flops_per_sample:e}")

optimizer = model.setup_optimizer(
    lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=ADAM_BETAS,
)
for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

model = torch.compile(model, dynamic=False)

# Fixed decoder input: single pad token per sample (T5Chem regression uses 1-step decoder)
decoder_input_ids = torch.full((DEVICE_BATCH_SIZE, 1), tokenizer.get_pad_token_id(),
                                dtype=torch.long, device=device)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
input_ids, attn_mask, labels, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    else:
        # Cosine decay from 1.0 to FINAL_LR_FRAC after warmup
        cosine_progress = (progress - WARMUP_RATIO) / (1.0 - WARMUP_RATIO)
        return FINAL_LR_FRAC + 0.5 * (1.0 - FINAL_LR_FRAC) * (1.0 + math.cos(math.pi * cosine_progress))

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    with autocast_ctx:
        loss, logits = model(input_ids, attn_mask, decoder_input_ids, labels)
    loss.backward()

    # Gradient clipping (T5 default: max_grad_norm=1.0)
    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = loss.detach().item()

    # Prefetch next batch
    input_ids, attn_mask, labels, epoch = next(train_loader)

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    remaining = max(0, TIME_BUDGET - total_training_time)

    with torch.no_grad():
        pred_yield = F.softmax(logits.float(), dim=-1)[:, 1] * 100
        mae = (pred_yield - labels).abs().mean().item()

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | mae: {mae:.2f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

# Final eval
model.eval()
with autocast_ctx:
    val_mae = evaluate_mae(model, tokenizer, DEVICE_BATCH_SIZE)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_mae:          {val_mae:.4f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
