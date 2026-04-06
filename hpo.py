"""
Optuna HPO for T5Chem yield prediction.
Traditional hyperparameter optimization for comparison with agentic approach.

Each trial trains for the full 5-minute time budget (identical to agentic).
Architecture is FIXED to match pretrained weights (d_model=256, d_ff=2048,
n_head=8, d_kv=64, 4L encoder, 4L decoder).

Usage:
    uv run hpo.py                    # run 100 trials (default)
    uv run hpo.py --n-trials 50      # run 50 trials
"""

import argparse
import gc
import math
import os
import time

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_mae
from train_t5 import T5Config, T5ForRegression, PRETRAINED_DIR

# ---------------------------------------------------------------------------
# Search space (same hyperparameters the agentic approach explored)
# ---------------------------------------------------------------------------

def suggest_hparams(trial: optuna.Trial) -> dict:
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 0.0, 0.05),
        batch_size=trial.suggest_categorical("batch_size", [8, 12, 16, 24, 32, 64]),
        dropout=trial.suggest_float("dropout", 0.0, 0.3, step=0.05),
        head_dropout=trial.suggest_float("head_dropout", 0.0, 0.5, step=0.05),
        warmup_ratio=trial.suggest_float("warmup_ratio", 0.0, 0.15, step=0.02),
        lr_schedule=trial.suggest_categorical("lr_schedule", ["cosine", "linear"]),
        head_type=trial.suggest_categorical("head_type", ["linear", "mlp"]),
        beta1=trial.suggest_float("beta1", 0.85, 0.95, step=0.05),
    )


# ---------------------------------------------------------------------------
# Training function (5-minute budget, same as agentic)
# ---------------------------------------------------------------------------

def train_and_evaluate(hparams: dict, seed: int = 42) -> dict:
    """Train T5Chem with given hparams for TIME_BUDGET seconds. Returns results dict."""
    from dataclasses import asdict

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()

    config = T5Config(
        vocab_size=vocab_size, d_model=256, d_ff=2048, n_head=8, d_kv=64,
        n_encoder_layers=4, n_decoder_layers=4,
        dropout=hparams["dropout"],
    )

    # Build model
    with torch.device("meta"):
        model = T5ForRegression(config)

    # Swap regression head if MLP
    if hparams["head_type"] == "mlp":
        d = config.d_model
        model.regression_head = nn.Sequential(
            nn.Linear(d, d, bias=False),
            nn.ReLU(),
            nn.Dropout(hparams["head_dropout"]),
            nn.Linear(d, 2, bias=False),
        )

    model.pad_token_id = tokenizer.get_pad_token_id()
    model.to_empty(device=device)
    model.init_weights()
    model.load_pretrained(PRETRAINED_DIR, tokenizer, device)
    num_params = model.num_params()

    # Optimizer
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW([
        dict(params=decay, weight_decay=hparams["weight_decay"]),
        dict(params=no_decay, weight_decay=0.0),
    ], lr=hparams["learning_rate"], betas=(hparams["beta1"], 0.999), eps=1e-8)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]

    model = torch.compile(model, dynamic=False)

    B = hparams["batch_size"]
    decoder_input_ids = torch.full((B, 1), tokenizer.get_pad_token_id(),
                                    dtype=torch.long, device=device)
    train_loader = make_dataloader(tokenizer, B, MAX_SEQ_LEN, "train")
    input_ids, attn_mask, labels, epoch = next(train_loader)

    # LR schedule
    warmup = hparams["warmup_ratio"]
    schedule = hparams["lr_schedule"]

    def get_lr_mult(progress):
        if progress < warmup:
            return progress / warmup if warmup > 0 else 1.0
        if schedule == "cosine":
            cp = (progress - warmup) / (1.0 - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * cp))
        else:  # linear warmdown over last 30%
            if progress < 0.7:
                return 1.0
            return (1.0 - progress) / 0.3

    # Training loop (5 minutes)
    total_training_time = 0.0
    step = 0

    while True:
        torch.cuda.synchronize()
        t0 = time.time()

        with autocast_ctx:
            loss, logits = model(input_ids, attn_mask, decoder_input_ids, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lrm = get_lr_mult(progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
        optimizer.step()
        model.zero_grad(set_to_none=True)

        train_loss_f = loss.detach().item()
        input_ids, attn_mask, labels, epoch = next(train_loader)

        if math.isnan(train_loss_f) or train_loss_f > 100:
            # Cleanup and report crash
            del model, optimizer, train_loader, input_ids, attn_mask, labels, decoder_input_ids
            torch.cuda.empty_cache()
            gc.collect()
            return dict(val_mae=float("inf"), peak_vram_mb=0, num_steps=step,
                        num_params_M=num_params / 1e6, status="crash")

        torch.cuda.synchronize()
        dt = time.time() - t0
        if step > 10:
            total_training_time += dt

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()

        step += 1
        if step > 10 and total_training_time >= TIME_BUDGET:
            break

    # Eval
    gc.enable()
    model.eval()
    with autocast_ctx:
        val_mae = evaluate_mae(model, tokenizer, B)

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Cleanup for next trial
    del model, optimizer, train_loader, input_ids, attn_mask, labels, decoder_input_ids
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    return dict(val_mae=val_mae, peak_vram_mb=peak_vram_mb, num_steps=step,
                num_params_M=num_params / 1e6, status="ok")


# ---------------------------------------------------------------------------
# Optuna objective + logging
# ---------------------------------------------------------------------------

RESULTS_FILE = "hpo_results.tsv"


def write_header():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write("trial\tval_mae\tmemory_gb\tstatus\tdescription\n")


def log_result(trial_num, val_mae, peak_vram_mb, status, desc):
    mem_gb = peak_vram_mb / 1024 if peak_vram_mb > 0 else 0.0
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{trial_num}\t{val_mae:.4f}\t{mem_gb:.1f}\t{status}\t{desc}\n")


def objective(trial: optuna.Trial) -> float:
    hparams = suggest_hparams(trial)
    desc = " ".join(f"{k}={v}" for k, v in sorted(hparams.items()))

    print(f"\n{'='*70}")
    print(f"Trial {trial.number}: {desc}")
    print(f"{'='*70}")

    t0 = time.time()
    try:
        result = train_and_evaluate(hparams, seed=42)
    except Exception as e:
        print(f"Trial {trial.number} CRASHED: {e}")
        log_result(trial.number, 0.0, 0.0, "crash", desc)
        return float("inf")

    elapsed = time.time() - t0

    if result["status"] == "crash":
        print(f"Trial {trial.number}: CRASH (NaN/exploding loss) after {result['num_steps']} steps")
        log_result(trial.number, 0.0, 0.0, "crash", desc)
        return float("inf")

    val_mae = result["val_mae"]
    print(f"Trial {trial.number}: val_mae={val_mae:.4f} | "
          f"vram={result['peak_vram_mb']:.0f}MB | "
          f"steps={result['num_steps']} | "
          f"time={elapsed:.0f}s")

    log_result(trial.number, val_mae, result["peak_vram_mb"], "ok", desc)
    return val_mae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna HPO for T5Chem yield prediction")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of trials (default: 100)")
    parser.add_argument("--study-name", type=str, default="t5chem_hpo",
                        help="Study name")
    parser.add_argument("--seed", type=int, default=42,
                        help="Sampler seed")
    args = parser.parse_args()

    write_header()

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
    )

    est_hours = args.n_trials * 5.5 / 60
    print(f"Optuna HPO: {args.n_trials} trials x 5min = ~{est_hours:.1f} hours")
    print(f"Search: lr, wd, bs, dropout, head_dropout, warmup, schedule, head_type, beta1")
    print(f"Results: {RESULTS_FILE}")
    print()

    study.optimize(objective, n_trials=args.n_trials)

    # Summary
    print(f"\n{'='*70}")
    print(f"HPO COMPLETE: {len(study.trials)} trials")
    print(f"{'='*70}")
    best = study.best_trial
    print(f"Best val_mae: {best.value:.4f} (trial {best.number})")
    print(f"Best params:")
    for k, v in sorted(best.params.items()):
        print(f"  {k}: {v}")
