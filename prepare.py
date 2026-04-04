"""
One-time data preparation for autoresearch T5Chem experiments.
Downloads Suzuki-Miyaura yield prediction dataset and builds a character-level tokenizer.

Usage:
    python prepare.py                  # full prep (download + tokenizer)

Data and tokenizer are stored in ~/.cache/autoresearch/.
"""

import os
import sys
import time
import math
import csv
import json
import argparse
import random

import requests
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 512        # max encoder input length (Yield: prefix + SMILES)
TIME_BUDGET = 300         # training time budget in seconds (5 minutes)
EVAL_BATCHES = 50         # number of val batches for MAE evaluation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

# Suzuki-Miyaura cross-coupling yield dataset (Perera et al., Science 2018)
# 5760 Pd-catalyzed reactions with aryl halides and boronic acids
# Source: rxn4chemistry/rxn_yields processed data
DATA_URL = "https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/rxn_yields/data/suzuki_miyaura_data.csv"
TRAIN_FILE = "train.csv"
VAL_FILE = "val.csv"
VAL_SPLIT = 0.1
RANDOM_SEED = 42

# Character-level SMILES vocabulary (T5Chem SimpleTokenizer compatible)
SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
TASK_PREFIX_TOKEN = "Yield:"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"
VOCAB_SIZE = 100  # padded to 100 (matching T5Chem SimpleTokenizer max_size)

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_data():
    """Download Suzuki-Miyaura yield prediction dataset and split into train/val."""
    os.makedirs(DATA_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    val_path = os.path.join(DATA_DIR, VAL_FILE)

    if os.path.exists(train_path) and os.path.exists(val_path):
        with open(train_path) as f:
            train_count = sum(1 for _ in f) - 1
        with open(val_path) as f:
            val_count = sum(1 for _ in f) - 1
        print(f"Data: already downloaded ({train_count} train, {val_count} val) at {DATA_DIR}")
        return

    # Download raw data
    raw_path = os.path.join(DATA_DIR, "raw_suzuki_miyaura.csv")
    if not os.path.exists(raw_path):
        print("Data: downloading Suzuki-Miyaura dataset...")
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.get(DATA_URL, timeout=60)
                response.raise_for_status()
                temp_path = raw_path + ".tmp"
                with open(temp_path, "w") as f:
                    f.write(response.text)
                os.rename(temp_path, raw_path)
                print(f"  Downloaded to {raw_path}")
                break
            except (requests.RequestException, IOError) as e:
                print(f"  Attempt {attempt}/{max_attempts} failed: {e}")
                for path in [raw_path + ".tmp", raw_path]:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                if attempt == max_attempts:
                    print(f"\nFailed to download. Please place a CSV at: {raw_path}")
                    print(f"Expected format: CSV with reaction SMILES and yield columns")
                    print(f"  Column names: 'rxn'/'reaction'/'smiles' and 'yield'/'y'")
                    print(f"  Example row: CCBr.OB(O)c1ccccc1>>CCc1ccccc1,85.3")
                    sys.exit(1)
                time.sleep(2 ** attempt)

    # Parse and split
    print("Data: parsing and splitting...")
    rows = []
    with open(raw_path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        # Auto-detect column names (handles multiple naming conventions)
        rxn_col = next((c for c in fieldnames
                        if c.lower() in ('rxn', 'rxn_smiles', 'reaction', 'reaction_smiles', 'smiles')), None)
        yield_col = next((c for c in fieldnames
                          if c.lower() in ('yield', 'y', 'yield_percent', 'yield(%)', 'output')), None)
        if rxn_col is None or yield_col is None:
            print(f"  Could not auto-detect columns. Found: {fieldnames}")
            print(f"  Expected a reaction SMILES column and a yield column")
            sys.exit(1)
        for row in reader:
            rxn = row[rxn_col].strip()
            try:
                yield_val = float(row[yield_col].strip())
            except ValueError:
                continue
            if rxn and 0 <= yield_val <= 100:
                rows.append((rxn, yield_val))

    print(f"  Parsed {len(rows)} valid reactions")

    # Shuffle and split
    random.seed(RANDOM_SEED)
    random.shuffle(rows)
    val_size = max(1, int(len(rows) * VAL_SPLIT))
    val_rows = rows[:val_size]
    train_rows = rows[val_size:]

    for path, split_rows in [(train_path, train_rows), (val_path, val_rows)]:
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['rxn', 'yield'])
            for rxn, yield_val in split_rows:
                writer.writerow([rxn, yield_val])

    print(f"Data: {len(train_rows)} train, {len(val_rows)} val at {DATA_DIR}")

# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def build_vocab_from_data(data_dir=DATA_DIR):
    """Scan training data to collect all SMILES characters."""
    chars = set()
    for filename in [TRAIN_FILE, VAL_FILE]:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            continue
        with open(filepath, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                chars.update(row['rxn'])
    return sorted(chars)


def train_tokenizer():
    """Build character-level SMILES tokenizer and save vocabulary."""
    vocab_path = os.path.join(TOKENIZER_DIR, "vocab.json")

    if os.path.exists(vocab_path):
        print(f"Tokenizer: already built at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    if not os.path.exists(train_path):
        print("Tokenizer: need training data first. Run download step.")
        sys.exit(1)

    print("Tokenizer: building character-level SMILES vocabulary...")
    t0 = time.time()

    smiles_chars = build_vocab_from_data()

    # Build vocab: special tokens + task prefix + data chars + padding
    vocab = {}
    idx = 0
    for token in SPECIAL_TOKENS:
        vocab[token] = idx
        idx += 1
    vocab[TASK_PREFIX_TOKEN] = idx
    idx += 1
    for char in smiles_chars:
        if char not in vocab:
            vocab[char] = idx
            idx += 1
    # Pad to VOCAB_SIZE with extra tokens
    while idx < VOCAB_SIZE:
        vocab[f"<extra_{idx}>"] = idx
        idx += 1

    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)

    t1 = time.time()
    actual_chars = idx - len(SPECIAL_TOKENS) - 1  # subtract specials and prefix
    print(f"Tokenizer: built in {t1 - t0:.1f}s, {len(vocab)} tokens ({actual_chars} SMILES chars), saved to {vocab_path}")

    # Sanity check
    test = "CCBr.OB(O)c1ccccc1>>CCc1ccccc1"
    tok = Tokenizer.from_directory()
    encoded = tok.encode(test)
    decoded = tok.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={tok.get_vocab_size()})")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train_t5.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Character-level SMILES tokenizer (T5Chem SimpleTokenizer compatible)."""

    def __init__(self, vocab):
        self.vocab = vocab          # token -> id
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.pad_token_id = vocab[PAD_TOKEN]
        self.bos_token_id = vocab[BOS_TOKEN]
        self.eos_token_id = vocab[EOS_TOKEN]
        self.unk_token_id = vocab[UNK_TOKEN]
        self.yield_prefix_id = vocab[TASK_PREFIX_TOKEN]

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "vocab.json"), "r") as f:
            vocab = json.load(f)
        return cls(vocab)

    def get_vocab_size(self):
        return len(self.vocab)

    def get_pad_token_id(self):
        return self.pad_token_id

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, add_prefix=False):
        """Encode SMILES text to token ids (character-level).

        If add_prefix=True, prepends the Yield: task prefix token.
        """
        if isinstance(text, str):
            ids = []
            if add_prefix:
                ids.append(self.yield_prefix_id)
            for char in text:
                ids.append(self.vocab.get(char, self.unk_token_id))
            return ids
        elif isinstance(text, list):
            return [self.encode(t, add_prefix=add_prefix) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def decode(self, ids):
        """Decode token ids back to text (strips special/prefix tokens)."""
        tokens = []
        for tok_id in ids:
            token = self.inv_vocab.get(tok_id, UNK_TOKEN)
            if token in SPECIAL_TOKENS or token == TASK_PREFIX_TOKEN:
                continue
            tokens.append(token)
        return ''.join(tokens)


def _load_csv_data(split):
    """Load reaction SMILES and yields from the prepared CSV split."""
    filename = TRAIN_FILE if split == "train" else VAL_FILE
    filepath = os.path.join(DATA_DIR, filename)
    assert os.path.exists(filepath), f"Data file not found: {filepath}. Run prepare.py first."

    rxns = []
    yields = []
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rxns.append(row['rxn'])
            yields.append(float(row['yield']))
    return rxns, yields


def make_dataloader(tokenizer, B, T, split):
    """
    Yield prediction dataloader for T5Chem regression.
    Returns (input_ids, attention_mask, labels, epoch) per batch.
      input_ids:      (B, T) padded encoder input with Yield: prefix
      attention_mask:  (B, T) 1 for real tokens, 0 for padding
      labels:          (B,)  yield values 0-100
    """
    assert split in ["train", "val"]
    rxns, yields_list = _load_csv_data(split)
    n = len(rxns)
    assert n > 0, f"No data found for split '{split}'"

    # Pre-encode all reactions with Yield: prefix
    encoded = tokenizer.encode(rxns, add_prefix=True)

    # Pre-allocate pinned CPU and GPU buffers
    cpu_input_ids = torch.full((B, T), tokenizer.pad_token_id, dtype=torch.long, pin_memory=True)
    cpu_attn_mask = torch.zeros(B, T, dtype=torch.long, pin_memory=True)
    cpu_labels = torch.zeros(B, dtype=torch.float32, pin_memory=True)
    gpu_input_ids = torch.zeros(B, T, dtype=torch.long, device="cuda")
    gpu_attn_mask = torch.zeros(B, T, dtype=torch.long, device="cuda")
    gpu_labels = torch.zeros(B, dtype=torch.float32, device="cuda")

    epoch = 1
    indices = list(range(n))
    pos = 0

    while True:
        if split == "train" and pos == 0:
            random.shuffle(indices)

        # Fill batch
        cpu_input_ids.fill_(tokenizer.pad_token_id)
        cpu_attn_mask.zero_()

        for b in range(B):
            idx = indices[pos % n]
            pos += 1
            if pos >= n:
                pos = 0
                epoch += 1

            ids = encoded[idx][:T]  # truncate to max length
            seq_len = len(ids)
            cpu_input_ids[b, :seq_len] = torch.tensor(ids, dtype=torch.long)
            cpu_attn_mask[b, :seq_len] = 1
            cpu_labels[b] = yields_list[idx]

        gpu_input_ids.copy_(cpu_input_ids, non_blocking=True)
        gpu_attn_mask.copy_(cpu_attn_mask, non_blocking=True)
        gpu_labels.copy_(cpu_labels, non_blocking=True)
        yield gpu_input_ids, gpu_attn_mask, gpu_labels, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_mae(model, tokenizer, batch_size):
    """
    Mean Absolute Error (MAE) on validation set.
    Predicts yield (0-100) for each reaction and computes average |pred - actual|.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    import torch.nn.functional as _F
    decoder_input_ids = torch.full((batch_size, 1), tokenizer.get_pad_token_id(),
                                    dtype=torch.long, device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    total_ae = 0.0
    total_count = 0
    for _ in range(EVAL_BATCHES):
        input_ids, attn_mask, labels, _ = next(val_loader)
        loss, logits = model(input_ids, attn_mask, decoder_input_ids, labels)
        pred = _F.softmax(logits.float(), dim=-1)[:, 1] * 100
        total_ae += (pred - labels).abs().sum().item()
        total_count += labels.size(0)
    return total_ae / total_count

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for T5Chem experiments")
    parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download data
    download_data()
    print()

    # Step 2: Build tokenizer
    train_tokenizer()
    print()
    print("Done! Ready to train.")
