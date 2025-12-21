import json
import random
from pathlib import Path

# === CONFIG ===
INPUT_FILE = ""
OUTPUT_DIR = Path("")
TRAIN_FILE = OUTPUT_DIR / "full_train_pairs.jsonl"
VAL_FILE   = OUTPUT_DIR / "full_val_pairs.jsonl"
TEST_FILE  = OUTPUT_DIR / "full_test_pairs.jsonl"
SEED = 42
TRAIN_RATIO = 0.6
VAL_RATIO   = 0.2  # (test ratio = 1 - train - val)
# ===============

# --- Load all records ---
records = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print(f"Loaded {len(records)} examples from {INPUT_FILE}")

# --- Shuffle deterministically ---
random.seed(SEED)
random.shuffle(records)

# --- Compute split sizes ---
n_total = len(records)
n_train = int(n_total * TRAIN_RATIO)
n_val   = int(n_total * VAL_RATIO)
n_test  = n_total - n_train - n_val

print(f"Splitting: {n_train} train | {n_val} val | {n_test} test")

# --- Split the data ---
train_records = records[:n_train]
val_records   = records[n_train:n_train+n_val]
test_records  = records[n_train+n_val:]

# --- Save subsets ---
def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

save_jsonl(TRAIN_FILE, train_records)
save_jsonl(VAL_FILE, val_records)
save_jsonl(TEST_FILE, test_records)

print("Split completed:")
print(f" - Train: {TRAIN_FILE} ({len(train_records)})")
print(f" - Val:   {VAL_FILE} ({len(val_records)})")
print(f" - Test:  {TEST_FILE} ({len(test_records)})")
