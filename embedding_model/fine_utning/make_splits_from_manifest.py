#!/usr/bin/env python3
import json
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
MANIFEST_FILE = "/ltstorage/home/4baba/EUR_lex/embedding_model/fine_utning/tunning_data/total_split_manifest.json"
PAIRS_FILE    = "/ltstorage/home/4baba/EUR_lex/embedding_model/fine_utning/tunning_data/EN_SFR-Embedding-Mistral/pairs.jsonl"
OUTPUT_DIR    = Path("/ltstorage/home/4baba/EUR_lex/embedding_model/fine_utning/tunning_data/EN_SFR-Embedding-Mistral")
# ===============

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load manifest ---
with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
    manifest = json.load(f)["splits"]

# Build sets for fast lookup
split_paths = {split: set(paths) for split, paths in manifest.items()}
print(f"Loaded manifest: {sum(len(v) for v in split_paths.values())} total paths")

# --- Prepare output writers ---
writers = {
    split: open(OUTPUT_DIR / f"{split}.jsonl", "w", encoding="utf-8")
    for split in split_paths
}

# --- Filter pairs ---
matched = {split: 0 for split in split_paths}
total = 0

with open(PAIRS_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Processing pairs"):
        if not line.strip():
            continue
        total += 1
        try:
            record = json.loads(line)
            qpath = record.get("query_path", "").strip()
            # assign to correct split if exists
            for split, paths in split_paths.items():
                if qpath in paths:
                    writers[split].write(json.dumps(record, ensure_ascii=False) + "\n")
                    matched[split] += 1
                    break
        except Exception as e:
            print(f"Skipping line {total}: {e}")

# --- Close writers ---
for w in writers.values():
    w.close()

# --- Summary ---
print("Done")
print(f"Processed: {total:,} pairs")
for split, count in matched.items():
    print(f"  {split:<6}: {count:,} matched → {OUTPUT_DIR / (split + '.jsonl')}")
