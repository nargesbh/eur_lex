
#!/usr/bin/env python3
import json
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
MANIFEST_FILE = "embedding_model/fine_utning/tunning_data/total_split_manifest.json"
PAIRS_FILE    = "embedding_model/fine_utning/tunning_data/LV/Qwen3_4B/pairs.jsonl"
OUTPUT_DIR    = Path("embedding_model/fine_utning/tunning_data/LV/Qwen3_4B/")
# ===============

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize(path: str) -> str:
    """
    Convert:
        .../32007D0364_DE.jsonl → .../32007D0364
        .../32007D0364_DE.txt   → .../32007D0364
    """
    p = Path(path.strip())
    stem = p.stem.split("_")[0]  # remove "_DE", "_FR", "_EN"
    return str(p.with_name(stem))  # keep directory, replace name

# --- Load manifest ---
with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
    manifest = json.load(f)["splits"]

# Build sets of normalized paths
split_paths = {
    split: {normalize(p) for p in paths}
    for split, paths in manifest.items()
}

print(f"Loaded manifest: {sum(len(v) for v in split_paths.values())} normalized paths")

# --- Prepare output writers ---
writers = {
    split: open(OUTPUT_DIR / f"{split}.jsonl", "w", encoding="utf-8")
    for split in split_paths
}

matched = {split: 0 for split in split_paths}
total = 0

# --- Filter pairs ---
with open(PAIRS_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Processing pairs"):
        if not line.strip():
            continue
        total += 1

        try:
            record = json.loads(line)
            qpath_norm = normalize(record.get("query_path", ""))

            # Match normalized path
            for split, norm_paths in split_paths.items():
                if qpath_norm in norm_paths:
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
