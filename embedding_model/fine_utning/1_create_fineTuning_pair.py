import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm

# === CONFIG ===
META_FILE = "/ltstorage/shares/datasets/eu/category15/metadata_results.csv"
CHROMA_TOP5_FILE = "/ltstorage/home/4baba/EUR_lex/embedding_model/test_chromadb/english_Qwen0.6B/english_top5_retrieval.csv"
OUTPUT_JSONL = "/ltstorage/home/4baba/EUR_lex/embedding_model/fine_utning/english_datasets/fine_tune_pairs.jsonl"
# ===============

meta_df = pd.read_csv(META_FILE)
chroma_df = pd.read_csv(CHROMA_TOP5_FILE)
chroma_map = chroma_df.set_index("metadata_filepath").to_dict(orient="index")


# --- Filter only English metadata (_EN.jsonl) ---
meta_df = meta_df[meta_df["filepath"].astype(str).str.endswith("_EN.jsonl")].reset_index(drop=True)
print(f"Filtered to {len(meta_df)} English (_EN.jsonl) entries")


def jsonl_to_txt(path: str) -> str:
    return (
        path.replace("json_category15", "txt_of_json")
            .replace(".jsonl", ".txt")
    )

records = []

for _, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
    meta_path = row["filepath"]
    query_text = row["metadata"]
    positive_path = jsonl_to_txt(meta_path)
    negative_paths = []

    if meta_path in chroma_map:
        entry = chroma_map[meta_path]
        top5 = [v for v in entry.values() if isinstance(v, str)]

        # Positive might be in top5 — handle both cases
        if positive_path in top5:
            negative_paths = [p for p in top5 if p != positive_path][:4]
        else:
            negative_paths = top5[:4]
    else:
        negative_paths = []

    records.append({
        "query_text": query_text,
        "query_path": meta_path,
        "positive_path": positive_path,
        "negative_paths": negative_paths
    })

with open(OUTPUT_JSONL, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

print(f"Saved {len(records)} examples to {OUTPUT_JSONL}")
