import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import csv
import time

# --- CONFIG ---
CSV_PATH     = "/ltstorage/shares/datasets/eu/category15/metadata_results.csv"
CHROMA_PATH  = "/ltstorage/home/4baba/EUR_lex/embedding_model/EN_SFR-Embedding-Mistral"
COLLECTION   = "eu_cat15_txt_SFR-Embedding-Mistral"
OUTPUT_PATH  = "/ltstorage/home/4baba/EUR_lex/embedding_model/test_chromadb/EN_SFR-Embedding-Mistral/english_top5_retrieval.csv"
MODEL_ID     = "Salesforce/SFR-Embedding-Mistral"
DEVICE       = "cuda"   
# ----------------

# --- Load metadata CSV ---
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} metadata entries")

# --- Filter only English (_EN.jsonl) files ---
df = df[df["filepath"].astype(str).str.endswith("_EN.jsonl")].reset_index(drop=True)
print(f"Filtered to {len(df)} English (_EN.jsonl) entries")


# --- Setup Chroma client & embedding function ---
client = chromadb.PersistentClient(path=CHROMA_PATH)
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=MODEL_ID,
    normalize_embeddings=True,
    device=DEVICE,
)
coll = client.get_collection(name=COLLECTION, embedding_function=ef)

# --- Prepare output CSV (create header if file doesn't exist) ---
header = [
    "metadata_filepath",
    "closest_1",
    "closest_2",
    "closest_3",
    "closest_4",
    "closest_5"
]

import os
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

f_out = open(OUTPUT_PATH, "a", newline="", encoding="utf-8")
writer = csv.writer(f_out)

# Write header only if file is empty
if f_out.tell() == 0:
    writer.writerow(header)
    f_out.flush()

# --- Query each metadata text and write on the fly ---
start_time = time.time()

for row in tqdm(df.itertuples(), total=len(df), desc="Querying Chroma"):
    metadata_text = getattr(row, "metadata", "")
    metadata_path = getattr(row, "filepath", "")

    # Skip invalid or empty metadata
    if not isinstance(metadata_text, str) or not metadata_text.strip():
        writer.writerow([metadata_path, "EMPTY_METADATA", "", "", "", ""])
        f_out.flush()
        continue

    try:
        result = coll.query(
            query_texts=[metadata_text],
            n_results=5,
        )

        if "metadatas" in result and result["metadatas"]:
            top_paths = [m["path"] for m in result["metadatas"][0]]
        else:
            top_paths = [""] * 5

        # Pad to 5 results
        while len(top_paths) < 5:
            top_paths.append("")

        writer.writerow([metadata_path] + top_paths)
        f_out.flush()  # flush after each row (write to disk immediately)

    except Exception as e:
        writer.writerow([metadata_path, f"ERROR: {str(e)}", "", "", "", ""])
        f_out.flush()
        continue

f_out.close()

elapsed = (time.time() - start_time) / 60
print(f"Done! Saved results to: {OUTPUT_PATH}")
print(f"Total runtime: {elapsed:.2f} minutes")
