import os
import csv
import time
import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from typing import List, Set

# --- CONFIG ---
# The large CSV containing all document metadata.
CSV_PATH     = "/ltstorage/shares/datasets/eu/category15/metadata_results.csv"
# The JSONL file containing the specific subset of queries to evaluate.
TEST_JSONL_PATH = "/ltstorage/home/4baba/EUR_lex/embedding_model/fine_utning/tunning_data/EN_Qwen3-Embedding-4B/test.jsonl" 

CHROMA_PATH  = "/ltstorage/home/4baba/EUR_lex/embedding_model/chroma_DBs/EN_Qwen3-Embedding-4B"
COLLECTION   = "eu_cat15_txt_Qwen3-Embedding-4B"
OUTPUT_PATH  = "/ltstorage/home/4baba/EUR_lex/embedding_model/test_chromadb/full_dataset_searchTest/original_model.csv"
MODEL_ID     = "Qwen/Qwen3-Embedding-4B"

# ----------------
DEVICE = "cuda"  

def load_target_query_paths(jsonl_path: str) -> Set[str]:
    """
    Reads the JSONL file and extracts all unique 'query_path' values.
    
    Args:
        jsonl_path: Path to the input .jsonl file.

    Returns:
        Set of unique query paths (strings).
    """
    if not os.path.exists(jsonl_path):
        print(f"Error: Input TEST_JSONL_PATH file not found at {jsonl_path}")
        return set()

    query_paths = set()
    print(f"Loading target query paths from {jsonl_path}...")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line)
                if 'query_path' in data:
                    query_paths.add(data['query_path'])
            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON line {line_num + 1}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred processing line {line_num + 1}: {e}")
                
    return query_paths

def main():
    target_paths = load_target_query_paths(TEST_JSONL_PATH)
    if not target_paths:
        print("No target query paths found. Exiting.")
        return

    print(f"Found {len(target_paths)} unique query paths to evaluate.")

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} total metadata entries from CSV_PATH.")

    df_filtered = df[df["filepath"].isin(target_paths)].reset_index(drop=True)
    print(f"Filtered DataFrame to {len(df_filtered)} matching entries based on TEST_JSONL_PATH.")
    
    if df_filtered.empty:
        print("The filtered DataFrame is empty. No queries to run. Check if filepaths match exactly.")
        return

    # --- Setup Chroma client & embedding function ---
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_ID,
        normalize_embeddings=True,
        device=DEVICE,
    )
    coll = client.get_collection(name=COLLECTION, embedding_function=ef)

    header = [
        "metadata_filepath",
        "closest_1",
        "closest_2",
        "closest_3",
        "closest_4",
        "closest_5"
    ]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    f_out = open(OUTPUT_PATH, "a", newline="", encoding="utf-8")
    writer = csv.writer(f_out)

    if f_out.tell() == 0:
        writer.writerow(header)
        f_out.flush()

    # --- Query each metadata text and write on the fly ---
    start_time = time.time()

    for row in tqdm(df_filtered.itertuples(), total=len(df_filtered), desc="Querying Chroma (Test Subset)"):
        # The query text is the 'metadata' column
        metadata_text = getattr(row, "metadata", "")
        query_path = getattr(row, "filepath", "")


        if not isinstance(metadata_text, str) or not metadata_text.strip():
            writer.writerow([query_path, "EMPTY_METADATA", "", "", "", ""])
            f_out.flush()
            continue

        try:
            result = coll.query(
                query_texts=[metadata_text],
                n_results=5,
            )

            if "ids" in result and result["ids"]:
                top_ids = result["ids"][0]
            else:
                top_ids = [""] * 5

            # Pad to 5 results
            while len(top_ids) < 5:
                top_ids.append("")

            writer.writerow([query_path] + top_ids)
            f_out.flush() 

        except Exception as e:
            writer.writerow([query_path, f"ERROR: {str(e)}", "", "", "", ""])
            f_out.flush()
            continue

    f_out.close()

    elapsed = (time.time() - start_time) / 60
    print(f"Saved results to: {OUTPUT_PATH}")
    print(f"Total runtime: {elapsed:.2f} minutes")


if __name__ == "__main__":
    main()