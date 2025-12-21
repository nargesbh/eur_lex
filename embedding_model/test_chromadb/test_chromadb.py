# CUDA_VISIBLE_DEVICES=1 \
# python test_chromadb.py \
#   --mode test \
#   --csv-path metadata_results.csv \
#   --test-jsonl embedding_model/fine_tuning/tunning_data/≥/EN_multilingual-e5-large-instruct/test.jsonl \
#   --chroma-path embedding_model/chroma_DBs/LV-EN/fine_tuned_test_instances_EN/multilingual-e5-large-instruct \
#   --collection cat15 \
#   --model-idembedding_model/fine_tuning/tunning_data/LV-EN/multilingual-e5-large-instruct/output_mnr/best_model \
#   --output-path embedding_model/test_chromadb/full_dataset_searchTest/LV-EN/multilingual-e5-large-instruct/fine_tune_englishTests.csv \
#   --device cuda

#!/usr/bin/env python3
import os
import csv
import time
import json
import argparse
from typing import Set

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm


# --------------------- ARGPARSE --------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Query a ChromaDB collection using metadata texts.\n"
            "Two modes:\n"
            "  - all  : query all docs (filtered by --lang-ids)\n"
            "  - test : query only test subset defined in --test-jsonl"
        )
    )

    parser.add_argument(
        "--mode",
        choices=["all", "test"],
        required=True,
        help="Mode to run: 'all' for full dataset + lang filter, 'test' for test subset only.",
    )

    parser.add_argument(
        "--csv-path",
        required=True,
        help="Path to metadata CSV file (must contain 'filepath' and 'metadata' columns).",
    )

    # Only used in test mode
    parser.add_argument(
        "--test-jsonl",
        help="Path to JSONL file containing test instances (needed in mode 'test').",
    )

    parser.add_argument(
        "--chroma-path",
        required=True,
        help="Path to ChromaDB directory.",
    )

    parser.add_argument(
        "--collection",
        required=True,
        help="ChromaDB collection name.",
    )

    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to output CSV with retrieval results.",
    )

    parser.add_argument(
        "--model-id",
        required=True,
        help="SentenceTransformer model ID/path for embedding function (fine-tuned or original).",
    )

    parser.add_argument(
        "--lang-ids",
        nargs="+",
        help=(
            "[Mode all] One or more language IDs used to filter rows, e.g. 'MT', 'DE', 'FR', 'LV'. "
            "You can pass space-separated values (LV DE FR) or comma-separated (LV,DE,FR). "
            "Rows are kept if filepath ends with '_(LANG).jsonl' for any LANG."
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for embedding model, e.g. 'cuda' or 'cpu' (default: cuda).",
    )

    return parser.parse_args()


# --------------------- HELPERS --------------------- #

def normalize_lang_ids(raw_lang_ids):
    # """
    # Accepts something like:
    #   ['LV'] or ['LV', 'DE'] or ['LV,DE,FR']
    # and returns a clean list: ['LV', 'DE', 'FR'].
    # """
    if not raw_lang_ids:
        return []
    lang_ids = []
    for token in raw_lang_ids:
        parts = [p.strip() for p in token.split(",") if p.strip()]
        lang_ids.extend(parts)
    # Remove duplicates, preserve order
    seen = set()
    cleaned = []
    for l in lang_ids:
        if l not in seen:
            seen.add(l)
            cleaned.append(l)
    return cleaned


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


# --------------------- MAIN LOGIC --------------------- #

def main():
    args = parse_args()

    MODE        = args.mode
    CSV_PATH    = args.csv_path
    TEST_JSONL  = args.test_jsonl
    CHROMA_PATH = args.chroma_path
    COLLECTION  = args.collection
    MODEL_ID    = args.model_id
    OUTPUT_PATH = args.output_path
    DEVICE      = args.device

    LANG_IDS = normalize_lang_ids(args.lang_ids)

    # Basic sanity checks based on mode
    if MODE == "all" and not LANG_IDS:
        raise ValueError("In mode 'all', --lang-ids is required (at least one language).")

    if MODE == "test" and not TEST_JSONL:
        raise ValueError("In mode 'test', --test-jsonl is required.")

    if not os.path.isdir(CHROMA_PATH):
        print(f"[ERROR] Chroma path not found: {CHROMA_PATH}")
        return

    # --- Load CSV with metadata ---
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} total metadata entries from CSV_PATH: {CSV_PATH}")
    df["filepath"] = df["filepath"].astype(str)

    # Mode-specific filtering
    if MODE == "all":
        SUFFIXES = [f"_{lang}.jsonl" for lang in LANG_IDS]

        def match_any_suffix(path: str) -> bool:
            return any(path.endswith(suf) for suf in SUFFIXES)

        mask = df["filepath"].apply(match_any_suffix)
        df_filtered = df[mask].reset_index(drop=True)

        print("\n" + "=" * 100)
        print(" NEW RUN (MODE: all)")
        print(f"   LANG_IDS        : {LANG_IDS}")
        print(f"   SUFFIXES        : {SUFFIXES}")
        print(f"   MODEL_ID        : {MODEL_ID}")
        print(f"   CSV_PATH        : {CSV_PATH}")
        print(f"   CHROMA_PATH     : {CHROMA_PATH}")
        print(f"   COLLECTION      : {COLLECTION}")
        print(f"   OUTPUT_PATH     : {OUTPUT_PATH}")
        print(f"   DEVICE          : {DEVICE}")
        print("=" * 100 + "\n")

    else:  # MODE == "test"
        target_paths = load_target_query_paths(TEST_JSONL)
        if not target_paths:
            print("No target query paths found. Exiting.")
            return

        print(f"Found {len(target_paths)} unique query paths to evaluate.")

        df_filtered = df[df["filepath"].isin(target_paths)].reset_index(drop=True)

        print("\n" + "=" * 100)
        print(" NEW RUN (MODE: test)")
        print(f"   TEST_JSONL      : {TEST_JSONL}")
        print(f"   MODEL_ID        : {MODEL_ID}")
        print(f"   CSV_PATH        : {CSV_PATH}")
        print(f"   CHROMA_PATH     : {CHROMA_PATH}")
        print(f"   COLLECTION      : {COLLECTION}")
        print(f"   OUTPUT_PATH     : {OUTPUT_PATH}")
        print(f"   DEVICE          : {DEVICE}")
        print(f"   MATCHING_ROWS   : {len(df_filtered)}")
        print("=" * 100 + "\n")

    if df_filtered.empty:
        print("[WARN] Filtered DataFrame is empty. No queries to run.")
        return

    # --- Setup Chroma client & embedding function ---
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_ID,
        normalize_embeddings=True,
        device=DEVICE,
    )
    
    print(coll.metadata)
    coll = client.get_collection(name=COLLECTION, embedding_function=ef)

    header = [
        "metadata_filepath",
        "closest_1",
        "closest_2",
        "closest_3",
        "closest_4",
        "closest_5",
    ]

    out_dir = os.path.dirname(OUTPUT_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    start_time = time.time()

    with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)

        # Write header if file is empty
        if f_out.tell() == 0:
            writer.writerow(header)
            f_out.flush()

        desc = "Querying Chroma (all: {})".format(",".join(LANG_IDS)) if MODE == "all" else "Querying Chroma (Test Subset)"

        # --- Query each metadata text and write on the fly ---
        for row in tqdm(df_filtered.itertuples(), total=len(df_filtered), desc=desc):
            metadata_text = getattr(row, "metadata", "")
            metadata_path = getattr(row, "filepath", "")

            if not isinstance(metadata_text, str) or not metadata_text.strip():
                writer.writerow([metadata_path, "EMPTY_METADATA", "", "", "", ""])
                f_out.flush()
                continue

            try:
                result = coll.query(
                    query_texts=[metadata_text],
                    n_results=5,
                )

                if MODE == "all":
                    # original "everything" script: use metadatas["path"]
                    if "metadatas" in result and result["metadatas"]:
                        top_vals = [m.get("path", "") for m in result["metadatas"][0]]
                    else:
                        top_vals = [""] * 5
                else:
                    # original "test set" script: use ids
                    if "ids" in result and result["ids"]:
                        top_vals = result["ids"][0]
                    else:
                        top_vals = [""] * 5

                # Pad to 5
                while len(top_vals) < 5:
                    top_vals.append("")

                writer.writerow([metadata_path] + top_vals)
                f_out.flush()

            except Exception as e:
                writer.writerow([metadata_path, f"ERROR: {str(e)}", "", "", "", ""])
                f_out.flush()
                continue

    elapsed = (time.time() - start_time) / 60
    print(f"Saved results to: {OUTPUT_PATH}")
    print(f"Total runtime: {elapsed:.2f} minutes")


if __name__ == "__main__":
    main()
