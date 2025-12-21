#!/usr/bin/env python3
import os
import sys
import gc
import csv
import json
import argparse

import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer





# =====================================================
# ---- DEFAULT CONFIG (can be overridden via CLI) ----

# Used for generating relative IDs
ROOT_DIR = ""

# JSONL base dir used inside query_path
JSONL_BASE_DIR = ""

# Default path to a test JSONL file (can be overridden or left unused)
TEST_JSONL_PATH = ""

# Base model setup (only used if use_finetuned == False)
BASE_MODEL_ID = ""
COLLECTION    = "cat15"
DB_PATH       = ""

# Fine-tuned model setup
FINETUNE_MODEL_PATH = ""
FINETUNE_DB_PATH    = ""
FINETUNE_COLLECTION = ""

# Default: use base model unless overridden
USE_FINE_TUNED_MODEL = False

# Shared parameters
TOKEN_LIMIT = 30_000
CAPS_TRY    = [30_000, 20_000, 16_000, 12_000, 8_000, 4_096, 2_048, 1_024]
DEVICE      = "cuda"


# =====================================================
# ------------------- ARG PARSING ---------------------
# =====================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Index documents into ChromaDB. "
            "Can either index all txt files for a given language under ROOT_DIR, "
            "or only those referenced by a test JSONL file (via query_path)."
        )
    )

    # Basic paths
    parser.add_argument(
        "--root-dir",
        default=ROOT_DIR,
        help=f"Root directory containing txt files (default: {ROOT_DIR})",
    )

    parser.add_argument(
        "--jsonl-base-dir",
        default=JSONL_BASE_DIR,
        help=f"Base directory that appears in 'query_path' inside JSONL (default: {JSONL_BASE_DIR})",
    )

    parser.add_argument(
        "--test-jsonl",
        default=TEST_JSONL_PATH,
        help=f"Path to test JSONL file for 'from-testset' mode (default: {TEST_JSONL_PATH})",
    )

    # Language code / suffix
    parser.add_argument(
        "--lang-code",
        help=(
            "Language code suffix used in filenames. "
            "For example: EN, DE, MT. "
            "The script will look for '*_<LANG>.txt' in all-files mode, "
            "and will map '*_<LANG>.jsonl' → '*_<LANG>.txt' in testset mode. "
        ),
    )

    # Mode: all files vs testset only
    parser.add_argument(
        "--from-testset",
        action="store_true",
        help=(
            "If set, only documents referenced in --test-jsonl (via query_path) "
            "are embedded. Otherwise, all '*_<LANG>.txt' files under --root-dir "
            "are embedded."
        ),
    )

    # Base model settings
    parser.add_argument(
        "--base-model-id",
        default=BASE_MODEL_ID,
        help=f"Base embedding model ID (default: {BASE_MODEL_ID})",
    )

    parser.add_argument(
        "--collection",
        default=COLLECTION,
        help=f"Chroma collection name for base model (default: {COLLECTION})",
    )

    parser.add_argument(
        "--db-path",
        default=DB_PATH,
        help=f"ChromaDB directory for base model (default: {DB_PATH})",
    )

    # Fine-tuned model settings
    parser.add_argument(
        "--finetune-model-path",
        default=FINETUNE_MODEL_PATH,
        help="Fine-tuned SentenceTransformer model path (default: empty)",
    )

    parser.add_argument(
        "--finetune-db-path",
        default=FINETUNE_DB_PATH,
        help="ChromaDB path for fine-tuned embeddings (default: empty)",
    )

    parser.add_argument(
        "--finetune-collection",
        default=FINETUNE_COLLECTION,
        help=f"Fine-tuned collection name (default: {FINETUNE_COLLECTION})",
    )

    # Toggle base vs fine-tuned
    parser.add_argument(
        "--use-finetuned",
        dest="use_finetuned",
        action="store_true",
        help="Use fine-tuned model + FINETUNE_* settings",
    )
    parser.add_argument(
        "--use-base",
        dest="use_finetuned",
        action="store_false",
        help="Use base model + BASE_* settings",
    )
    parser.set_defaults(use_finetuned=USE_FINE_TUNED_MODEL)

    return parser.parse_args()


# =====================================================
# -------------------- UTILITIES ----------------------
# =====================================================

def maybe_free_cuda():
    """Release cached CUDA memory and collect garbage."""
    if DEVICE.startswith("cuda"):
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()


def ensure_csv(path, header):
    """Creates the CSV file with headers if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def log_row(path, row):
    """Appends a single row to the CSV file."""
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def read_text(fp):
    """Reads the content of a text file, handling common encoding errors."""
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(fp, "r", encoding="latin-1", errors="ignore") as f:
            return f.read()


def iter_txt_files(root, lang_code):
    """
    Yield txt files under `root` that match the pattern '*_<lang_code>.txt'.

    If you ever want 'all .txt files', you can pass lang_code="" and then
    we simply match ".txt" without language suffix.
    """
    suffix = ".txt" if not lang_code else f"_{lang_code}.txt"
    for dp, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(suffix):
                yield os.path.join(dp, fn)


def load_query_document_paths(jsonl_path, root_dir, jsonl_base_dir, lang_code):
    """
    Reads the JSONL file and extracts query paths, transforming them into
    paths pointing to the corresponding text files in ROOT_DIR.

    Transformation logic:
      1. Replace jsonl_base_dir with root_dir.
      2. Replace the '.jsonl' extension with '.txt', with special handling
         for '_<LANG>.jsonl' → '_<LANG>.txt'.

    Args:
        jsonl_path (str): Path to the input .jsonl file.
        root_dir (str): Root directory where txt_of_json files live.
        jsonl_base_dir (str): Base directory that appears in 'query_path'.
        lang_code (str): Language suffix to look for (e.g. 'EN', 'DE', 'MT').

    Returns:
        list[str]: Unique file paths to be embedded.
    """
    if not os.path.exists(jsonl_path):
        print(f"Error: Input JSONL file not found at {jsonl_path}")
        sys.exit(1)

    unique_doc_paths = set()
    print(f"Loading and transforming query paths from {jsonl_path}...")

    lang_suffix_jsonl = f"_{lang_code}.jsonl" if lang_code else ".jsonl"
    lang_suffix_txt   = f"_{lang_code}.txt" if lang_code else ".txt"

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line)

                if "query_path" in data:
                    qp = data["query_path"]

                    # 1. Replace base directory
                    if qp.startswith(jsonl_base_dir):
                        doc_path = qp.replace(jsonl_base_dir, root_dir, 1)
                    else:
                        print(
                            f"Warning: Line {line_num + 1}: Query path {qp} does not "
                            f"start with expected prefix ({jsonl_base_dir}). Skipping."
                        )
                        continue

                    # 2. Replace extension (language-aware)
                    if lang_code and doc_path.endswith(lang_suffix_jsonl):
                        # e.g. ..._EN.jsonl -> ..._EN.txt
                        doc_path = doc_path[: -len(".jsonl")] + ".txt"
                    elif doc_path.endswith(".jsonl"):
                        # Generic fallback
                        doc_path = doc_path[:-6] + ".txt"
                    else:
                        print(
                            f"Warning: Line {line_num + 1}: Transformed path {doc_path} "
                            "does not end with expected extension. Skipping."
                        )
                        continue

                    unique_doc_paths.add(doc_path)

            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON line {line_num + 1}: {e}")
            except Exception as e:
                print(f"Unexpected error processing line {line_num + 1}: {e}")

    return list(unique_doc_paths)


# =====================================================
# ----------------------- MAIN ------------------------
# =====================================================

def main():
    args = parse_args()

    root_dir        = args.root_dir
    jsonl_base_dir  = args.jsonl_base_dir
    test_jsonl_path = args.test_jsonl
    use_finetuned   = args.use_finetuned
    lang_code       = args.lang_code
    from_testset    = args.from_testset

    base_model_id = args.base_model_id
    collection    = args.collection
    db_path       = args.db_path

    finetune_model_path = args.finetune_model_path
    finetune_db_path    = args.finetune_db_path
    finetune_collection = args.finetune_collection

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # --- Decide which files to index ---
    if from_testset:
        files = load_query_document_paths(
            test_jsonl_path,
            root_dir,
            jsonl_base_dir,
            lang_code,
        )
        if not files:
            print(f"No unique document paths found in {test_jsonl_path}")
            sys.exit(0)
        source_desc = f"query_path in test set ({os.path.basename(test_jsonl_path)})"
    else:
        files = list(iter_txt_files(root_dir, lang_code))
        if not files:
            print(
                f"No .txt files found under {root_dir} with suffix "
                f"'{('_' + lang_code + '.txt') if lang_code else '.txt'}'"
            )
            sys.exit(0)
        source_desc = f"all '*_{lang_code}.txt' files under {root_dir}" if lang_code else f"all '.txt' files under {root_dir}"

    # --- Model & Chroma setup ---
    if use_finetuned:
        # Validate CLI for finetuned mode
        if not finetune_model_path:
            print("ERROR: --finetune-model-path is required when using --use-finetuned.")
            sys.exit(1)
        if not finetune_db_path:
            print("ERROR: --finetune-db-path is required when using --use-finetuned.")
            sys.exit(1)

        current_model_id   = finetune_model_path
        current_db_path    = finetune_db_path
        current_collection = finetune_collection

        print(f"Loading fine-tuned model from {current_model_id} ...")
        model = SentenceTransformer(current_model_id, device=DEVICE, trust_remote_code=True)
        tokenizer = model.tokenizer

        client = chromadb.PersistentClient(path=current_db_path)
        coll = client.get_or_create_collection(
            name=current_collection,
            metadata={"hnsw:space": "cosine"},
        )

        # Logs for finetuned model
        if from_testset:
            LOG_CSV    = os.path.join(current_db_path, "testset_embedding_log.csv")
            FAILED_CSV = os.path.join(current_db_path, "testset_problematic_docs.csv")
        else:
            LOG_CSV    = os.path.join(current_db_path, "embedding_log.csv")
            FAILED_CSV = os.path.join(current_db_path, "problematic_docs.csv")

    else:
        # Validate base settings
        if not base_model_id:
            print("ERROR: --base-model-id must be provided when using base model.")
            sys.exit(1)
        if not collection:
            print("ERROR: --collection must be provided when using base model.")
            sys.exit(1)
        if not db_path:
            print("ERROR: --db-path must be provided when using base model.")
            sys.exit(1)

        current_model_id   = base_model_id
        current_db_path    = db_path
        current_collection = collection

        print(f"Using base model: {current_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(current_model_id)

        client = chromadb.PersistentClient(path=current_db_path)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=current_model_id,
            normalize_embeddings=True,
            device=DEVICE,
        )
        coll = client.get_or_create_collection(
            name=current_collection,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

        model = None  # embeddings handled by Chroma embedding_function

        model_short = current_model_id.split("/")[-1]
        if from_testset:
            LOG_CSV    = os.path.join(current_db_path, f"testset_{model_short}_embedding_log.csv")
            FAILED_CSV = os.path.join(current_db_path, f"testset_{model_short}_problematic_docs.csv")
        else:
            LOG_CSV    = os.path.join(current_db_path, f"{model_short}_embedding_log.csv")
            FAILED_CSV = os.path.join(current_db_path, f"{model_short}_problematic_docs.csv")

    # Ensure log files are created
    ensure_csv(LOG_CSV, ["path", "id", "original_tokens", "used_tokens_saved"])
    ensure_csv(FAILED_CSV, ["path", "id", "original_tokens", "last_error"])

    print(
        f"Found {len(files)} documents from {source_desc}. "
        f"Backoff caps: {CAPS_TRY}"
    )

    total = 0
    mode_name = "Fine-tuned" if use_finetuned else "Base"

    with tqdm(total=len(files), desc=f"Indexing ({mode_name})", unit="file") as pbar:
        for fp in files:
            # Check if file exists before processing
            if not os.path.exists(fp):
                _id = os.path.relpath(fp, root_dir)
                log_row(FAILED_CSV, [fp, _id, 0, "File not found"])
                pbar.update(1)
                continue

            text = read_text(fp)
            if not text.strip():
                _id = os.path.relpath(fp, root_dir)
                log_row(FAILED_CSV, [fp, _id, 0, "Empty document"])
                pbar.update(1)
                continue

            # ID generation uses root_dir
            _id = os.path.relpath(fp, root_dir)

            # --- Tokenization and capping logic ---
            enc = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                truncation=False,
            )
            ids = enc["input_ids"]
            orig_tokens = len(ids)
            caps = sorted(
                set([min(c, orig_tokens) for c in [TOKEN_LIMIT, *CAPS_TRY]]),
                reverse=True,
            )

            used_tokens_saved = None
            last_err = ""

            for cap in caps:
                used_ids = ids[:cap]
                text_to_embed = tokenizer.decode(used_ids, skip_special_tokens=True)
                meta = {
                    "path": fp,
                    "n_tokens_orig": orig_tokens,
                    "n_tokens_used": len(used_ids),
                    "token_limit_initial": TOKEN_LIMIT,
                    "cap_attempt": cap,
                    "device_used": DEVICE,
                }
                try:
                    if use_finetuned:
                        # Encode using SentenceTransformer model instance
                        emb = model.encode(
                            text_to_embed,
                            normalize_embeddings=True,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                        )
                        # Add pre-calculated embedding to collection
                        coll.add(
                            ids=[_id],
                            documents=[text_to_embed],
                            embeddings=[emb],
                            metadatas=[meta],
                        )
                    else:
                        # Let ChromaDB's embedding function calculate the embedding
                        coll.add(
                            ids=[_id],
                            documents=[text_to_embed],
                            metadatas=[meta],
                        )

                    used_tokens_saved = len(used_ids)
                    total += 1
                    del text_to_embed, used_ids
                    maybe_free_cuda()
                    break
                except Exception as e:
                    last_err = str(e).split("\n", 1)[0]
                    del text_to_embed, used_ids
                    maybe_free_cuda()
                    continue

            if used_tokens_saved is None:
                log_row(FAILED_CSV, [fp, _id, orig_tokens, last_err])
            else:
                log_row(LOG_CSV, [fp, _id, orig_tokens, used_tokens_saved])

            del text, enc, ids
            maybe_free_cuda()
            pbar.update(1)

    print(f"Done. Successfully indexed {total} documents.")
    print(f"Success log → {LOG_CSV}")
    print(f"Failures log → {FAILED_CSV}")


if __name__ == "__main__":
    main()
