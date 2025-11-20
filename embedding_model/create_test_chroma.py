import os, sys, gc, csv, json
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer


# # Models to try: 
# # Qwen/Qwen3-Embedding-0.6B 16bit / batch 2
# # Salesforce/SFR-Embedding-Mistral 16 / batch 2 
# # Qwen/Qwen3-Embedding-4B 16bit / batch 2 
# # Linq-AI-Research/Linq-Embed-Mistral 16 bit / batch 2 
# # intfloat/multilingual-e5-large-instruct max_token = 512 / batch 4


# =====================================================
# ---- CONFIG ----
# This is used for generating relative IDs (like '2007/law.../doc.txt')
ROOT_DIR = "/ltstorage/shares/datasets/eu/category15/txt_of_json"

# --- REQUIRED INPUT ---
# The path to your JSONL file containing the queries
TEST_JSONL_PATH = "/ltstorage/home/4baba/EUR_lex/embedding_model/fine_utning/tunning_data/EN_Qwen3-Embedding-4B/test.jsonl"

# Base model setup (Only used if USE_FINE_TUNED_MODEL is False)
BASE_MODEL_ID = "Qwen/Qwen3-Embedding-4B"
COLLECTION = "EN_cat15"
DB_PATH = "/ltstorage/home/4baba/EUR_lex/embedding_model/chroma_DBs/original_test_instance_only/Qwen3-Embedding-4B/"

# Fine-tuned model setup
FINETUNE_MODEL_PATH = ""
FINETUNE_DB_PATH    = ""
FINETUNE_COLLECTION = "EN_cat15"

USE_FINE_TUNED_MODEL = False 

# Shared parameters
TOKEN_LIMIT = 30_000
CAPS_TRY    = [30_000, 20_000, 16_000, 12_000, 8_000, 4_096, 2_048, 1_024]
DEVICE      = "cuda"


# Logs — stored inside the ChromaDB folder
# Determine log file paths based on the USE_FINE_TUNED_MODEL flag
if USE_FINE_TUNED_MODEL:
    # Use FINETUNE paths
    LOG_CSV    = os.path.join(FINETUNE_DB_PATH, "testset_embedding_log.csv")
    FAILED_CSV = os.path.join(FINETUNE_DB_PATH, "testset_problematic_docs.csv")
else:
    # Use BASE paths
    LOG_CSV    = os.path.join(DB_PATH, "testset_embedding_log.csv")
    FAILED_CSV = os.path.join(DB_PATH, "testset_problematic_docs.csv")


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


def load_query_document_paths(jsonl_path):
    """
    Reads the JSONL file and extracts query paths, transforming them into
    paths pointing to the corresponding English text files in ROOT_DIR.

    The transformation logic is:
    1. Replace the JSONL base directory with the ROOT_DIR.
    2. Replace the '.jsonl' extension with '.txt'.

    Args:
        jsonl_path (str): Path to the input .jsonl file.

    Returns:
        list: A list of unique file paths (strings) to be embedded.
    """
    if not os.path.exists(jsonl_path):
        print(f"Error: Input JSONL file not found at {jsonl_path}")
        sys.exit(1)

    # Base path prefix to be replaced in the query_path
    JSONL_BASE_DIR = "/ltstorage/shares/datasets/eu/category15/json_category15"

    unique_doc_paths = set()
    print(f"Loading and transforming query paths from {jsonl_path}...")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line)
                
                if 'query_path' in data:
                    qp = data['query_path']
                    
                    # 1. Check and replace the base directory with the ROOT_DIR (txt_of_json)
                    if qp.startswith(JSONL_BASE_DIR):
                        doc_path = qp.replace(JSONL_BASE_DIR, ROOT_DIR, 1)
                    else:
                        print(f"Warning: Line {line_num + 1}: Query path {qp} does not start with expected prefix ({JSONL_BASE_DIR}). Skipping.")
                        continue
                    
                    # 2. Replace the .jsonl extension with .txt
                    if doc_path.endswith('_EN.jsonl'):
                        doc_path = doc_path.replace('_EN.jsonl', '_EN.txt')
                    elif doc_path.endswith('.jsonl'):
                        # Fallback for just .jsonl, though the example shows _EN.jsonl
                        doc_path = doc_path[:-6] + '.txt'
                    else:
                         print(f"Warning: Line {line_num + 1}: Transformed path {doc_path} does not end with expected extension. Skipping.")
                         continue

                    unique_doc_paths.add(doc_path)

            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON line {line_num + 1}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred processing line {line_num + 1}: {e}")

    return list(unique_doc_paths)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # --- Load all document paths from the test set JSONL file ---
    files = load_query_document_paths(TEST_JSONL_PATH)

    if not files:
        print(f"No unique document paths found in {TEST_JSONL_PATH}")
        sys.exit(0)

    # --- Model setup ---
    # Fix: Initialize variables using global values to avoid UnboundLocalError
    current_model_id = BASE_MODEL_ID
    current_db_path = DB_PATH
    current_collection = COLLECTION

    if USE_FINE_TUNED_MODEL:
        current_model_id = FINETUNE_MODEL_PATH
        current_db_path = FINETUNE_DB_PATH
        current_collection = FINETUNE_COLLECTION
        
        print(f"Loading fine-tuned model from {current_model_id} ...")
        model = SentenceTransformer(current_model_id, device=DEVICE, trust_remote_code=True)
        tokenizer = model.tokenizer
        
        client = chromadb.PersistentClient(path=current_db_path)
        coll = client.get_or_create_collection(name=current_collection, metadata={"hnsw:space": "cosine"})
    else: 
        # Base Model Setup
        # Check for command-line override
        if len(sys.argv) > 1:
            current_model_id = sys.argv[1]
        
        # This line is now safe because current_model_id is guaranteed to have a value (either
        # the default BASE_MODEL_ID or the command-line argument).
        print(f"Using base model: {current_model_id}")
        
        tokenizer = AutoTokenizer.from_pretrained(current_model_id)
        
        client = chromadb.PersistentClient(path=current_db_path)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=current_model_id, normalize_embeddings=True, device=DEVICE
        )
        coll = client.get_or_create_collection(name=current_collection, embedding_function=ef, metadata={"hnsw:space": "cosine"})

    # Ensure log files are created
    ensure_csv(LOG_CSV, ["path", "id", "original_tokens", "used_tokens_saved"])
    ensure_csv(FAILED_CSV, ["path", "id", "original_tokens", "last_error"])

    print(f"Found {len(files)} unique documents derived from query_path in the test set. Backoff caps: {CAPS_TRY}")

    total = 0
    with tqdm(total=len(files), desc=f"Indexing ({'Fine-tuned' if USE_FINE_TUNED_MODEL else 'Base'})", unit="file") as pbar:
        for fp in files:
            # Check if file exists before processing (paths are loaded from JSONL, not guaranteed to exist)
            if not os.path.exists(fp):
                log_row(FAILED_CSV, [fp, os.path.relpath(fp, ROOT_DIR), 0, "File not found"])
                pbar.update(1)
                continue

            text = read_text(fp)
            if not text.strip():
                log_row(FAILED_CSV, [fp, os.path.relpath(fp, ROOT_DIR), 0, "Empty document"])
                pbar.update(1)
                continue

            # ID generation uses the same ROOT_DIR logic as before
            _id = os.path.relpath(fp, ROOT_DIR)
            
            # --- Tokenization and capping logic (unchanged) ---
            enc = tokenizer(
                text, add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                truncation=False,
            )
            ids = enc["input_ids"]
            orig_tokens = len(ids)
            caps = sorted(set([min(c, orig_tokens) for c in [TOKEN_LIMIT, *CAPS_TRY]]), reverse=True)

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
                    if USE_FINE_TUNED_MODEL:
                        # Encode using SentenceTransformer model instance
                        emb = model.encode(
                            text_to_embed,
                            normalize_embeddings=True,
                            convert_to_numpy=True,
                            show_progress_bar=False
                        )
                        # Add pre-calculated embedding to collection
                        coll.add(ids=[_id], documents=[text_to_embed], embeddings=[emb], metadatas=[meta])
                    else:
                        # Let ChromaDB's embedding function calculate the embedding
                        coll.add(ids=[_id], documents=[text_to_embed], metadatas=[meta])
                        
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

    print(f"Done. Successfully indexed {total} documents from the test set.")
    print(f"Success log → {LOG_CSV}")
    print(f"Failures log → {FAILED_CSV}")


if __name__ == "__main__":
    main()