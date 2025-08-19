# import os
# import sys
# import chromadb
# from chromadb.utils import embedding_functions
# from tqdm import tqdm

# # ---- CONFIG ----
# ROOT_DIR   = "/ltstorage/shares/datasets/eu/category15/txt_of_json"  # top-level folder
# DB_PATH    = "./chroma_db_category15"                                 # persistent DB folder
# COLLECTION = "eu_cat15_txt"                                           # collection name
# MODEL_ID   = "Qwen/Qwen3-Embedding-4B"                                # HF model
# BATCH_SIZE = 8
# # ---------------

# def iter_txt_files(root):
#     for dp, _, files in os.walk(root):
#         for fn in files:
#             if fn.endswith(".txt"):
#                 yield os.path.join(dp, fn)

# def read_text(fp):
#     try:
#         with open(fp, "r", encoding="utf-8") as f:
#             return f.read()
#     except UnicodeDecodeError:
#         with open(fp, "r", encoding="latin-1", errors="ignore") as f:
#             return f.read()

# def main():
#     # 1) Persistent client
#     client = chromadb.PersistentClient(path=DB_PATH)

#     # 2) Attach embedding_function (Chroma will embed docs & queries for you)
#     ef = embedding_functions.SentenceTransformerEmbeddingFunction(
#         model_name=MODEL_ID,
#         normalize_embeddings=True,
#         device="cuda",   # remove or set to "cpu" if needed
#     )

#     coll = client.get_or_create_collection(
#         name=COLLECTION,
#         embedding_function=ef,
#         metadata={"hnsw:space": "cosine"}
#     )

#     # 3) Walk files and add in batches (no manual vectors now)
#     files = list(iter_txt_files(ROOT_DIR))
#     if not files:
#         print(f"No .txt files under {ROOT_DIR}")
#         sys.exit(0)

#     print(f"Found {len(files)} files. Indexing in batches of {BATCH_SIZE}...")

#     batch_ids, batch_docs, batch_paths = [], [], []
#     total_indexed = 0

#     with tqdm(total=len(files), desc="Indexing files", unit="file") as pbar:
#         for fp in files:
#             txt = read_text(fp)
#             if txt and txt.strip():
#                 _id = os.path.relpath(fp, ROOT_DIR)  # stable, unique id
#                 batch_ids.append(_id)
#                 batch_docs.append(txt)
#                 batch_paths.append(fp)

#                 if len(batch_ids) >= BATCH_SIZE:
#                     coll.add(
#                         ids=batch_ids,
#                         documents=batch_docs,
#                         metadatas=[{"path": p} for p in batch_paths]
#                     )
#                     total_indexed += len(batch_ids)
#                     batch_ids, batch_docs, batch_paths = [], [], []
#                     pbar.set_postfix(indexed=total_indexed, refresh=False)

#             # count the file as processed (even if skipped)
#             pbar.update(1)

#         # Flush remainder
#         if batch_ids:
#             coll.add(
#                 ids=batch_ids,
#                 documents=batch_docs,
#                 metadatas=[{"path": p} for p in batch_paths]
#             )
#             total_indexed += len(batch_ids)
#             pbar.set_postfix(indexed=total_indexed, refresh=False)

#     print(f"Done. Indexed {total_indexed} documents into '{COLLECTION}' at {DB_PATH}")

# if __name__ == "__main__":
#     main()


import os, sys, gc, csv
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer
from tqdm import tqdm

# ---- CONFIG ----
ROOT_DIR    = "/ltstorage/shares/datasets/eu/category15/txt_of_json"
DB_PATH     = "./chroma_db_category15"
COLLECTION  = "eu_cat15_txt"
MODEL_ID    = "Qwen/Qwen3-Embedding-4B"
TOKEN_LIMIT = 30_000                 # initial cap
CAPS_TRY    = [30_000, 20_000, 16_000, 12_000, 8_000, 4_096, 2_048, 1_024]
LOG_CSV     = "./embedding_log.csv"       # path,id,original_tokens,used_tokens_saved
FAILED_CSV  = "./problematic_docs.csv"    # path,id,original_tokens,last_error
DEVICE      = "cuda"                      # GPU only
# ------------------------------------

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

def maybe_free_cuda():
    """Force-release cached CUDA blocks and GC Python refs."""
    if HAS_TORCH and DEVICE.startswith("cuda"):
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()

def ensure_csv(path, header):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

def log_row(path, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

def iter_txt_files(root):
    for dp, _, files in os.walk(root):
        for fn in files:
            if fn.endswith(".txt"):
                yield os.path.join(dp, fn)

def read_text(fp):
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(fp, "r", encoding="latin-1", errors="ignore") as f:
            return f.read()

def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    client = chromadb.PersistentClient(path=DB_PATH)
    # Create EF & collection ONCE (keeps weights resident)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_ID, normalize_embeddings=True, device=DEVICE
    )
    coll = client.get_or_create_collection(
        name=COLLECTION, embedding_function=ef, metadata={"hnsw:space":"cosine"}
    )

    ensure_csv(LOG_CSV,    ["path","id","original_tokens","used_tokens_saved"])
    ensure_csv(FAILED_CSV, ["path","id","original_tokens","last_error"])

    files = list(iter_txt_files(ROOT_DIR))
    if not files:
        print(f"No .txt files under {ROOT_DIR}")
        sys.exit(0)

    print(f"Found {len(files)} files. Backoff caps: {CAPS_TRY} tokens (GPU only).")

    total = 0
    with tqdm(total=len(files), desc="Indexing (cap/backoff per file)", unit="file") as pbar:
        for fp in files:
            text = read_text(fp)
            if not text or not text.strip():
                pbar.update(1); continue

            _id = os.path.relpath(fp, ROOT_DIR)

            # Tokenize once
            enc = tokenizer(
                text, add_special_tokens=False,
                return_attention_mask=False, return_token_type_ids=False,
                truncation=False,
            )
            ids = enc["input_ids"]
            orig_tokens = len(ids)

            # Build caps (unique, <= orig_tokens)
            caps = []
            first_cap = min(TOKEN_LIMIT, orig_tokens)
            caps.append(first_cap)
            for c in CAPS_TRY:
                c = min(c, orig_tokens)
                if c not in caps:
                    caps.append(c)

            used_tokens_saved = None
            last_err = ""

            # Try caps on GPU; clean memory after each attempt
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
                    coll.add(ids=[_id], documents=[text_to_embed], metadatas=[meta])
                    used_tokens_saved = len(used_ids)
                    total += 1
                    # Drop big refs before moving on
                    del text_to_embed, used_ids
                    maybe_free_cuda()
                    break
                except Exception as e:
                    last_err = str(e).split("\n", 1)[0]
                    # Drop attempt refs + free cache, then try smaller cap
                    del text_to_embed, used_ids
                    maybe_free_cuda()
                    continue

            # Log outcome
            if used_tokens_saved is None:
                log_row(FAILED_CSV, [fp, _id, orig_tokens, last_err])
            else:
                log_row(LOG_CSV, [fp, _id, orig_tokens, used_tokens_saved])

            # Per-file cleanup
            del text, enc, ids
            maybe_free_cuda()

            pbar.update(1)
            if total and total % 100 == 0:
                maybe_free_cuda()

    print(f"Done. Indexed {total} documents into '{COLLECTION}' at {DB_PATH}")
    print(f"Saved docs log: {LOG_CSV}")
    print(f"Problematic docs (failed even at 1024): {FAILED_CSV}")

if __name__ == "__main__":
    main()
