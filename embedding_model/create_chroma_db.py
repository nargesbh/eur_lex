import os, sys, gc, csv
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer

# =====================================================
# ---- CONFIG ----
ROOT_DIR    = "/ltstorage/shares/datasets/eu/category15/txt_of_json"

# Base model setup
BASE_MODEL_ID   = ""  
COLLECTION = ""
DB_PATH = ""

# Fine-tuned model setup
FINETUNE_MODEL_PATH = "/ltstorage/home/4baba/EUR_lex/embedding_model/fine_utning/tunning_data/EN_Qwen3-Embedding-4B/output_mnr/best_model"
FINETUNE_DB_PATH    = "/ltstorage/home/4baba/EUR_lex/embedding_model/chroma_DBs/fine_tunned/EN_Qwen3-Embedding-4B_mnr/"
FINETUNE_COLLECTION = "EN_cat15"


USE_FINE_TUNED_MODEL = True 

# Shared parameters
TOKEN_LIMIT = 30_000
CAPS_TRY    = [30_000, 20_000, 16_000, 12_000, 8_000, 4_096, 2_048, 1_024]
DEVICE      = "cuda"


# Logs — stored inside the ChromaDB folder
if USE_FINE_TUNED_MODEL:
    DB_PATH = FINETUNE_DB_PATH
    LOG_CSV    = os.path.join(DB_PATH, "EN_embedding_log.csv")
    FAILED_CSV = os.path.join(DB_PATH, "EN_problematic_docs.csv")
else:
    print(1)
    # If model is given via command-line, use that to name the folder
    model_short = sys.argv[1].split("/")[-1] if len(sys.argv) > 1 else "Qwen3-0.6B"
    # DB_PATH = f"./EN_{model_short}"  # folder for ChromaDB
    LOG_CSV    = os.path.join(DB_PATH, f"{model_short}_embedding_log.csv")
    FAILED_CSV = os.path.join(DB_PATH, f"{model_short}_problematic_docs.csv")


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
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)


def log_row(path, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def iter_txt_files(root):
    """Yield only English text files ending with '_EN.txt'."""
    for dp, _, files in os.walk(root):
        for fn in files:
            if fn.endswith("_EN.txt"):
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



    # --- Model setup ---
    if USE_FINE_TUNED_MODEL:
        print(f"Loading fine-tuned model from {FINETUNE_MODEL_PATH} ...")
        model = SentenceTransformer(FINETUNE_MODEL_PATH, device=DEVICE, trust_remote_code=True)
        tokenizer = model.tokenizer
        client = chromadb.PersistentClient(path=FINETUNE_DB_PATH)
        coll = client.get_or_create_collection(name=FINETUNE_COLLECTION, metadata={"hnsw:space": "cosine"})
    else: 
        # --- Read model name dynamically from command-line ---
        if len(sys.argv) > 1:
            BASE_MODEL_ID = sys.argv[1]
        # else:
        #     BASE_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
        print(f"Using base model: {BASE_MODEL_ID}")

        model_short = BASE_MODEL_ID.split("/")[-1]
        print(f"Loading base model: {BASE_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        client = chromadb.PersistentClient(path=DB_PATH)
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=BASE_MODEL_ID, normalize_embeddings=True, device=DEVICE
        )
        coll = client.get_or_create_collection(name=COLLECTION, embedding_function=ef, metadata={"hnsw:space": "cosine"})

    ensure_csv(LOG_CSV, ["path", "id", "original_tokens", "used_tokens_saved"])
    ensure_csv(FAILED_CSV, ["path", "id", "original_tokens", "last_error"])

    #if experimenting only with english files
    files = list(iter_txt_files(ROOT_DIR))
    if not files:
        print(f"No .txt files under {ROOT_DIR}")
        sys.exit(0)

    print(f"Found {len(files)} files. Backoff caps: {CAPS_TRY}")

    total = 0
    with tqdm(total=len(files), desc=f"Indexing ({'' if USE_FINE_TUNED_MODEL else 'Base'})", unit="file") as pbar:
        for fp in files:
            text = read_text(fp)
            if not text.strip():
                pbar.update(1)
                continue

            _id = os.path.relpath(fp, ROOT_DIR)
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
                        emb = model.encode(
                            text_to_embed,
                            normalize_embeddings=True,
                            convert_to_numpy=True,
                            show_progress_bar=False
                        )
                        coll.add(ids=[_id], documents=[text_to_embed], embeddings=[emb], metadatas=[meta])
                    else:
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

    print(f"Success log → {LOG_CSV}")
    print(f"Failures log → {FAILED_CSV}")


if __name__ == "__main__":
    main()
