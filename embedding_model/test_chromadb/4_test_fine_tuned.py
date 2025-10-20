import os, json, csv, time
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# ====================== CONFIG ======================
TEST_FILE = "/ltstorage/home/4baba/EUR_lex/embedding_model/fine_utning/english_datasets_0.6b/test_pairs.jsonl"

# --- Fine-tuned LoRA model setup ---
LORA_MODEL_PATH  = "/ltstorage/home/4baba/EUR_lex/embedding_model/fine_utning/english_datasets_0.6b/qwen3_0.6b_lora"
LORA_CHROMA_PATH = "/ltstorage/home/4baba/EUR_lex/embedding_model/chromadb_english_lora_0.6b"
LORA_COLLECTION  = "eu_cat15_txt_lora"
LORA_OUTPUT_CSV  = "/ltstorage/home/4baba/EUR_lex/embedding_model/test_chromadb/english_qwen3_0.6b_lora/english_top5_test.csv"

# --- Original model setup ---
BASE_MODEL_ID    = "Qwen/Qwen3-Embedding-0.6B"
BASE_CHROMA_PATH = "/ltstorage/home/4baba/EUR_lex/embedding_model/chromadb_english_06B"
BASE_COLLECTION  = "eu_cat15_txt"
BASE_OUTPUT_CSV  = "/ltstorage/home/4baba/EUR_lex/embedding_model/test_chromadb/english_Qwen0.6B/english_top5_test.csv"

# --- Choose which model to run ---
USE_FINE_TUNED_MODEL = True   # ← True for fine-tuned, False for base model

DEVICE = "cuda"
# ====================================================


def load_test_pairs(path):
    """Load all test lines as JSON objects."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ensure_dir(path):
    """Ensure directory exists for output file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # --- Select model + Chroma setup ---
    if USE_FINE_TUNED_MODEL:
        model_name = "Fine-tuned LoRA Qwen3-0.6B"
        MODEL_PATH = LORA_MODEL_PATH
        CHROMA_PATH = LORA_CHROMA_PATH
        COLLECTION = LORA_COLLECTION
        OUTPUT_CSV = LORA_OUTPUT_CSV
    else:
        model_name = "Base Qwen3-0.6B"
        MODEL_PATH = BASE_MODEL_ID
        CHROMA_PATH = BASE_CHROMA_PATH
        COLLECTION = BASE_COLLECTION
        OUTPUT_CSV = BASE_OUTPUT_CSV

    ensure_dir(OUTPUT_CSV)

    # ---------- Load model ----------
    print(f"Loading {model_name} ...")
    if USE_FINE_TUNED_MODEL:
        model = SentenceTransformer(MODEL_PATH, device=DEVICE, trust_remote_code=True)
    else:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MODEL_PATH,
            normalize_embeddings=True,
            device=DEVICE,
        )

    # ---------- Connect to Chroma ----------
    print(f"Connecting to Chroma at: {CHROMA_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if USE_FINE_TUNED_MODEL:
        coll = client.get_collection(name=COLLECTION)
    else:
        coll = client.get_collection(name=COLLECTION, embedding_function=ef)

    # ---------- Prepare output CSV ----------
    header = ["metadata_filepath", "closest_1", "closest_2", "closest_3", "closest_4", "closest_5"]
    f_out = open(OUTPUT_CSV, "w", newline="", encoding="utf-8")
    writer = csv.writer(f_out)
    writer.writerow(header)
    f_out.flush()

    # ---------- Load test pairs ----------
    print(f"Loading test pairs from {TEST_FILE} ...")
    test_lines = load_test_pairs(TEST_FILE)
    print(f"Loaded {len(test_lines)} test samples.")

    # ---------- Query loop ----------
    start_time = time.time()

    for ex in tqdm(test_lines, desc=f"Querying with {model_name}", unit="query"):
        query_text = ex.get("query_text", "").strip()
        query_path = ex.get("query_path", "")

        if not query_text:
            writer.writerow([query_path, "EMPTY_QUERY", "", "", "", ""])
            f_out.flush()
            continue

        try:
            if USE_FINE_TUNED_MODEL:
                # Encode manually with fine-tuned LoRA model
                query_emb = model.encode(
                    query_text,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )
                result = coll.query(
                    query_embeddings=[query_emb],
                    n_results=5,
                )
            else:
                # Query using the base model (embedding function attached to Chroma)
                result = coll.query(
                    query_texts=[query_text],
                    n_results=5,
                )

            # Extract paths
            if "metadatas" in result and result["metadatas"]:
                top_paths = [m.get("path", "") for m in result["metadatas"][0]]
            else:
                top_paths = [""] * 5

            while len(top_paths) < 5:
                top_paths.append("")

            writer.writerow([query_path] + top_paths)
            f_out.flush()

        except Exception as e:
            writer.writerow([query_path, f"ERROR: {str(e)}", "", "", "", ""])
            f_out.flush()
            continue

    f_out.close()
    elapsed = (time.time() - start_time) / 60
    print(f"Done! Saved results to: {OUTPUT_CSV}")
    print(f"Total runtime: {elapsed:.2f} minutes")


if __name__ == "__main__":
    main()
