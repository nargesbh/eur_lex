#!/usr/bin/env python3
import os
import sys
import json
import gc
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BitsAndBytesConfig


# ===========================================
# USAGE:
# CUDA_VISIBLE_DEVICES=3 python fine_tune_full.py \
#    /ltstorage/home/4baba/EUR_lex/embedding_model/fine_utning/tunning_data/EN_SFR-Embedding-Mistral \
#  Salesforce/SFR-Embedding-Mistral
# CUDA_VISIBLE_DEVICES=3 python fine_tune_full.py    /ltstorage/home/4baba/EUR_lex/embedding_model/fine_utning/tunning_data/EN_Linq-Embed-Mistral  Linq-AI-Research/Linq-Embed-Mistral
# ===========================================

if len(sys.argv) != 3:
    print("Usage: python fine_tune_mnr.py <DATA_DIR> <MODEL_ID>")
    sys.exit(1)

DATA_DIR = sys.argv[1]
MODEL_ID = sys.argv[2]

TRAIN_PATH = os.path.join(DATA_DIR, "full_train_pairs.jsonl")
VAL_PATH   = os.path.join(DATA_DIR, "full_val_pairs.jsonl")
OUTPUT_DIR = os.path.join(DATA_DIR, "output_mnr")
BEST_DIR   = os.path.join(OUTPUT_DIR, "best_model")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- CONFIG ----------
EPOCHS        = 30
LR            = 2e-5
BATCH_SIZE    = 4          # must be >= 2 for MN-RL
MAX_SEQ_LEN   = 512
EVAL_STEPS    = 500
PATIENCE      = 3
WARMUP_PCT    = 0.1
LOG_EVERY     = 50
TEMPERATURE   = 0.05       # standard value for MN-RL / InfoNCE-style
# -----------------------------

assert BATCH_SIZE >= 2, "MultipleNegativesRankingLoss requires batch_size >= 2"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True

print(f"Model: {MODEL_ID}")
print(f"Data dir: {DATA_DIR}")
print(f"Output dir: {OUTPUT_DIR}")

# ---------- Load (query, positive) pairs ----------
def load_pairs(path):
    """
    Expects JSONL lines with:
      - query_text: str
      - positive_path: str (file path to the target doc text, WITHOUT metadata)
    Ignores negative_paths (MN-RL uses in-batch negatives implicitly).
    """
    pairs = []
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return pairs
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                ex = json.loads(line)
                q = ex.get("query_text", "")
                pos_path = ex.get("positive_path")
                if not q or not pos_path or not os.path.exists(pos_path):
                    continue
                with open(pos_path, "r", encoding="utf-8") as pf:
                    pos = pf.read()
                # Basic guard: skip if query equals positive (bad pair construction)
                if q.strip() and pos.strip() and q.strip() != pos.strip():
                    pairs.append((q, pos))
            except Exception as e:
                print(f"Skipping line {i}: {e}")
                continue
    return pairs

print("Loading training data...")
train_pairs = load_pairs(TRAIN_PATH)
print(f"Loaded {len(train_pairs)} (query, positive) pairs for training")

print("Loading validation data...")
val_pairs = load_pairs(VAL_PATH)
print(f"Loaded {len(val_pairs)} (query, positive) pairs for validation")

# ---------- Load Model ----------

# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,         # change to load_in_4bit=True if still OOM
#     llm_int8_threshold=6.0,
#     llm_int8_has_fp16_weights=False
# )



print("Loading model...")
# model = SentenceTransformer(MODEL_ID, device="cuda", trust_remote_code=True)

model = SentenceTransformer(
    MODEL_ID,
    device="cuda",
    trust_remote_code=True,
    model_kwargs={"torch_dtype": torch.bfloat16}
)


# Cap max sequence length
try:
    cfg = model._first_module().auto_model.config
    max_len_model = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "max_sequence_length", None)
except Exception:
    max_len_model = None

if max_len_model:
    model.max_seq_length = min(MAX_SEQ_LEN, max_len_model)
    print(f"Model supports {max_len_model} tokens → using {model.max_seq_length}")
else:
    model.max_seq_length = MAX_SEQ_LEN
    print(f"Using default max sequence length: {MAX_SEQ_LEN}")

try:
    model._first_module().auto_model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled.")
except Exception:
    print("Gradient checkpointing not available.")

model.to("cuda")
model.train()

# ---------- DataLoaders ----------
train_loader = DataLoader(train_pairs, shuffle=True,  batch_size=BATCH_SIZE, collate_fn=lambda x: x)
val_loader   = DataLoader(val_pairs,   shuffle=False, batch_size=BATCH_SIZE, collate_fn=lambda x: x)

# ---------- Optimizer & Scheduler ----------
steps_per_epoch = max(1, math.ceil(len(train_loader)))
total_steps  = steps_per_epoch * EPOCHS
warmup_steps = int(steps_per_epoch * WARMUP_PCT)

optimizer = AdamW(model.parameters(), lr=LR)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

# ---------- Helpers ----------
def _to_device(batch_dict, device):
    for k, v in batch_dict.items():
        if torch.is_tensor(v):
            batch_dict[k] = v.to(device, non_blocking=True)
    return batch_dict

def encode(model, texts):
    batch = model.tokenize(texts)
    batch = _to_device(batch, model.device)
    out = model(batch)
    return out["sentence_embedding"]  # L2-normalized by ST models (usually)

def mnr_loss(emb_q, emb_p, temperature=TEMPERATURE, symmetric=True):
    """
    MultipleNegativesRankingLoss (InfoNCE-style):
      - Similarity matrix S = cos(q_i, p_j) (here dot equals cosine if normalized).
      - CE over diagonal as targets.
      - Optionally symmetric (both q->p and p->q).
    """
    # Cosine similarity of normalized embeddings == dot product
    sim = torch.matmul(emb_q, emb_p.t())  # [B, B]
    logits = sim / temperature
    labels = torch.arange(emb_q.size(0), device=emb_q.device)

    loss_qp = F.cross_entropy(logits, labels)
    if symmetric:
        loss_pq = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_qp + loss_pq)
    return loss_qp

@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total, count = 0.0, 0
    for batch in dataloader:
        if len(batch) < 2:
            continue  # MN-RL needs >= 2 for meaningful loss
        q_texts = [x[0] for x in batch]
        p_texts = [x[1] for x in batch]
        q = encode(model, q_texts)
        p = encode(model, p_texts)
        # q = F.normalize(encode(model, q_texts), p=2, dim=1)
        # p = F.normalize(encode(model, p_texts), p=2, dim=1)
        loss_val = mnr_loss(q, p).item()
        total += loss_val
        count += 1
    model.train()
    return total / max(1, count)

def save_best(model, path):
    os.makedirs(path, exist_ok=True)
    model.save(path)
    try:
        model._first_module().auto_model.save_pretrained(path)
        print(f"HuggingFace weights saved to {os.path.join(path, 'pytorch_model.bin')}")
    except Exception as e:
        print(f"(Non-fatal) Could not save HF model: {e}")

# ---------- Training Loop (MN-RL) ----------
print("Starting fine-tuning (MultipleNegativesRankingLoss) with early stopping...")
best_val = float("inf")
no_improve = 0
global_step = 0

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    print(f"Starting epoch {epoch+1}/{EPOCHS}")
    for step, batch in enumerate(train_loader):
        if len(batch) < 2:
            # Skip degenerate last batch
            continue

        q_texts = [x[0] for x in batch]
        p_texts = [x[1] for x in batch]

        q = encode(model, q_texts)
        p = encode(model, p_texts)

        # --- Diagnostics ---
        if step % 20 == 0:
            # Diagonal = true pairs similarity
            diag_sim = (q * p).sum(dim=1).mean().item()  # mean cos for matched pairs
            # Hardest imposter similarity
            with torch.no_grad():
                sim_matrix = torch.matmul(q, p.t())
                mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
                hardest_neg = sim_matrix[mask].view(sim_matrix.size(0), -1).max(dim=1).values.mean().item()
            print(f"[Step {step}] mean cos(q,p)={diag_sim:.4f} | mean hardest-neg={hardest_neg:.4f} | T={TEMPERATURE}")
            print("Q requires_grad:", q.requires_grad, "| grad_fn:", q.grad_fn)
        # --------------------

        optimizer.zero_grad(set_to_none=True)
        loss = mnr_loss(q, p)
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        global_step += 1

        if global_step % LOG_EVERY == 0:
            torch.cuda.empty_cache(); gc.collect()
            print(f"Epoch {epoch+1}/{EPOCHS} | Step {global_step} | Train loss: {loss.item():.4f}")

        if global_step % EVAL_STEPS == 0 and len(val_pairs) > 0:
            val_loss = evaluate(model, val_loader)
            print(f"Validation @ step {global_step}: {val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                save_best(model, BEST_DIR)
                print(f"New BEST saved → {BEST_DIR} (val_loss={val_loss:.6f})")
            else:
                no_improve += 1
                print(f"No improvement for {no_improve} evals (best={best_val:.6f})")
                if no_improve >= PATIENCE:
                    print(f"Early stopping triggered after {no_improve} non-improving evals.")
                    if not os.path.exists(BEST_DIR):
                        save_best(model, BEST_DIR)
                    sys.exit(0)

    avg_train = epoch_loss / max(1, len(train_loader))
    print(f"Epoch {epoch+1}/{EPOCHS} finished | avg train loss: {avg_train:.6f}")

if not os.path.exists(BEST_DIR):
    save_best(model, BEST_DIR)

print(f"Training complete. Best model at: {BEST_DIR}")


# Models to try:
# Qwen/Qwen3-Embedding-0.6B     16bit / batch 2
# Salesforce/SFR-Embedding-Mistral 16 / batch 2
# Qwen/Qwen3-Embedding-4B       16bit / batch 2
# Linq-AI-Research/Linq-Embed-Mistral 16 bit  / batch 2
# intfloat/multilingual-e5-large-instruct max_token = 512 / batch 4


# CUDA_VISIBLE_DEVICES=3 python fine_tune_full.py    /ltstorage/home/4baba/EUR_lex/embedding_model/fine_utning/tunning_data/EN_multilingual-e5-large-instruct  intfloat/multilingual-e5-large-instruct