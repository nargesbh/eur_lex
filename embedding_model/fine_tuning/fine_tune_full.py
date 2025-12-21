
import os
import sys
import json
import gc
import math
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# ----------------- W&B -----------------
try:
    import wandb
    _USE_WANDB = True
except Exception:
    wandb = None
    _USE_WANDB = False
# --------------------------------------


# CUDA_VISIBLE_DEVICES=3 python fine_tune_full.py \
#   embedding_model/fine_utning/tunning_data/DE/Qwen3_4B \
#   Qwen/Qwen3-Embedding-4B \
#   DE-Qwen3-4B


if len(sys.argv) < 3:
    print("Usage: python fine_tune_full.py <DATA_DIR> <MODEL_ID> [RUN_NAME]")
    sys.exit(1)

DATA_DIR = sys.argv[1]
MODEL_ID = sys.argv[2]

# Optional run name
if len(sys.argv) >= 4:
    RUN_NAME = sys.argv[3]
else:
    RUN_NAME = f"{MODEL_ID.replace('/','_')}_MNR_{int(time.time())}"


TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
VAL_PATH   = os.path.join(DATA_DIR, "val.jsonl")
OUTPUT_DIR = os.path.join(DATA_DIR, "output_mnr")
BEST_DIR   = os.path.join(OUTPUT_DIR, "best_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- CONFIG ----------
EPOCHS        = 30
LR            = 5e-5
BATCH_SIZE    = 8           # >=2 for in-batch negatives
MAX_SEQ_LEN   = 512         # tokenizer truncates: "first max token length"
EVAL_STEPS    = 500
PATIENCE      = 3
WARMUP_PCT    = 0.1         # fraction of total steps
LOG_EVERY     = 50
TEMPERATURE   = 0.07
SEED          = 42
# -----------------------------

assert BATCH_SIZE >= 2, "MultipleNegativesRankingLoss requires batch_size >= 2"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"Model: {MODEL_ID}")
print(f"Data dir: {DATA_DIR}")
print(f"Output dir: {OUTPUT_DIR}")

# ----------------- wandb init -----------------
# run_name = f"{MODEL_ID.replace('/','_')}_MNR_{int(time.time())}"

if _USE_WANDB:
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "law-embedding-finetune"),
        entity=os.getenv("WANDB_ENTITY", None),
        name=RUN_NAME,
        config={
            "model_id": MODEL_ID,
            "data_dir": DATA_DIR,
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "max_seq_len": MAX_SEQ_LEN,
            "eval_steps": EVAL_STEPS,
            "patience": PATIENCE,
            "warmup_pct": WARMUP_PCT,
            "temperature": TEMPERATURE,
            "precision": "bf16",
            "loss": "MultipleNegativesRankingLoss (symmetric, in-batch)",
        },
        settings=wandb.Settings(start_method="thread"),
    )

# ---------- Load (query, positive) pairs ----------
def load_pairs(path):
    pairs = []
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return pairs
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                ex = json.loads(line)
                q = (ex.get("query_text") or "").strip()
                pos_path = ex.get("positive_path")
                if not q or not pos_path or not os.path.exists(pos_path):
                    continue
                with open(pos_path, "r", encoding="utf-8") as pf:
                    pos = pf.read().strip()
                if q and pos and q != pos:
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

if len(train_pairs) < 2:
    print("Need at least 2 training pairs for in-batch negatives.")
    sys.exit(1)

# ---------- Model ----------
print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(
    MODEL_ID,
    device=device,
    trust_remote_code=True,
    model_kwargs={"torch_dtype": torch.bfloat16} if device == "cuda" else {}
)

# Cap max sequence length (tokenizer handles truncation)
try:
    cfg = model._first_module().auto_model.config
    max_len_model = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "max_sequence_length", None)
except Exception:
    max_len_model = None

model.max_seq_length = min(MAX_SEQ_LEN, max_len_model) if max_len_model else MAX_SEQ_LEN
print(f"Using max_seq_length={model.max_seq_length}")

# Try to enable gradient checkpointing (saves memory)
try:
    model._first_module().auto_model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled.")
except Exception:
    print("Gradient checkpointing not available.")

model.to(device)
model.train()

# ----------------- wandb watch -----------------
if _USE_WANDB:
    try:
        wandb.watch(model, log="gradients", log_freq=200)
    except Exception:
        pass

# ---------- DataLoaders ----------
train_loader = DataLoader(train_pairs, shuffle=True,  batch_size=BATCH_SIZE, collate_fn=lambda x: x, drop_last=True)
val_loader   = DataLoader(val_pairs,   shuffle=False, batch_size=BATCH_SIZE, collate_fn=lambda x: x, drop_last=True)

# ---------- Optimizer & Scheduler ----------
steps_per_epoch = len(train_loader)
total_steps     = max(1, steps_per_epoch * EPOCHS)
warmup_steps    = int(total_steps * WARMUP_PCT)    

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

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
    emb = out["sentence_embedding"]
    # L2-normalize → cosine sim with dot product
    emb = F.normalize(emb, p=2, dim=1)
    return emb

def mnr_loss(emb_q, emb_p, temperature=TEMPERATURE, symmetric=True):
    # cosine-sim matrix via normalized dot product
    sim = torch.matmul(emb_q, emb_p.t())        # [B, B]
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
            continue
        q_texts = [x[0] for x in batch]
        p_texts = [x[1] for x in batch]
        q = encode(model, q_texts)
        p = encode(model, p_texts)
        total += mnr_loss(q, p).item()
        count += 1
    model.train()
    return total / max(1, count)

def save_best(model, path):
    os.makedirs(path, exist_ok=True)
    model.save(path)
    try:
        model._first_module().auto_model.save_pretrained(path)
        model.tokenizer.save_pretrained(path)
        print(f"HuggingFace weights saved to {os.path.join(path, 'pytorch_model.bin')}")
    except Exception as e:
        print(f"(Non-fatal) Could not save HF model: {e}")

# ---------- Training Loop ----------
print("Starting fine-tuning (MultipleNegativesRankingLoss) with early stopping...")
best_val = float("inf")
no_improve = 0
global_step = 0

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    print(f"Starting epoch {epoch+1}/{EPOCHS}")
    for step, batch in enumerate(train_loader, start=1):
        q_texts = [x[0] for x in batch]
        p_texts = [x[1] for x in batch]

        q = encode(model, q_texts)
        p = encode(model, p_texts)

        if step % 20 == 0:
            with torch.no_grad():
                pos_cos = (q * p).sum(dim=1).mean().item()
                sim_matrix = torch.matmul(q, p.t())
                mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
                hardest_neg = sim_matrix[mask].view(sim_matrix.size(0), -1).max(dim=1).values.mean().item()
            print(f"[Step {step}] mean_pos={pos_cos:.4f} | mean_hard_neg={hardest_neg:.4f} | T={TEMPERATURE}")
            if _USE_WANDB:
                wandb.log({
                    "train/mean_pos_sim": float(pos_cos),
                    "train/mean_neg_sim": float(hardest_neg),
                    "train/sim_gap": float(pos_cos - hardest_neg),
                    "step": global_step,
                    "epoch": epoch + 1,
                })

        optimizer.zero_grad(set_to_none=True)
        loss = mnr_loss(q, p)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        global_step += 1

        if _USE_WANDB:
            try:
                lr_now = scheduler.get_last_lr()[0]
                wandb.log({
                    "train/loss": float(loss.item()),
                    "train/lr": float(lr_now),
                    "epoch": epoch + 1,
                    "step": global_step,
                })
            except Exception:
                pass

        if global_step % LOG_EVERY == 0:
            torch.cuda.empty_cache(); gc.collect()
            print(f"Epoch {epoch+1}/{EPOCHS} | Step {global_step} | Train loss: {loss.item():.4f}")

        if (global_step % EVAL_STEPS == 0) and (len(val_pairs) > 0):
            val_loss = evaluate(model, val_loader)
            print(f"Validation @ step {global_step}: {val_loss:.6f}")
            if _USE_WANDB:
                try:
                    wandb.log({"val/loss": float(val_loss), "step": global_step, "epoch": epoch + 1})
                except Exception:
                    pass

            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                save_best(model, BEST_DIR)
                print(f"New BEST saved → {BEST_DIR} (val_loss={val_loss:.6f})")
                if _USE_WANDB:
                    try:
                        art = wandb.Artifact("best_model", type="model", metadata={"val_loss": float(val_loss)})
                        art.add_dir(BEST_DIR)
                        wandb.log_artifact(art)
                    except Exception:
                        pass
            else:
                no_improve += 1
                print(f"No improvement for {no_improve} evals (best={best_val:.6f})")
                if no_improve >= PATIENCE:
                    print(f"Early stopping triggered after {no_improve} non-improving evals.")
                    if not os.path.exists(BEST_DIR):
                        save_best(model, BEST_DIR)
                    if _USE_WANDB:
                        wandb.summary["best_val_loss"] = float(best_val)
                    sys.exit(0)

    avg_train = epoch_loss / max(1, len(train_loader))
    print(f"Epoch {epoch+1}/{EPOCHS} finished | avg train loss: {avg_train:.6f}")

if not os.path.exists(BEST_DIR):
    save_best(model, BEST_DIR)

print(f"Training complete. Best model at: {BEST_DIR}")

if _USE_WANDB:
    wandb.summary["best_val_loss"] = float(best_val if best_val != float("inf") else -1.0)
    wandb.finish()
