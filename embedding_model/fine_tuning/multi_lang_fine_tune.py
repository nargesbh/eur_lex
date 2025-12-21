#!/usr/bin/env python3
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

# ===========================================
# USAGE:
# CUDA_VISIBLE_DEVICES=3 python multi_lang_fine_tune.py \
#   /ltstorage/home/4baba/EUR_lex/embedding_model/fine_tuning/tunning_data/LV-EN/Qwen3_4B \
#   Qwen/Qwen3-Embedding-4B \
#   multi-EN-LV-Qwen3_4B
# ===========================================

if len(sys.argv) < 3:
    print("Usage: python fine_tune_full_grouped.py <DATA_DIR> <MODEL_ID> [RUN_NAME]")
    sys.exit(1)

DATA_DIR = sys.argv[1]
MODEL_ID = sys.argv[2]

# Optional run name
if len(sys.argv) >= 4:
    RUN_NAME = sys.argv[3]
else:
    RUN_NAME = f"{MODEL_ID.replace('/','_')}_MNR_GROUP_{int(time.time())}"

TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
VAL_PATH   = os.path.join(DATA_DIR, "val.jsonl")
OUTPUT_DIR = os.path.join(DATA_DIR, "output_mnr")
BEST_DIR   = os.path.join(OUTPUT_DIR, "best_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- CONFIG ----------
EPOCHS        = 30
LR            = 5e-5
BATCH_SIZE    = 8
MAX_SEQ_LEN   = 512
EVAL_STEPS    = 500
PATIENCE      = 3
WARMUP_PCT    = 0.1
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

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Model: {MODEL_ID}")
print(f"Data dir: {DATA_DIR}")
print(f"Output dir: {OUTPUT_DIR}")

# ----------------- wandb init -----------------
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
            "loss": "Multi-positive MNR with grouped positives (symmetric, in-batch)",
        },
        settings=wandb.Settings(start_method="thread"),
    )

# ---------- Load data ----------
def load_pairs(path):
    """
    Returns list of (query_text, [pos_text_1, pos_text_2, ...], group_key).

    group_key = tuple(sorted(positive_path_list)) to identify
    the EN+LV pair (or multi-path set).
    """
    pairs = []
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return pairs

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                q = (ex.get("query_text") or "").strip()
                if not q:
                    continue

                pos_paths_raw = ex.get("positive_path")
                if isinstance(pos_paths_raw, str):
                    pos_paths = [pos_paths_raw]
                elif isinstance(pos_paths_raw, list):
                    pos_paths = pos_paths_raw
                else:
                    continue

                pos_paths_clean = [str(p).strip() for p in pos_paths if p]
                if not pos_paths_clean:
                    continue

                group_key = tuple(sorted(pos_paths_clean))

                pos_texts = []
                for pth in pos_paths_clean:
                    if not os.path.exists(pth):
                        continue
                    try:
                        with open(pth, "r", encoding="utf-8") as pf:
                            txt = pf.read().strip()
                        if txt and txt != q:
                            pos_texts.append(txt)
                    except Exception as e:
                        print(f"[WARN] Could not read positive_path '{pth}': {e}")
                        continue

                if q and pos_texts:
                    pairs.append((q, pos_texts, group_key))
            except Exception as e:
                print(f"Skipping line {i} in {path}: {e}")
                continue
    return pairs

print("Loading training data...")
train_pairs = load_pairs(TRAIN_PATH)
print(f"Loaded {len(train_pairs)} training triples")

print("Loading validation data...")
val_pairs = load_pairs(VAL_PATH)
print(f"Loaded {len(val_pairs)} validation triples")

if len(train_pairs) < 2:
    print("Need at least 2 training examples.")
    sys.exit(1)

# ---------- Model ----------
print("Loading model...")
model = SentenceTransformer(
    MODEL_ID,
    device=device,
    trust_remote_code=True,
    model_kwargs={"torch_dtype": torch.bfloat16} if device == "cuda" else {}
)

try:
    cfg = model._first_module().auto_model.config
    max_len_model = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "max_sequence_length", None)
except Exception:
    max_len_model = None

model.max_seq_length = min(MAX_SEQ_LEN, max_len_model) if max_len_model else MAX_SEQ_LEN
print(f"Using max_seq_length={model.max_seq_length}")

try:
    model._first_module().auto_model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled.")
except Exception:
    print("Gradient checkpointing not available.")

model.to(device)
model.train()

if _USE_WANDB:
    try:
        wandb.watch(model, log="gradients", log_freq=200)
    except Exception:
        pass

# ---------- DataLoaders ----------
train_loader = DataLoader(
    train_pairs,
    shuffle=True,
    batch_size=BATCH_SIZE,
    collate_fn=lambda x: x,
    drop_last=True,
)
val_loader = DataLoader(
    val_pairs,
    shuffle=False,
    batch_size=BATCH_SIZE,
    collate_fn=lambda x: x,
    drop_last=True,
)

# ---------- Optimizer & Scheduler ----------
steps_per_epoch = len(train_loader)
total_steps     = max(1, steps_per_epoch * EPOCHS)
warmup_steps    = int(total_steps * WARMUP_PCT)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
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
    emb = F.normalize(emb, p=2, dim=1)
    return emb

def mnr_loss_grouped(emb_q, emb_p, owner, query_group_ids, doc_group_ids,
                     temperature=TEMPERATURE, symmetric=True):
    """
    Multi-positive MNR with "grouped" positives.

    For query i:
      POSITIVES = all docs j with doc_group_ids[j] == query_group_ids[i]
      NEGATIVES = all other docs in the batch
    """
    owner = owner.to(emb_q.device)
    query_group_ids = query_group_ids.to(emb_q.device)
    doc_group_ids   = doc_group_ids.to(emb_q.device)

    sim_qp = torch.matmul(emb_q, emb_p.t()) / temperature    # [B,M]
    B, M = sim_qp.size()

    pos_mask = (doc_group_ids.unsqueeze(0) == query_group_ids.unsqueeze(1))  # [B,M]

    sim_qp_max = sim_qp.max(dim=1, keepdim=True).values
    sim_qp_stable = sim_qp - sim_qp_max

    exp_all = torch.exp(sim_qp_stable)
    exp_pos = exp_all * pos_mask

    sum_exp_all = exp_all.sum(dim=1)
    sum_exp_pos = exp_pos.sum(dim=1)

    valid = sum_exp_pos > 0
    loss_q = torch.zeros_like(sum_exp_pos, device=sim_qp.device)
    loss_q[valid] = -torch.log(sum_exp_pos[valid] / sum_exp_all[valid])

    if valid.any():
        loss_q = loss_q[valid].mean()
    else:
        loss_q = torch.tensor(0.0, device=sim_qp.device)

    if not symmetric:
        return loss_q

    sim_pq = torch.matmul(emb_p, emb_q.t()) / temperature    # [M,B]
    loss_p = F.cross_entropy(sim_pq, owner)

    return 0.5 * (loss_q + loss_p)

@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total, count = 0.0, 0
    for batch in dataloader:
        if len(batch) < 2:
            continue

        q_texts    = [x[0] for x in batch]
        pos_lists  = [x[1] for x in batch]
        group_keys = [x[2] for x in batch]

        # Per-batch group-id mapping
        group_id_map = {}
        def get_gid(key):
            if key not in group_id_map:
                group_id_map[key] = len(group_id_map)
            return group_id_map[key]

        query_group_ids_list = [get_gid(k) for k in group_keys]

        flat_pos_texts = []
        owner_list = []
        doc_group_ids_list = []
        for qi, (q, pos_texts, key) in enumerate(batch):
            gid = get_gid(key)
            for ptxt in pos_texts:
                flat_pos_texts.append(ptxt)
                owner_list.append(qi)
                doc_group_ids_list.append(gid)

        if not flat_pos_texts:
            continue

        owner = torch.tensor(owner_list, dtype=torch.long, device=device)
        query_group_ids = torch.tensor(query_group_ids_list, dtype=torch.long, device=device)
        doc_group_ids   = torch.tensor(doc_group_ids_list, dtype=torch.long, device=device)

        q = encode(model, q_texts)
        p = encode(model, flat_pos_texts)

        total += mnr_loss_grouped(q, p, owner, query_group_ids, doc_group_ids).item()
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
print("Starting fine-tuning (Grouped multi-positive MNR) with early stopping...")
best_val = float("inf")
no_improve = 0
global_step = 0

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    print(f"Starting epoch {epoch+1}/{EPOCHS}")
    for step, batch in enumerate(train_loader, start=1):
        q_texts    = [x[0] for x in batch]
        pos_lists  = [x[1] for x in batch]
        group_keys = [x[2] for x in batch]

        # Per-batch group-id mapping
        group_id_map = {}
        def get_gid(key):
            if key not in group_id_map:
                group_id_map[key] = len(group_id_map)
            return group_id_map[key]

        query_group_ids_list = [get_gid(k) for k in group_keys]

        flat_pos_texts = []
        owner_list = []
        doc_group_ids_list = []
        for qi, (q, pos_texts, key) in enumerate(batch):
            gid = get_gid(key)
            for ptxt in pos_texts:
                flat_pos_texts.append(ptxt)
                owner_list.append(qi)
                doc_group_ids_list.append(gid)

        if not flat_pos_texts:
            continue

        owner = torch.tensor(owner_list, dtype=torch.long, device=device)
        query_group_ids = torch.tensor(query_group_ids_list, dtype=torch.long, device=device)
        doc_group_ids   = torch.tensor(doc_group_ids_list, dtype=torch.long, device=device)

        q = encode(model, q_texts)
        p = encode(model, flat_pos_texts)

        # Logging sims every 20 steps
        if step % 20 == 0:
            with torch.no_grad():
                sim_qp = torch.matmul(q, p.t())
                pos_mask = (doc_group_ids.unsqueeze(0) == query_group_ids.unsqueeze(1))
                neg_mask = ~pos_mask

                if pos_mask.any():
                    mean_pos = sim_qp[pos_mask].mean().item()
                else:
                    mean_pos = float("nan")

                if neg_mask.any():
                    neg_sims = sim_qp.masked_fill(~neg_mask, float("-inf"))
                    hardest_neg = neg_sims.max(dim=1).values.mean().item()
                else:
                    hardest_neg = float("nan")

            print(f"[Step {step}] mean_pos={mean_pos:.4f} | mean_hard_neg={hardest_neg:.4f} | T={TEMPERATURE}")
            if _USE_WANDB:
                wandb.log({
                    "train/mean_pos_sim": float(mean_pos),
                    "train/mean_neg_sim": float(hardest_neg)
                                         if not math.isnan(hardest_neg) else float("nan"),
                    "train/sim_gap": float(mean_pos - hardest_neg)
                                     if (not math.isnan(mean_pos) and not math.isnan(hardest_neg)) else float("nan"),
                    "step": global_step,
                    "epoch": epoch + 1,
                })

        optimizer.zero_grad(set_to_none=True)
        loss = mnr_loss_grouped(q, p, owner, query_group_ids, doc_group_ids)
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
                    print("Early stopping triggered.")
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
