import os, json, gc, torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from peft import get_peft_model, LoraConfig, TaskType
from transformers import BitsAndBytesConfig

# ---------------- CONFIG ----------------
MODEL_ID   = "Qwen/Qwen3-Embedding-0.6B"
TRAIN_PATH = "/ltstorage/home/4baba/EUR_lex/embedding_model/fine_utning/english_datasets/train_pairs.jsonl"
OUTPUT_DIR = "/ltstorage/home/4baba/EUR_lex/embedding_model/fine_utning/english_datasets/qwen3_0.6b_lora"
BATCH_SIZE = 2
EPOCHS     = 30
LR         = 2e-5
# ----------------------------------------

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"

# ---------- Load data ----------
def load_triplets(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            q = ex["query_text"]
            try:
                pos = open(ex["positive_path"], "r", encoding="utf-8").read()
            except: 
                continue
            for negp in ex["negative_paths"]:
                try:
                    neg = open(negp, "r", encoding="utf-8").read()
                    samples.append(InputExample(texts=[q, pos, neg]))
                except:
                    continue
    return samples

train_samples = load_triplets(TRAIN_PATH)
print(f"Loaded {len(train_samples)} triplets")

# ---------- Load model in 8-bit using new API ----------
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf4",
    llm_int8_threshold=6.0,
)

model = SentenceTransformer(
    MODEL_ID,
    device="cuda",
    trust_remote_code=True,
    model_kwargs={"quantization_config": bnb_config}
)

# ---------- Attach LoRA adapters ----------
peft_cfg = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    inference_mode=False
)

base_model = model._first_module().auto_model

# Add these two lines:
# base_model.gradient_checkpointing_enable()   # reduces activation memory
model.max_seq_length = 2048  

lora_model = get_peft_model(base_model, peft_cfg)
model._first_module().auto_model = lora_model

# Print trainable params safely
trainable, total = 0, 0
for name, p in lora_model.named_parameters():
    total += p.numel()
    if p.requires_grad:
        trainable += p.numel()
print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ---------- Prepare training ----------
train_loader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)
train_loss   = losses.MultipleNegativesRankingLoss(model)

# # ---------- Train with mixed precision ----------
# model.fit(
#     train_objectives=[(train_loader, train_loss)],
#     epochs=EPOCHS,
#     warmup_steps=int(len(train_loader)*0.1),
#     show_progress_bar=True,
#     output_path=OUTPUT_DIR,
#     optimizer_params={'lr': LR},
#     use_amp=True,          # fp16 mixed precision
#     callback=lambda step, **kwargs: (torch.cuda.empty_cache(), gc.collect()) if step % 5 == 0 else None
# )

# print(f"LoRA fine-tuning finished → {OUTPUT_DIR}")

# ---------- Custom callback ----------
def periodic_callback(step, epoch, num_steps_per_epoch, **kwargs):
    """Runs every training step. Saves after each epoch."""
    # Free some GPU cache
    if step % 5 == 0:
        torch.cuda.empty_cache()
        gc.collect()
    # Save at end of each epoch
    if (step + 1) % num_steps_per_epoch == 0:
        save_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
        model.save(save_dir)
        print(f"Saved model checkpoint at end of epoch {epoch+1}: {save_dir}")

# ---------- Train with mixed precision ----------
num_steps_per_epoch = len(train_loader)
total_steps = num_steps_per_epoch * EPOCHS
current_epoch = 0

def callback_wrapper(step, **kwargs):
    global current_epoch
    epoch = step // num_steps_per_epoch
    if epoch != current_epoch:
        current_epoch = epoch
    periodic_callback(step, epoch, num_steps_per_epoch, **kwargs)

model.fit(
    train_objectives=[(train_loader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=int(len(train_loader)*0.1),
    show_progress_bar=True,
    output_path=OUTPUT_DIR,
    optimizer_params={'lr': LR},
    use_amp=True,           # fp16 mixed precision
    callback=callback_wrapper
)

print(f" LoRA fine-tuning finished → {OUTPUT_DIR}")