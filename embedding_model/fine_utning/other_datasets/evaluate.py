#!/usr/bin/env python3
import os, torch, mteb
from sentence_transformers import SentenceTransformer


# MODEL_ID = "/ltstorage/home/4baba/EUR_lex/embedding_model/fine_utning/tunning_data/EN_multilingual-e5-large-instruct/output_mnr/best_model"  
# OUTDIR   = "results_GerDaLIRSmall/EN_multilingual-e5-large-instruct/FineTuned"

# MODEL_ID = "intfloat/multilingual-e5-large-instruct"  
# OUTDIR   = "results_GerDaLIRSmall/EN_multilingual-e5-large-instruct/Original"


MODEL_ID = "/ltstorage/home/4baba/EUR_lex/embedding_model/fine_utning/tunning_data/EN_Qwen3_Embedding_06B/output_mnr/best_model"  
OUTDIR   = "results_GerDaLIRSmall/EN_Qwen3_Embedding_06B/FineTuned_2"

# MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"  
# OUTDIR   = "results_GerDaLIRSmall/EN_Qwen3_Embedding_06B/Original"

# MODEL_ID = "/ltstorage/home/4baba/EUR_lex/embedding_model/fine_utning/tunning_data/EN_Qwen3-Embedding-4B/output_mnr/best_model"  
# OUTDIR   = "results_GerDaLIRSmall/EN_Qwen3-Embedding-4B/FineTuned"

# MODEL_ID = "Qwen/Qwen3-Embedding-4B"  
# OUTDIR   = "results_GerDaLIRSmall/EN_Qwen3-Embedding-4B/Original"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_grad_enabled(False)

tasks = mteb.get_tasks(["GerDaLIRSmall"])
evaluator = mteb.MTEB(tasks=tasks)

model = SentenceTransformer(
    MODEL_ID,
    device="cuda",
    trust_remote_code=True,
    model_kwargs={"dtype": torch.float16},  
)

results = evaluator.run(
    model,
    output_folder=OUTDIR,
    model_name=MODEL_ID,                
    revision="main",                    
    encode_kwargs={
        "batch_size": 4,                
        "convert_to_tensor": True,
        "normalize_embeddings": True,
        "show_progress_bar": True,
    },
    overwrite_results=True,
)

print("Results written under:", os.path.abspath(OUTDIR))
