#!/usr/bin/env python3
import os
import torch
import argparse
import mteb
from sentence_transformers import SentenceTransformer


def run_mteb_eval(model_id: str, outdir: str, task_names: list[str]):
    """Run MTEB evaluation using given tasks and model."""

    print("\n==============================")
    print("     MTEB EVALUATION START    ")
    print("==============================")
    print(f"Model ID: {model_id}")
    print(f"Output Dir: {outdir}")
    print(f"Tasks: {task_names}\n")
    print("==============================\n")

    # Disable tokenizer parallelism & gradients
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_grad_enabled(False)

    # Load task(s)
    print(">>> Loading tasks...")
    tasks = mteb.get_tasks(task_names)
    evaluator = mteb.MTEB(tasks=tasks)

    # Load model
    print(">>> Loading model...")
    model = SentenceTransformer(
        model_id,
        device="cuda",
        trust_remote_code=True,
        model_kwargs={"dtype": torch.float16},
    )

    # Run evaluation
    print(">>> Running evaluation...")
    results = evaluator.run(
        model,
        output_folder=outdir,
        model_name=model_id,
        revision="main",
        encode_kwargs={
            "batch_size": 4,
            "convert_to_tensor": True,
            "normalize_embeddings": True,
            "show_progress_bar": True,
        },
        overwrite_results=True,
    )

    print(f"\n✔ Results saved to: {os.path.abspath(outdir)}\n")
    return results


# ---------------------
#      CLI PARSER
# ---------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MTEB evaluation for any embedding model."
    )

    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Path or HuggingFace model ID",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for results",
    )

    parser.add_argument(
        "--task",
        nargs="+",
        required=True,
        help="One or more MTEB tasks (e.g., GerDaLIRSmall LegalQuAD)",
    )

    args = parser.parse_args()

    run_mteb_eval(args.model_id, args.outdir, args.task)
