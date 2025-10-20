import pandas as pd
from pathlib import Path
from collections import defaultdict

def analyze_retrieval(input_csv: str, result_csv: str):
    """
    Analyze retrieval accuracy (Top-1/3/5) from a CSV and save year-wise + overall results.
    
    Args:
        input_csv (str): Path to the input CSV containing retrieval results.
        result_csv (str): Path where the analysis CSV should be saved.
    """

    def extract_key(p: str) -> str:
        """Extract 'YYYY/lawXXXX/..._LANG' part from path."""
        if not isinstance(p, str) or not p:
            return ""
        parts = Path(p).parts
        for i, part in enumerate(parts):
            if part.isdigit() and len(part) == 4:
                return "/".join(parts[i:i+3]).replace(".jsonl", "").replace(".txt", "")
        return ""

    # --- Load data ---
    df = pd.read_csv(input_csv)

    # --- Normalize paths ---
    df["key_meta"] = df["metadata_filepath"].apply(extract_key)
    for i in range(1, 6):
        df[f"key_{i}"] = df[f"closest_{i}"].apply(extract_key)

    # --- Initialize counters ---
    total = len(df)
    top1 = top3 = top5 = 0
    per_year = defaultdict(lambda: {"total": 0, "top1": 0, "top3": 0, "top5": 0})

    # --- Evaluate ---
    for _, row in df.iterrows():
        key_meta = row["key_meta"]
        if not key_meta:
            continue
        year = key_meta.split("/")[0]
        per_year[year]["total"] += 1
        candidates = [row[f"key_{i}"] for i in range(1, 6) if isinstance(row[f"key_{i}"], str)]
        correct_rank = None
        for idx, c in enumerate(candidates, start=1):
            if c == key_meta:
                correct_rank = idx
                break
        if correct_rank:
            if correct_rank == 1:
                top1 += 1
                per_year[year]["top1"] += 1
            if correct_rank <= 3:
                top3 += 1
                per_year[year]["top3"] += 1
            if correct_rank <= 5:
                top5 += 1
                per_year[year]["top5"] += 1

    # --- Compute overall stats ---
    def pct(n): 
        return 100 * n / total if total else 0

    overall = {
        "year": "ALL",
        "top1": pct(top1),
        "top3": pct(top3),
        "top5": pct(top5),
        "total": total,
    }

    # --- Prepare year-wise data ---
    year_rows = [overall]
    for year, stats in sorted(per_year.items()):
        t = stats["total"]
        if t == 0:
            continue
        year_rows.append({
            "year": year,
            "top1": 100 * stats["top1"] / t,
            "top3": 100 * stats["top3"] / t,
            "top5": 100 * stats["top5"] / t,
            "total": t,
        })

    # --- Save CSV ---
    pd.DataFrame(year_rows).to_csv(result_csv, index=False)

    # --- Print summary ---
    print("Accuracy evaluation for embedding retrieval model")
    print(f"Input file: {input_csv}")
    print(f"Total samples: {total}")
    print(f"Top-1 accuracy: {pct(top1):.2f}%")
    print(f"Top-3 accuracy: {pct(top3):.2f}%")
    print(f"Top-5 accuracy: {pct(top5):.2f}%\n")
    print(f"Saved combined results (overall + per-year) to: {result_csv}")


# Example usage:
# analyze_retrieval(
#     "/ltstorage/home/4baba/EUR_lex/embedding_model/test_chromadb/EN_SFR-Embedding-Mistral/english_top5_retrieval.csv",
#     "/ltstorage/home/4baba/EUR_lex/embedding_model/test_chromadb/EN_SFR-Embedding-Mistral/analysis.csv"
# )