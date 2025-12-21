import pandas as pd
from pathlib import Path
from collections import defaultdict

def analyze_retrieval(input_csv: str, result_csv: str, allow_other_langs: bool = False):
    """
    Analyze retrieval accuracy (Top-1/3/5) from a CSV and save year-wise + overall results.
    
    Args:
        input_csv (str): Path to the input CSV containing retrieval results.
        result_csv (str): Path where the analysis CSV should be saved.
        allow_other_langs (bool): 
            - False (default): only the exact language match counts as correct.
            - True: any language variant of the same document (same year + law + doc ID) counts as correct.
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

    def base_doc_key(key: str) -> str:
        """
        Convert 'YYYY/lawXXXX/32007D0364_LV' -> 'YYYY/lawXXXX/32007D0364'.
        Used when allow_other_langs=True to ignore language suffixes.
        """
        if not isinstance(key, str) or not key:
            return ""
        parts = key.split("/")
        last = parts[-1]
        # If there's a language suffix like _LV, _EN, etc., drop it
        if "_" in last:
            last = last.split("_", 1)[0]
        parts[-1] = last
        return "/".join(parts)

    df = pd.read_csv(input_csv)

    # --- Normalize paths ---
    df["key_meta"] = df["metadata_filepath"].apply(extract_key)
    for i in range(1, 6):
        df[f"key_{i}"] = df[f"closest_{i}"].apply(extract_key)

    # --- Initialize counters ---
    total = len(df)
    top1 = top3 = top5 = 0
    per_year = defaultdict(lambda: {"total": 0, "top1": 0, "top3": 0, "top5": 0})

    for _, row in df.iterrows():
        key_meta = row["key_meta"]
        if not key_meta:
            continue

        # Year = first segment of the key: '2007/law.../...' -> '2007'
        year = key_meta.split("/")[0]
        per_year[year]["total"] += 1

        # Build candidate list
        candidates = [
            row[f"key_{i}"] 
            for i in range(1, 6) 
            if isinstance(row[f"key_{i}"], str)
        ]

        correct_rank = None

        if allow_other_langs:
            # Compare by document base ID, ignoring language suffix
            base_meta = base_doc_key(key_meta)
            for idx, c in enumerate(candidates, start=1):
                if base_doc_key(c) == base_meta:
                    correct_rank = idx
                    break
        else:
            # Exact key match (same language)
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

    def pct(n): 
        return 100 * n / total if total else 0

    overall = {
        "year": "ALL",
        "top1": pct(top1),
        "top3": pct(top3),
        "top5": pct(top5),
        "total": total,
    }

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

    pd.DataFrame(year_rows).to_csv(result_csv, index=False)

    mode_str = "MULTI-LANG (accept all languages for same doc)" if allow_other_langs else "EXACT (same language only)"
    print("Accuracy evaluation for embedding retrieval model")
    print(f"Mode: {mode_str}")
    print(f"Input file: {input_csv}")
    print(f"Total samples: {total}")
    print(f"Top-1 accuracy: {pct(top1):.2f}%")
    print(f"Top-3 accuracy: {pct(top3):.2f}%")
    print(f"Top-5 accuracy: {pct(top5):.2f}%\n")
    print(f"Saved combined results (overall + per-year) to: {result_csv}")

