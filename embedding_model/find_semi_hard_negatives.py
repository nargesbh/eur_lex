import csv
import argparse
import os
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def skip_first_n_words(text: str, n: int) -> str:
    """Return text after the first n whitespace-delimited words."""
    if not text:
        return ""
    count = 0
    for m in re.finditer(r"\S+", text):
        count += 1
        if count == n:
            return text[m.end():].lstrip()
    # fewer than n words → empty
    return ""


def main(input_csv: str, output_csv: str, skip_n: int = 5, flush_every: int = 1000):
    # Load (metadata, filepath)
    df = pd.read_csv(input_csv)
    texts_full = df["metadata"].fillna("").astype(str).tolist()
    paths = df["filepath"].astype(str).tolist()
    n = len(paths)

    # Early out (but still create file with header)
    out_path = Path(output_csv)
    need_header = not out_path.exists()
    f = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f)
    if need_header:
        writer.writerow(["filepath1", "filepath2", "similarity_full", "similarity_skip5"])
        f.flush()
        os.fsync(f.fileno())

    if n < 2:
        f.close()
        return

    # Build texts with first N words removed
    texts_skip = [skip_first_n_words(t, skip_n) for t in texts_full]

    # Two TF-IDF spaces
    vec_full = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    tfidf_full = vec_full.fit_transform(texts_full)

    vec_skip = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    tfidf_skip = vec_skip.fit_transform(texts_skip)

    total_pairs = n * (n - 1) // 2
    written = 0

    with tqdm(total=total_pairs, desc="Computing pairwise cosine (full & skip5)") as pbar:
        for i in range(n - 1):
            # Compute similarities vs. the tail block in one shot
            sims_full = cosine_similarity(tfidf_full[i], tfidf_full[i + 1:]).ravel()
            sims_skip = cosine_similarity(tfidf_skip[i], tfidf_skip[i + 1:]).ravel()

            for offset, (sim_f, sim_s) in enumerate(zip(sims_full, sims_skip), start=i + 1):
                writer.writerow([
                    paths[i],
                    paths[offset],
                    round(float(sim_f), 4),
                    round(float(sim_s), 4),
                ])
                written += 1

                # Periodic flush to really write to disk “on the fly”
                if written % flush_every == 0:
                    f.flush()
                    os.fsync(f.fileno())

            pbar.update(n - (i + 1))

    # Final flush
    f.flush()
    os.fsync(f.fileno())
    f.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Cosine similarity between metadata strings (full vs. after skipping first N words), streaming results to CSV."
    )
    ap.add_argument("--input-csv", required=True, help="CSV with columns: metadata, filepath")
    ap.add_argument("--output-csv", required=True, help="Output CSV path")
    ap.add_argument("--skip-n", type=int, default=5, help="Words to skip for the second similarity.")
    ap.add_argument("--flush-every", type=int, default=1000, help="Flush to disk every K rows.")
    args = ap.parse_args()
    main(args.input_csv, args.output_csv, args.skip_n, args.flush_every)
