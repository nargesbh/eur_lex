import re
import difflib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List
from collections import Counter

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
import Levenshtein
from tqdm import tqdm
import argparse

@dataclass
class ExtractResult:
    file_name: str
    language: str
    celex_id: str
    html_text: str
    mmd_text: str
    metrics: Dict[str, float] = field(default_factory=dict)

class Evaluator:
    def clean_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(True):
            for attr in ["style", "class", "id", "width", "height"]:
                tag.attrs.pop(attr, None)
        return " ".join(soup.stripped_strings).lower()

    def clean_mmd(self, mmd: str) -> str:
        lines = [line for line in mmd.splitlines() if line.strip() and not line.strip().startswith("##")]
        return " ".join(lines).lower()

    def _sequence_similarity(self, t1: str, t2: str) -> float:
        return difflib.SequenceMatcher(None, t1, t2).ratio()

    def _jaccard_similarity(self, t1: str, t2: str) -> float:
        tokens1 = set(re.findall(r"\w+", t1))
        tokens2 = set(re.findall(r"\w+", t2))
        return len(tokens1 & tokens2) / len(tokens1 | tokens2) if tokens1 | tokens2 else 0.0

    def _levenshtein_similarity(self, t1: str, t2: str) -> float:
        max_len = max(len(t1), len(t2))
        return 1 - (Levenshtein.distance(t1, t2) / max_len) if max_len else 1.0

    def _content_similarity(self, t1: str, t2: str) -> float:
        w1, w2 = Counter(t1.split()), Counter(t2.split())
        common = set(w1) & set(w2)
        dot = sum(w1[w] * w2[w] for w in common)
        norm1 = sum(v*v for v in w1.values()) ** 0.5
        norm2 = sum(v*v for v in w2.values()) ** 0.5
        return dot / (norm1 * norm2) if norm1 and norm2 else 0.0

    def _f1_similarity(self, t1: str, t2: str) -> float:
        try:
            vectorizer = CountVectorizer(binary=True)
            x = vectorizer.fit_transform([t1, t2])
            return f1_score(x.toarray()[0], x.toarray()[1], average="binary")
        except:
            return 0.0

    def evaluate(self, html_text: str, mmd_text: str) -> Dict[str, float]:
        cleaned_html = self.clean_text(html_text)
        cleaned_mmd = self.clean_mmd(mmd_text)

        return {
            "sequence": self._sequence_similarity(cleaned_html, cleaned_mmd),
            "jaccard": self._jaccard_similarity(cleaned_html, cleaned_mmd),
            "levenshtein": self._levenshtein_similarity(cleaned_html, cleaned_mmd),
            "content": self._content_similarity(cleaned_html, cleaned_mmd),
            "content2": self._content_similarity(cleaned_html, cleaned_mmd),
            "f1": self._f1_similarity(cleaned_html, cleaned_mmd),
        }

def main():
    parser = argparse.ArgumentParser(description="Evaluate MMD vs HTML using multiple metrics.")
    parser.add_argument("mmd_dir", type=Path, help="Directory containing MMD files")
    parser.add_argument("html_dir", type=Path, help="Directory containing HTML ground truth files")
    parser.add_argument("output_csv", type=Path, help="Output CSV path")
    args = parser.parse_args()

    evaluator = Evaluator()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    headers = ["Filepath", "CELEX ID", "Language", "sequence", "jaccard", "levenshtein", "content", "content2", "f1"]
    if not args.output_csv.exists():
        pd.DataFrame(columns=headers).to_csv(args.output_csv, index=False)

    all_rows = []

    for mmd_file in tqdm(list(args.mmd_dir.rglob("*.mmd")), desc="Evaluating MMDs"):
        try:
            stem = mmd_file.stem
            if "_" not in stem:
                continue
            celex_id, lang = stem.rsplit("_", 1)

            html_matches = list(args.html_dir.rglob(f"{celex_id}_{lang}.html"))
            if not html_matches:
                print(f"No HTML found for {celex_id}_{lang}")
                continue

            html_file = html_matches[0]
            html_text = html_file.read_text(encoding="utf-8")
            mmd_text = mmd_file.read_text(encoding="utf-8")

            scores = evaluator.evaluate(html_text, mmd_text)

            row = {
                "Filepath": str(mmd_file.relative_to(args.mmd_dir)),
                "CELEX ID": celex_id,
                "Language": lang,
                **{k: round(scores[k], 4) for k in scores}
            }
            all_rows.append(row)
            pd.DataFrame([row]).to_csv(args.output_csv, mode='a', header=False, index=False)

        except Exception as e:
            print(f"Error processing {mmd_file.name}: {e}")

    df = pd.DataFrame(all_rows)
    print(f"Saved evaluation results to {args.output_csv}")
    if not df.empty:
        for metric in ["sequence", "jaccard", "levenshtein", "content", "content2", "f1"]:
            print(f"Average {metric} score:", df[metric].mean().round(4))

if __name__ == "__main__":
    main()



