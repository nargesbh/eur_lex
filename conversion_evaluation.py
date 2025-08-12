import os
import re
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List
from collections import Counter

import difflib
import Levenshtein
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from tqdm import tqdm

@dataclass
class ExtractPDFResult:
    tool_name: str
    file_name: str
    language: str
    celex_id: str
    ground_truth_html: str
    extracted_html: str
    metrics: Dict[str, float] = field(default_factory=dict)

class Evaluator:
    def __init__(self) -> None:
        self.results: Dict[str, List[ExtractPDFResult]] = {}

    def clean_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(True):
            for attribute in ["style", "class", "id", "width", "height"]:
                tag.attrs.pop(attribute, None)
        for element in soup.find_all(string=True):
            clean_text = re.sub(r"\$\s+(\d+)", r"$\1", element.strip())
            clean_text = re.sub(r"(\.\s*)+", " ", clean_text)
            element.replace_with(clean_text)
        return " ".join(soup.stripped_strings).lower()

    def _sequence_similarity(self, text1: str, text2: str) -> float:
        return difflib.SequenceMatcher(None, text1, text2).ratio()

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        tokens1 = set(re.findall(r"<[^>]+>|\w+", text1.lower()))
        tokens2 = set(re.findall(r"<[^>]+>|\w+", text2.lower()))
        return len(tokens1 & tokens2) / len(tokens1 | tokens2) if tokens1 | tokens2 else 0.0

    def _levenshtein_similarity(self, text1: str, text2: str) -> float:
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0
        return 1 - (Levenshtein.distance(text1, text2) / max_len)

    def _structure_similarity(self, html1: str, html2: str) -> float:
        try:
            s1 = [tag.name for tag in BeautifulSoup(html1, "html.parser").find_all()]
            s2 = [tag.name for tag in BeautifulSoup(html2, "html.parser").find_all()]
            return difflib.SequenceMatcher(None, s1, s2).ratio()
        except:
            return 0.0

    def _content_similarity(self, html1: str, html2: str) -> float:
        try:
            t1 = " ".join(BeautifulSoup(html1, "html.parser").stripped_strings).lower()
            t2 = " ".join(BeautifulSoup(html2, "html.parser").stripped_strings).lower()
            w1 = Counter(t1.split())
            w2 = Counter(t2.split())
            common = set(w1) & set(w2)
            dot = sum(w1[w] * w2[w] for w in common)
            norm1 = sum(v*v for v in w1.values()) ** 0.5
            norm2 = sum(v*v for v in w2.values()) ** 0.5
            return dot / (norm1 * norm2) if norm1 and norm2 else 0.0
        except:
            return 0.0

    def _content2_similarity(self, html1: str, html2: str) -> float:
        try:
            t1 = self.clean_html(html1)
            t2 = self.clean_html(html2)
            w1 = Counter(t1.split())
            w2 = Counter(t2.split())
            common = set(w1) & set(w2)
            dot = sum(w1[w] * w2[w] for w in common)
            norm1 = sum(v*v for v in w1.values()) ** 0.5
            norm2 = sum(v*v for v in w2.values()) ** 0.5
            return dot / (norm1 * norm2) if norm1 and norm2 else 0.0
        except:
            return 0.0

    def _f1_similarity(self, html1: str, html2: str) -> float:
        try:
            t1 = self.clean_html(html1)
            t2 = self.clean_html(html2)
            vectorizer = CountVectorizer(binary=True)
            x = vectorizer.fit_transform([t1, t2])
            return f1_score(x.toarray()[0], x.toarray()[1], average="binary")
        except:
            return 0.0

def main():
    parser = argparse.ArgumentParser(description="Evaluate PDF HTML conversions with multiple metrics.")
    parser.add_argument("converted_dir", type=Path, help="Directory of converted HTML files")
    parser.add_argument("official_dir", type=Path, help="Directory of ground-truth HTML files")
    parser.add_argument("output_path", type=Path, help="CSV output file path")
    args = parser.parse_args()

    tool_name = args.converted_dir.name.capitalize()
    evaluator = Evaluator()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    metric_keys = ["sequence", "jaccard", "levenshtein", "structure", "content", "content2", "f1"]

    if args.output_path.exists():
        existing_df = pd.read_csv(args.output_path)
        already_done = set(str(Path(fp).with_suffix("")) for fp in existing_df["Filepath"])
    else:
        pd.DataFrame(columns=["Filepath", "CELEX ID", "Language"] + metric_keys).to_csv(args.output_path, index=False)
        already_done = set()

    converted_files = {str(f.relative_to(args.converted_dir)): f for f in args.converted_dir.rglob("*.html")}
    official_files = {str(f.relative_to(args.official_dir)): f for f in args.official_dir.rglob("*.html")}
    common_files = sorted(set(converted_files) & set(official_files))
    print(f"Matched files: {len(common_files)}")

    all_rows = []

    for filename in tqdm(common_files, desc="Evaluating files", unit="file"):
        base_path = str(Path(filename).with_suffix(""))

        if base_path in already_done:
            continue

        celex_id, lang = Path(filename).stem.rsplit("_", 1)
        jsonl_path = f"{base_path}.jsonl"

        with open(official_files[filename], "r", encoding="utf-8") as f:
            gt = f.read()
        with open(converted_files[filename], "r", encoding="utf-8") as f:
            pred = f.read()

        result = ExtractPDFResult(
            tool_name=tool_name,
            file_name=filename,
            language=lang,
            celex_id=celex_id,
            ground_truth_html=gt,
            extracted_html=pred,
        )

        for metric in metric_keys:
            func = getattr(evaluator, f"_{metric}_similarity")
            result.metrics[metric] = func(gt, pred) if pred.strip() else 0.0

        row = {
            "Filepath": jsonl_path,
            "CELEX ID": celex_id,
            "Language": lang,
            **{k: round(result.metrics[k], 4) for k in metric_keys}
        }

        pd.DataFrame([row]).to_csv(args.output_path, mode='a', header=False, index=False)
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    print(f"Evaluation results saved to: {args.output_path}")
    if not df.empty:
        for metric in metric_keys:
            print(f"Average {metric} score:", df[metric].mean().round(4))
    else:
        print("No new files were evaluated.")

if __name__ == "__main__":
    main()


# python conversion_evaluation.py /ltstorage/shares/datasets/eu/category15/html_pymupdf_category15 /ltstorage/shares/datasets/eu/category15/htmls_category15 /ltstorage/shares/datasets/eu/category15/evaluation/pymupdf_all_metrics.csv