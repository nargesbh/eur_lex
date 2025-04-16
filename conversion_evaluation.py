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

    def evaluate_tool(self, result: ExtractPDFResult, weights: Dict[str, float]) -> ExtractPDFResult:
        missed = not bool(result.extracted_html)
        methods = {
            "sequence": self._sequence_similarity,
            "jaccard": self._jaccard_similarity,
            "levenshtein": self._levenshtein_similarity,
            "structure": self._structure_similarity,
            "content": self._content_similarity,
            "content2": self._content2_similarity,
            "f1": self._f1_similarity,
        }

        result.metrics = {
            key: (methods[key](result.ground_truth_html, result.extracted_html) if not missed else 0.0)
            for key in weights
        }
        result.metrics["missed"] = 1.0 if missed else 0.0
        result.metrics["similarity"] = sum(result.metrics[k] * weights[k] for k in weights) / len(weights)
        self.results.setdefault(result.tool_name, []).append(result)
        return result

def main():
    parser = argparse.ArgumentParser(description="Evaluate PDF HTML conversions.")
    parser.add_argument("converted_dir", type=Path, help="Base directory of converted HTML files")
    parser.add_argument("official_dir", type=Path, help="Base directory of ground-truth official HTML files")
    parser.add_argument("output_path", type=Path, help="CSV output file path")

    args = parser.parse_args()

    tool_name = args.converted_dir.name.capitalize()
    weights = {
        "sequence": 1.0,
        "jaccard": 1.0,
        "levenshtein": 1.0,
        "structure": 1.0,
        "content": 1.0,
        "content2": 1.0,
        "f1": 1.0,
    }

    categories = ["category10", "category19"]
    base_converted_dir = args.converted_dir
    # base_official_dir = Path("/home/4baba/EUR_lex/htmls_2024")
    base_official_dir = args.official_dir


    evaluator = Evaluator()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create CSV and write header if it doesn't exist
    header_written = args.output_path.exists()
    if not header_written:
        pd.DataFrame(columns=[
            "Filename", "CELEX ID", "Language", "sequence", "jaccard", "levenshtein",
            "structure", "content", "content2", "f1", "missed", "similarity"
        ]).to_csv(args.output_path, index=False)

    all_rows = []

    for category in categories:
        converted_dir = base_converted_dir / category
        official_dir = base_official_dir / category

        converted_files = {f.name: f for f in converted_dir.rglob("*.html")}
        official_files = {f.name: f for f in official_dir.rglob("*.html")}
        common_files = sorted(set(converted_files) & set(official_files))

        print(f"[{category}] Matching files: {len(common_files)}")

        for filename in tqdm(common_files, desc=f"Evaluating {category}", unit="file"):
            celex_id, lang = filename.replace(".html", "").rsplit("_", 1)
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

            result = evaluator.evaluate_tool(result, weights)

            row = {
                "Filename": filename,
                "CELEX ID": celex_id,
                "Language": lang,
                **{k: round(v, 4) for k, v in result.metrics.items()}
            }

            # Append result row immediately to CSV
            pd.DataFrame([row]).to_csv(args.output_path, mode='a', header=False, index=False)

            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    print(f"Evaluation results saved to: {args.output_path}")

    print("Average Scores:")
    print(df.drop(columns=["Filename", "CELEX ID", "Language"]).mean().round(4))

if __name__ == "__main__":
    main()
