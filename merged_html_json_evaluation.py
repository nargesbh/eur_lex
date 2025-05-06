import argparse
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from collections import Counter
import re
from dataclasses import dataclass, field
from typing import Dict, List
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

    def clean_html_with_markdown(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        # Include markdown text if present in pre/code blocks
        markdown_texts = []
        for tag in soup.find_all(["pre", "code", "div"]):
            if tag.get("class") == ["markdown"] or "markdown" in str(tag.get("class")):
                markdown_texts.append(tag.get_text(" ", strip=True))
        for tag in soup.find_all(True):
            for attr in ["style", "class", "id", "width", "height"]:
                tag.attrs.pop(attr, None)
        cleaned = " ".join(soup.stripped_strings)
        all_text = cleaned + " " + " ".join(markdown_texts)
        return re.sub(r"\s+", " ", all_text).strip().lower()

    def _content2_similarity(self, html1: str, html2: str) -> float:
        try:
            t1 = self.clean_html_with_markdown(html1)
            t2 = self.clean_html_with_markdown(html2)
            w1 = Counter(t1.split())
            w2 = Counter(t2.split())
            dot = sum(w1[w] * w2[w] for w in set(w1) & set(w2))
            norm1 = sum(v * v for v in w1.values()) ** 0.5
            norm2 = sum(v * v for v in w2.values()) ** 0.5
            return dot / (norm1 * norm2) if norm1 and norm2 else 0.0
        except:
            return 0.0

    def evaluate(self, result: ExtractPDFResult) -> ExtractPDFResult:
        score = self._content2_similarity(result.ground_truth_html, result.extracted_html)
        result.metrics["content2"] = round(score, 4)
        self.results.setdefault(result.tool_name, []).append(result)
        return result

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("merged_html_dir", type=Path, help="Directory of merged HTML files")
    # parser.add_argument("official_html_dir", type=Path, help="Directory of ground truth HTML files")
    # parser.add_argument("output_csv", type=Path, help="Path to save content2 scores")
    
    merged_html_dir = Path('/ltstorage/home/4baba/EUR_lex/merged_olmocr_pymupdf')
    official_html_dir = Path('/ltstorage/home/4baba/EUR_lex/htmls_2024')
    output_csv = Path('merged_html_json_evaluation.csv')

    # args = parser.parse_args()
    # merged_html_dir = args.merged_html_dir
    # official_html_dir = args.official_html_dir
    # output_csv = args.output_csv
    # output_csv.parent.mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator()
    rows = []

    for merged_file in tqdm(list(merged_html_dir.rglob("*.html")), desc="Evaluating merged files"):
        rel_path = merged_file.relative_to(merged_html_dir)
        gt_file = official_html_dir / rel_path

        if not gt_file.exists():
            print(f"⚠️ Missing ground truth for: {rel_path}")
            continue

        filename = merged_file.name
        celex_id, lang = filename.replace(".html", "").rsplit("_", 1)

        with open(gt_file, encoding="utf-8") as f1, open(merged_file, encoding="utf-8") as f2:
            gt_html = f1.read()
            merged_html = f2.read()

        result = ExtractPDFResult(
            tool_name="Merged",
            file_name=filename,
            celex_id=celex_id,
            language=lang,
            ground_truth_html=gt_html,
            extracted_html=merged_html
        )

        result = evaluator.evaluate(result)

        rows.append({
            "Filename": filename,
            "CELEX ID": celex_id,
            "Language": lang,
            "content2": result.metrics["content2"]
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Evaluation complete. Saved to {output_csv}")
    print(df.describe().round(4))

if __name__ == "__main__":
    main()
