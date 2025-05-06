from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List
from collections import Counter
import pandas as pd
from bs4 import BeautifulSoup
import re
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

    def clean_html(self, html_content: str) -> str:
        soup = BeautifulSoup(html_content, "html.parser")
        full_text = soup.get_text("\n", strip=True)

        lines = full_text.splitlines()
        normal_lines, table_lines = [], []

        for line in lines:
            (table_lines if "|" in line else normal_lines).append(line)

        table_flat = []
        for line in table_lines:
            cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
            table_flat.append(" ".join(cells))

        combined = " ".join(normal_lines + table_flat)
        return re.sub(r"\s+", " ", combined).strip().lower()

    def _content2_similarity(self, html1: str, html2: str) -> float:
        try:
            t1 = self.clean_html(html1)
            t2 = self.clean_html(html2)
            w1 = Counter(t1.split())
            w2 = Counter(t2.split())
            common = set(w1) & set(w2)
            dot = sum(w1[w] * w2[w] for w in common)
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
    merged_html_dir = Path('/ltstorage/home/4baba/EUR_lex/merged_olmocr_pymupdf')
    official_html_dir = Path('/ltstorage/home/4baba/EUR_lex/htmls_2024')
    output_csv = Path('merged2.csv')

    evaluator = Evaluator()
    rows = []

    for merged_file in tqdm(list(merged_html_dir.rglob("*.html")), desc="Evaluating merged HTMLs"):
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
    print(f"✅ Evaluation complete. Saved to {output_csv}")

    if not df.empty:
        print("Average Scores:")
        print(df.describe().round(4))
    else:
        print("⚠️ No files evaluated. Check your input paths.")

if __name__ == "__main__":
    main()
