import json
import re
from pathlib import Path
from collections import Counter
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import argparse

def clean_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    normal_text = " ".join(soup.stripped_strings)
    table_texts = []

    for table in soup.find_all("table"):
        for cell in table.find_all(["td", "th"]):
            table_texts.append(cell.get_text(separator=" ", strip=True))

    combined = normal_text + " " + " ".join(table_texts)
    return re.sub(r"\s+", " ", combined).strip().lower()

def try_extract_nested_natural_text(natural):
    """Helper to parse nested JSON inside natural_text strings"""
    if isinstance(natural, str):
        try:
            parsed = json.loads(natural)
            if isinstance(parsed, dict) and "natural_text" in parsed:
                return parsed["natural_text"]
        except json.JSONDecodeError:
            pass
    return natural

def extract_json_text(json_file: Path) -> tuple[str, int]:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    missing_pages = 0

    if isinstance(data, dict) and "pages" in data:
        for page in data.get("pages", []):
            natural = page.get("natural_text", "")
            natural = try_extract_nested_natural_text(natural)
            if isinstance(natural, str):
                texts.append(natural)
            else:
                missing_pages += 1
    elif "natural_text" in data:
        if isinstance(data["natural_text"], str):
            texts.append(data["natural_text"])
        else:
            missing_pages += 1

    full_text = "\n".join(texts)
    lines = full_text.splitlines()

    normal_lines, table_lines = [], []
    for line in lines:
        (table_lines if "|" in line else normal_lines).append(line)

    table_flat = []
    for line in table_lines:
        cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
        table_flat.append(" ".join(cells))

    combined = " ".join(normal_lines + table_flat)
    return re.sub(r"\s+", " ", combined).strip().lower(), missing_pages

def content2_similarity(text1: str, text2: str) -> float:
    try:
        w1 = Counter(text1.split())
        w2 = Counter(text2.split())
        common = set(w1) & set(w2)
        dot = sum(w1[w] * w2[w] for w in common)
        norm1 = sum(v * v for v in w1.values()) ** 0.5
        norm2 = sum(v * v for v in w2.values()) ** 0.5
        return dot / (norm1 * norm2) if norm1 and norm2 else 0.0
    except Exception as e:
        print(f"Similarity error: {e}")
        return 0.0

def evaluate_content2(json_dir: Path, html_dir: Path, output_csv: Path):
    json_files = list(json_dir.rglob("*.jsonl"))
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        already_done = set(str(fp).rsplit('.', 1)[0] for fp in existing_df["Filepath"])
    else:
        pd.DataFrame(columns=["Filepath", "CELEX ID", "Language", "content2", "Missing_Pages"]).to_csv(output_csv, index=False)
        already_done = set()

    scores = []

    for json_file in tqdm(json_files, desc="Evaluating content2", unit="file"):
        try:
            rel_path = json_file.relative_to(json_dir)
            base_path = str(rel_path).rsplit('.', 1)[0]

            if base_path in already_done:
                continue

            html_file = html_dir / rel_path.with_suffix(".html")
            if not html_file.exists():
                print(f"Missing HTML for: {rel_path}")
                continue

            celex_lang = json_file.stem
            if "_" not in celex_lang:
                print(f"Unexpected format: {celex_lang}")
                continue

            celex_id, lang = celex_lang.rsplit("_", 1)
            html_text = clean_html(html_file.read_text(encoding="utf-8"))
            json_text, missing_pages = extract_json_text(json_file)
            sim = content2_similarity(html_text, json_text)
            scores.append(sim)

            row = {
                "Filepath": str(rel_path).replace("\\", "/"),
                "CELEX ID": celex_id,
                "Language": lang,
                "content2": round(sim, 4),
                "Missing_Pages": missing_pages
            }

            pd.DataFrame([row]).to_csv(output_csv, mode='a', header=False, index=False)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    if scores:
        avg = sum(scores) / len(scores)
        print(f"Evaluation complete. Saved to: {output_csv}")
        print(f"Average content2 score: {round(avg, 4)}")
    else:
        print("No new files were evaluated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate content2 similarity between JSON and HTML files.")
    parser.add_argument("json_dir", type=Path, help="Directory containing JSONL files")
    parser.add_argument("html_dir", type=Path, help="Directory containing HTML files")
    parser.add_argument("output_csv", type=Path, help="Output path for CSV results")
    args = parser.parse_args()

    evaluate_content2(args.json_dir, args.html_dir, args.output_csv)
