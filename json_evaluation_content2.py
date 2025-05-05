import json
import re
from pathlib import Path
from collections import Counter
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

def clean_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    normal_text = " ".join(soup.stripped_strings)
    table_texts = []

    for table in soup.find_all("table"):
        for cell in table.find_all(["td", "th"]):
            table_texts.append(cell.get_text(separator=" ", strip=True))

    combined = normal_text + " " + " ".join(table_texts)
    return re.sub(r"\s+", " ", combined).strip().lower()

def extract_json_text(json_file: Path) -> str:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    # Handles both {"pages": [...]} and {"natural_text": "..."} formats
    if isinstance(data, dict) and "pages" in data:
        for page in data.get("pages", []):
            natural = page.get("natural_text", "")
            natural = try_extract_nested_natural_text(natural)
            texts.append(natural)
    elif "natural_text" in data:
        texts.append(data["natural_text"])

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
    results = []
    json_files = list(json_dir.rglob("*.json"))

    for json_file in tqdm(json_files, desc="Evaluating content2", unit="file"):
        rel_path = json_file.relative_to(json_dir)
        html_file = html_dir / rel_path.with_suffix(".html")

        if not html_file.exists():
            print(f"⚠️ Missing HTML for: {rel_path}")
            continue

        celex_lang = json_file.stem
        if "_" not in celex_lang:
            print(f"⚠️ Unexpected format: {celex_lang}")
            continue

        celex_id, lang = celex_lang.rsplit("_", 1)

        try:
            html_text = clean_html(html_file.read_text(encoding="utf-8"))
            json_text = extract_json_text(json_file)
            sim = content2_similarity(html_text, json_text)

            results.append({
                "Filepath": str(rel_path).replace("\\", "/"),
                "CELEX ID": celex_id,
                "Language": lang,
                "content2": round(sim, 4)
            })
        except Exception as e:
            print(f"❌ Error processing {rel_path}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Evaluation complete. Saved to: {output_csv}")

# === Run the script with your paths ===
if __name__ == "__main__":
    evaluate_content2(
        json_dir=Path("/ltstorage/home/4baba/EUR_lex/spell_checked_json"),
        html_dir=Path("/ltstorage/home/4baba/EUR_lex/htmls_2024"),
        output_csv=Path("content2_scores.csv")
    )
