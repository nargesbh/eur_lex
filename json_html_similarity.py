import os
import csv
import json
import re
from bs4 import BeautifulSoup
from collections import Counter
from tqdm import tqdm


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(True):
        for attribute in ["style", "class", "id", "width", "height"]:
            tag.attrs.pop(attribute, None)
    for element in soup.find_all(string=True):
        clean_text = re.sub(r"\$\s+(\d+)", r"$\1", element.strip())
        clean_text = re.sub(r"(\.\s*)+", " ", clean_text)
        element.replace_with(clean_text)
    return " ".join(soup.stripped_strings).lower()

def content2_similarity(html1: str, html2: str) -> float:
    try:
        t1 = clean_html(html1)
        t2 = clean_html(html2)
        w1 = Counter(t1.split())
        w2 = Counter(t2.split())
        common = set(w1) & set(w2)
        dot = sum(w1[w] * w2[w] for w in common)
        norm1 = sum(v*v for v in w1.values()) ** 0.5
        norm2 = sum(v*v for v in w2.values()) ** 0.5
        return dot / (norm1 * norm2) if norm1 and norm2 else 0.0
    except Exception:
        return 0.0

def load_html_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_json_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = []
    for page in data.get("pages", []):
        natural_text = page.get("natural_text", "")
        try:
            parsed = json.loads(natural_text)
            text = parsed.get("natural_text", "")
        except (json.JSONDecodeError, TypeError):
            text = natural_text
        texts.append(text)
    return "\n".join(texts)


html_dirs = [
    "/ltstorage/home/4baba/EUR_lex/htmls_2024/category10",
    "/ltstorage/home/4baba/EUR_lex/htmls_2024/category19"
]
json_dirs = [
    "/ltstorage/home/4baba/EUR_lex/converted_json/category10",
    "/ltstorage/home/4baba/EUR_lex/converted_json/category19"
]
output_csv = "html_json_content2.csv"

# Load Existing CSV to Skip Duplicates 
existing_entries = set()
if os.path.exists(output_csv):
    with open(output_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            existing_entries.add(row["Filepath"])

with open(output_csv, mode='a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    if not existing_entries:
        writer.writerow(["Filepath", "CELEX ID", "Language", "content2"])

    for html_base, json_base in zip(html_dirs, json_dirs):
        for root, _, files in os.walk(html_base):
            for file in tqdm(files, desc=f"Processing {html_base}"):
                if not file.endswith(".html"):
                    continue

                html_path = os.path.join(root, file)
                relative_path = os.path.relpath(html_path, html_base)
                celex_id = file.split("_")[0]
                language = file.split("_")[1].split(".")[0]
                if relative_path in existing_entries:
                    continue

                json_path = os.path.join(
                    json_base, os.path.relpath(root, html_base), file.replace(".html", ".json")
                )

                if not os.path.exists(json_path):
                    continue

                try:
                    html_text = load_html_file(html_path)
                    json_text = load_json_text(json_path)
                    sim = content2_similarity(html_text, json_text)
                    writer.writerow([relative_path, celex_id, language, f"{sim:.4f}"])
                    csvfile.flush()
                except Exception:
                    continue
