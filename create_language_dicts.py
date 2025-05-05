import os
import re
import json
from pathlib import Path
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
from tqdm import tqdm

HTML_ROOT = Path("/ltstorage/home/4baba/EUR_lex/htmls_2024")
OUTPUT_DIR = Path("./language_dictionaries2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_html_text(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    # Extract visible text and clean it
    text = " ".join(soup.stripped_strings)
    text = re.sub(r"\s+", " ", text).lower()
    return text

def tokenize(text):
    # Tokenize and exclude numbers
    return [word for word in re.findall(r"\b\w+\b", text) if not word.isdigit()]

# Dictionary: language -> Counter of words
language_word_counts = defaultdict(Counter)

# Traverse all HTML files
all_html_files = list(HTML_ROOT.rglob("*.html"))

print(f"📂 Found {len(all_html_files)} HTML files.")

for html_file in tqdm(all_html_files, desc="Processing files"):
    filename = html_file.name  # e.g., 32024R1449_ET.html
    try:
        celex, lang = filename.replace(".html", "").rsplit("_", 1)
        lang = lang.upper()
    except ValueError:
        print(f" Skipping malformed filename: {filename}")
        continue

    try:
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        cleaned_text = clean_html_text(html_content)
        tokens = tokenize(cleaned_text)
        language_word_counts[lang].update(tokens)
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# Save each language dictionary with filtered words (those that occur >= 5 times)
for lang, counter in language_word_counts.items():
    # Filter out words that occur less than 5 times
    filtered_counter = {word: count for word, count in counter.items() if count >= 50}
    
    output_path = OUTPUT_DIR / f"{lang.upper()}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_counter, f, ensure_ascii=False, indent=2)

print("Finished creating language dictionaries.")
