import json
import re
from pathlib import Path
from collections import Counter
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd

def clean_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract normal text from outside tables
    normal_text = " ".join(soup.stripped_strings)

    # Extract text inside tables separately and flatten
    table_texts = []
    for table in soup.find_all("table"):
        for cell in table.find_all(["td", "th"]):
            cell_text = cell.get_text(separator=" ", strip=True)
            table_texts.append(cell_text)

    # Combine normal text and table text
    combined_text = normal_text + " " + " ".join(table_texts)

    # Clean spacing and punctuation artifacts
    combined_text = re.sub(r"\s+", " ", combined_text)
    combined_text = combined_text.lower().strip()

    return combined_text

def extract_natural_text(json_path: Path) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    for page in data.get("pages", []):
        natural_text = page.get("natural_text", "")
        if isinstance(natural_text, str):
            try:
                parsed = json.loads(natural_text)
                if isinstance(parsed, dict) and "natural_text" in parsed:
                    natural_text = parsed["natural_text"]
            except json.JSONDecodeError:
                pass
        texts.append(natural_text)

    full_text = " \n ".join(texts)

    # Split into lines
    lines = full_text.splitlines()
    normal_lines = []
    table_lines = []

    for line in lines:
        if "|" in line:
            table_lines.append(line)
        else:
            normal_lines.append(line)

    # Flatten Markdown tables by removing pipes and joining cells
    flattened_table_text = []
    for line in table_lines:
        cells = [cell.strip() for cell in line.strip('|').split('|') if cell.strip()]
        flattened_table_text.append(" ".join(cells))

    # Combine normal text and table text
    combined_text = " ".join(normal_lines) + " " + " ".join(flattened_table_text)

    # Clean extra spaces
    combined_text = re.sub(r"\s+", " ", combined_text)

    return combined_text.lower().strip()

def content2_similarity(clean_html_text: str, extracted_text: str) -> float:
    try:
        w1 = Counter(clean_html_text.split())
        w2 = Counter(extracted_text.split())
        common = set(w1) & set(w2)
        dot = sum(w1[w] * w2[w] for w in common)
        norm1 = sum(v*v for v in w1.values()) ** 0.5
        norm2 = sum(v*v for v in w2.values()) ** 0.5
        return dot / (norm1 * norm2) if norm1 and norm2 else 0.0
    except:
        return 0.0

def main():
    html_base = Path("/ltstorage/home/4baba/EUR_lex/htmls_2024")
    json_base = Path("/ltstorage/home/4baba/EUR_lex/converted_json")
    output_csv = Path("content2.csv")

    allowed_categories = {"category10", "category19"}

    if not output_csv.exists():
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["Filepath", "CELEX ID", "Language", "content2"]).to_csv(output_csv, index=False)

    all_html_files = list(html_base.rglob("*.html"))

    html_files = [
        f for f in all_html_files
        if f.relative_to(html_base).parts[0] in allowed_categories
    ]

    for html_file in tqdm(html_files, desc="Evaluating", unit="file"):
        relative_path = html_file.relative_to(html_base)
        json_file = json_base / relative_path.with_suffix(".json")

        if not json_file.exists():
            print(f" Warning: Missing JSON for {relative_path}")
            continue

        celex_lang = html_file.stem
        if "_" not in celex_lang:
            print(f"Warning: Unexpected filename format: {celex_lang}")
            continue

        celex_id, lang = celex_lang.rsplit("_", 1)

        try:
            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()

            gt_cleaned = clean_html(html_content)
            extracted_text = extract_natural_text(json_file)

            sim = content2_similarity(gt_cleaned, extracted_text)

            result_row = {
                "Filepath": str(relative_path).replace("\\", "/"),
                "CELEX ID": celex_id,
                "Language": lang,
                "content2": round(sim, 4)
            }

            pd.DataFrame([result_row]).to_csv(output_csv, mode='a', header=False, index=False)

        except Exception as e:
            print(f"Error processing {relative_path}: {e}")

    print(f" Finished! Results saved to '{output_csv}'.")

if __name__ == "__main__":
    main()



# import json
# import re
# from pathlib import Path
# from collections import Counter
# from bs4 import BeautifulSoup
# from spellchecker import SpellChecker

# def clean_html(html_content: str) -> str:
#     soup = BeautifulSoup(html_content, "html.parser")

#     normal_text = " ".join(soup.stripped_strings)
#     table_texts = []

#     for table in soup.find_all("table"):
#         for cell in table.find_all(["td", "th"]):
#             cell_text = cell.get_text(separator=" ", strip=True)
#             table_texts.append(cell_text)

#     combined_text = normal_text + " " + " ".join(table_texts)
#     combined_text = re.sub(r"\s+", " ", combined_text)
#     return combined_text.lower().strip()

# def extract_natural_text(json_path: Path) -> str:
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     texts = []
#     for page in data.get("pages", []):
#         natural_text = page.get("natural_text", "")
#         if isinstance(natural_text, str):
#             try:
#                 parsed = json.loads(natural_text)
#                 if isinstance(parsed, dict) and "natural_text" in parsed:
#                     natural_text = parsed["natural_text"]
#             except json.JSONDecodeError:
#                 pass
#         texts.append(natural_text)

#     full_text = " \n ".join(texts)
#     lines = full_text.splitlines()
#     normal_lines = []
#     table_lines = []

#     for line in lines:
#         if "|" in line:
#             table_lines.append(line)
#         else:
#             normal_lines.append(line)

#     flattened_table_text = []
#     for line in table_lines:
#         cells = [cell.strip() for cell in line.strip('|').split('|') if cell.strip()]
#         flattened_table_text.append(" ".join(cells))

#     combined_text = " ".join(normal_lines) + " " + " ".join(flattened_table_text)
#     return re.sub(r"\s+", " ", combined_text).lower().strip()

# def build_spellchecker_from_html(html_text: str) -> SpellChecker:
#     words = set(html_text.split())
#     spell = SpellChecker(language=None)  # No built-in dictionary
#     spell.word_frequency.load_words(words)
#     return spell

# def correct_text(text: str, spell: SpellChecker) -> str:
#     corrected_words = []
#     for word in text.split():
#         if word.isalpha():
#             correction = spell.correction(word)
#             corrected_words.append(correction if correction is not None else word)
#         else:
#             corrected_words.append(word)
#     return " ".join(str(w) for w in corrected_words)

# def content2_similarity(text1: str, text2: str) -> float:
#     try:
#         w1 = Counter(text1.split())
#         w2 = Counter(text2.split())
#         common = set(w1) & set(w2)
#         dot = sum(w1[w] * w2[w] for w in common)
#         norm1 = sum(v*v for v in w1.values()) ** 0.5
#         norm2 = sum(v*v for v in w2.values()) ** 0.5
#         return dot / (norm1 * norm2) if norm1 and norm2 else 0.0
#     except:
#         return 0.0

# def main():
#     html_path = Path("/ltstorage/home/4baba/EUR_lex/htmls_2024/category19/law32024D1606/32024D1606_ET.html")
#     json_path = Path("/ltstorage/home/4baba/EUR_lex/converted_json/category19/law32024D1606/32024D1606_ET.json")

#     print("Cleaning HTML...")
#     html_text = clean_html(html_path.read_text(encoding="utf-8"))

#     print("Building spellchecker...")
#     spell = build_spellchecker_from_html(html_text)

#     print("Extracting JSON text...")
#     raw_json_text = extract_natural_text(json_path)

#     print("Correcting JSON text...")
#     corrected_json_text = correct_text(raw_json_text, spell)

#     print("Calculating content2 similarity...")
#     sim_score = content2_similarity(html_text, corrected_json_text)

#     print(f"content2 similarity score (corrected): {round(sim_score, 4)}")

# if __name__ == "__main__":
#     main()

