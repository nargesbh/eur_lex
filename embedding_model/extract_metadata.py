# import json
# import re
# from pathlib import Path

# # ≥3 ALL-CAPS words (2+ letters each) on a line
# SENTINEL_3PLUS_ALLCAPS = re.compile(r'^(?:.*?\b[A-Z]{2,}\b.*?){2,}$', re.M)

# def load_page1_natural_text(json_path: Path) -> str:
#     """Assumes OLMOCR JSON with pages and natural_text present."""
#     data = json.loads(json_path.read_text(encoding="utf-8"))
#     for p in data["pages"]:
#         if p.get("page_number") == 1:
#             return p.get("natural_text", "")
#     return ""

# def _char_index_after_n_words(text: str, n_words: int) -> int:
#     """Return the character index right after the Nth word (words = \S+)."""
#     count = 0
#     for m in re.finditer(r'\S+', text):
#         count += 1
#         if count == n_words:
#             return m.end()
#     return 0  # fewer than n_words → start at 0

# def extract_metadata_or_fallback(
#     page_text: str,
#     n_search: int = 1000,   # search window
#     min_chars: int = 150,   # too short → fallback
#     fallback_n: int = 200   # fallback length
# ) -> str:
#     # Limit to search window
#     head = page_text[:n_search]

#     # Compute where to start searching: after the first 4 words of the WHOLE page text,
#     # but clamp inside the head window
#     start_after_4_words = _char_index_after_n_words(page_text, 4)
#     start_in_head = min(max(start_after_4_words, 0), len(head))

#     # Search for the sentinel only in head[start_in_head:]
#     m = SENTINEL_3PLUS_ALLCAPS.search(head, pos=start_in_head)

#     if m:
#         candidate = head[:m.start()].strip()
#         if len(candidate) < min_chars:
#             return page_text[:fallback_n].strip()
#         return candidate
#     else:
#         return page_text[:fallback_n].strip()

# if __name__ == "__main__":
#     import argparse
#     ap = argparse.ArgumentParser(description="Extract metadata-like head from page 1 of an OLMOCR JSON.")
#     ap.add_argument("json_path", type=Path, help="Path to the JSON file")
#     ap.add_argument("--n-search", type=int, default=1500)
#     ap.add_argument("--min-chars", type=int, default=200)
#     ap.add_argument("--fallback-n", type=int, default=400)
#     args = ap.parse_args()

#     text = load_page1_natural_text(args.json_path)
#     result = extract_metadata_or_fallback(
#         text, n_search=args.n_search, min_chars=args.min_chars, fallback_n=args.fallback_n
#     )
#     print(result)


import json
import re
from pathlib import Path
import csv
from tqdm import tqdm

# ≥3 ALL-CAPS words (2+ letters each) on a line
SENTINEL_3PLUS_ALLCAPS = re.compile(r'^(?:.*?\b[A-Z]{2,}\b.*?){2,}$', re.M)

def load_page1_natural_text(json_path: Path) -> str:
    """Assumes OLMOCR JSON with pages and natural_text present."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            # JSONL → first line contains the metadata we want
            first_line = f.readline()
            data = json.loads(first_line)
        for p in data.get("pages", []):
            if p.get("page_number") == 1:
                return p.get("natural_text", "")
    except Exception as e:
        return ""
    return ""

def _char_index_after_n_words(text: str, n_words: int) -> int:
    """Return the character index right after the Nth word (words = \S+)."""
    count = 0
    for m in re.finditer(r'\S+', text):
        count += 1
        if count == n_words:
            return m.end()
    return 0

def extract_metadata_or_fallback(
    page_text: str,
    n_search: int = 1000,
    min_chars: int = 150,
    fallback_n: int = 200
) -> str:
    head = page_text[:n_search]
    start_after_4_words = _char_index_after_n_words(page_text, 4)
    start_in_head = min(max(start_after_4_words, 0), len(head))
    m = SENTINEL_3PLUS_ALLCAPS.search(head, pos=start_in_head)
    if m:
        candidate = head[:m.start()].strip()
        if len(candidate) < min_chars:
            return page_text[:fallback_n].strip()
        return candidate
    else:
        return page_text[:fallback_n].strip()

if __name__ == "__main__":
    base_dir = Path("/ltstorage/shares/datasets/eu/category15/json_category15")
    output_csv = Path("metadata_results.csv")

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["metadata", "filepath"])  # header

        jsonl_files = list(base_dir.rglob("*.jsonl"))

        for file_path in tqdm(jsonl_files, desc="Processing JSONL files"):
            text = load_page1_natural_text(file_path)
            metadata = extract_metadata_or_fallback(text, n_search=1500, min_chars=200, fallback_n=400)
            writer.writerow([metadata, str(file_path)])
            csvfile.flush()  # write to disk immediately
