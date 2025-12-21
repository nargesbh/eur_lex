#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from tqdm import tqdm

INPUT_ROOT = Path("")
OUTPUT_ROOT = Path("")
METADATA_CSV = Path("metadata_results.csv")
CHECK_CSV = Path("")


def load_metadata_map(csv_path: Path) -> dict:
    """Load metadata mapping: filepath -> metadata string. CSV header: metadata,filepath"""
    mapping = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fp = row.get("filepath", "")
            meta = row.get("metadata", "")
            if not fp:
                continue
            mapping[str(Path(fp))] = meta
    return mapping


def read_olmocr_jsonl(path: Path) -> dict | None:
    """
    Try to read an OLMOCR JSON that happens to be stored as .jsonl.
    - If the whole file is a single JSON object, parse it.
    - Else, try the first non-empty line as JSON.
    """
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            return json.loads(s)
        except Exception:
            break
    return None


def merge_natural_text(data: dict) -> str:
    """Concatenate natural_text across pages in page_number order."""
    pages = data.get("pages", [])
    pages_sorted = sorted(pages, key=lambda p: p.get("page_number", 1))
    return "\n\n".join(p.get("natural_text", "") for p in pages_sorted if p.get("natural_text"))


def normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def remove_metadata_prefix(merged_text: str, metadata: str) -> tuple[str, bool]:
    """
    Remove the metadata string if it appears exactly at the very beginning of merged_text
    (after normalizing newlines). Returns (cleaned_text, removed_flag).
    """
    mt = normalize_newlines(merged_text)
    if not metadata:
        return mt, False

    md = normalize_newlines(metadata)

    if mt.startswith(md):
        return mt[len(md):].lstrip(), True

    md_stripped = md.strip()
    if md_stripped and mt.startswith(md_stripped):
        return mt[len(md_stripped):].lstrip(), True

    return mt, False


def process_one(jsonl_path: Path, meta_map: dict) -> bool:
    """
    Convert one JSONL to TXT (preserving dir structure), removing metadata prefix if present.
    Returns True if metadata was removed, False otherwise.
    """
    data = read_olmocr_jsonl(jsonl_path)
    if data is None:
        return False  # unreadable; also treated as "no removal"

    merged = merge_natural_text(data)
    meta = meta_map.get(str(jsonl_path), "")
    cleaned, removed = remove_metadata_prefix(merged, meta)

    rel = jsonl_path.relative_to(INPUT_ROOT)
    out_path = OUTPUT_ROOT / rel
    out_path = out_path.with_suffix(".txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(cleaned, encoding="utf-8")

    return removed


def main():
    meta_map = load_metadata_map(METADATA_CSV)
    jsonl_files = list(INPUT_ROOT.rglob("*.jsonl"))

    # ensure check csv has a header
    if not CHECK_CSV.exists():
        with CHECK_CSV.open("w", encoding="utf-8", newline="") as cfw:
            writer = csv.writer(cfw)
            writer.writerow(["filepath"])

    cf = CHECK_CSV.open("a", encoding="utf-8", newline="")
    writer = csv.writer(cf)
    try:
        for p in tqdm(jsonl_files, desc="Converting JSONL to TXT", unit="file"):
            try:
                removed = process_one(p, meta_map)
                if not removed:
                    writer.writerow([str(p)])
                    cf.flush()  # write on the fly
            except Exception:
                # silent skip per your ask (no logging)
                writer.writerow([str(p)])  # still mark for checking
                cf.flush()
                continue
    finally:
        cf.close()


if __name__ == "__main__":
    main()
