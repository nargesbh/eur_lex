import os
import json
from pathlib import Path

# Input and output directories
source_dir = Path("/ltstorage/home/4baba/EUR_lex/all_jsonl_files")
#/ltstorage/home/4baba/EUR_lex/eur-lex-sum/Scraping/localworkspace/results

output_base = Path("/ltstorage/shares/datasets/eu/category15/json_all")

def get_output_path(source_file):
    try:
        parts = Path(source_file).parts
        idx = parts.index("pdfs_category15")
        relative_path = Path(*parts[idx + 1:])  # Skip "pdfs_category15"
        return output_base / relative_path.with_suffix(".jsonl")
    except ValueError:
        print(f" Could not find 'pdfs_category15' in path: {source_file}")
    except Exception as e:
        print(f" Error processing path: {source_file} — {e}")
    return None

# Process all .jsonl files
for file in source_dir.glob("*.jsonl"):
    print(f" Reading: {file}")
    with open(file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                source_file = data.get("metadata", {}).get("Source-File", None)
                if not source_file:
                    print(f"No Source-File found at line {line_num} in {file.name}")
                    continue

                output_path = get_output_path(source_file)
                if output_path is None:
                    print(f"⏭ Skipped line {line_num} in {file.name}")
                    continue

                if output_path.exists():
                    print(f"Already exists: {output_path} — skipping")
                    continue

                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as out_f:
                    json.dump(data, out_f, ensure_ascii=False)
                print(f"Saved: {output_path}")

            except json.JSONDecodeError as e:
                print(f"JSON decode error at line {line_num} in {file.name}: {e}")
            except Exception as e:
                print(f"Failed at line {line_num} in {file.name}: {e}")