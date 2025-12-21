import os
import json
from pathlib import Path

# Input/output directories
source_dir = Path("")
output_base = Path("")

def get_output_path(source_file):
    try:
        parts = Path(source_file).parts
        # Extract the last 3 parts: year, CELEX dir, filename
        relative_parts = parts[-3:]
        return output_base / Path(*relative_parts).with_suffix(".jsonl")
    except Exception as e:
        print(f"Error processing path: {source_file} — {e}")
        return None

# Process all .jsonl files
for file in source_dir.glob("*.jsonl"):
    print(f"Reading: {file}")
    with open(file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                source_file = data.get("metadata", {}).get("Source-File")
                if not source_file:
                    print(f"No Source-File found at line {line_num} in {file.name}")
                    continue

                output_path = get_output_path(source_file)
                if output_path is None:
                    print(f"Skipped line {line_num} in {file.name}")
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