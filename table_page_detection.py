import json
from pathlib import Path
import csv
from tqdm import tqdm

def find_pages_with_tables(input_root: str, output_csv: str):
    input_root = Path(input_root)
    output_csv = Path(output_csv)

    # Find all JSONL files recursively
    json_files = list(input_root.rglob("*.jsonl"))
    print(f"Found {len(json_files)} JSONL files under {input_root}")

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "pages_with_tables"])
        writer.writeheader()

        # Use tqdm to show progress while looping over files
        for json_file in tqdm(json_files, desc="Processing JSONL files", unit="file"):
            try:
                # Read the entire JSONL file (one JSON object)
                with json_file.open("r", encoding="utf-8") as jf:
                    data = json.load(jf)

                # Check pages for is_table flag
                pages = data.get("pages", [])
                table_pages = [str(p["page_number"]) for p in pages if p.get("is_table", False)]

                # Write to CSV if there are any table pages
                if table_pages:
                    writer.writerow({
                        "filepath": str(json_file),
                        "pages_with_tables": ",".join(table_pages)
                    })

            except Exception as e:
                print(f"!! Error processing {json_file}: {e}")

    print(f"CSV saved to: {output_csv}")

if __name__ == "__main__":
    find_pages_with_tables(
        input_root="",
        output_csv=""
    )
