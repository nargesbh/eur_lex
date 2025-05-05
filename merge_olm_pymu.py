# import os
# import json
# import re
# import pandas as pd
# from tqdm import tqdm

# # === Helper Functions ===

# def extract_markdown_table(natural_text):
#     pattern = r"((\|.+?\|(\s*\n))+)"
#     match = re.search(pattern, natural_text)
#     return match.group(1).strip() if match else None

# def replace_first_table_block(html_content, markdown_table):
#     blocks = re.findall(r"((?:<p .*?</p>\s*){5,})", html_content, re.DOTALL)
#     if blocks:
#         return html_content.replace(blocks[0], f"<pre><code>{markdown_table}</code></pre>\n", 1)
#     else:
#         return html_content + f"<pre><code>{markdown_table}</code></pre>\n"

# # === Base Paths ===
# json_path_1 = "/ltstorage/home/4baba/EUR_lex/converted_json"
# json_path_2 = "/ltstorage/home/4baba/EUR_lex/spell_checked_json"
# html_base_path = "/ltstorage/home/4baba/EUR_lex/converted_html_2024/pymupdf_deleted_image_tags"
# output_base_path = "/ltstorage/home/4baba/EUR_lex/merged_olmocr_pymupdf"
# csv_path = "/ltstorage/home/4baba/EUR_lex/merged_content2_comparison.csv"

# # === Load Evaluation CSV ===
# errors = []
# processed = []

# try:
#     df = pd.read_csv(csv_path)
# except Exception as e:
#     raise RuntimeError(f"❌ Failed to read CSV file: {e}")

# # === Process Rows ===
# for _, row in tqdm(df.iterrows(), total=len(df)):
#     try:
#         celex_id = row["CELEX ID"]
#         lang = row["Language"]
#         score_json = row["content2_json"]
#         score_spell = row["content2_json_spellCorrected"]

#         source_json_base = json_path_1 if score_json >= score_spell else json_path_2
#         category_folder = f"category{celex_id[5:7]}"
#         law_folder = f"law{celex_id}"
#         filename = f"{celex_id}_{lang}.json"
#         html_filename = f"{celex_id}_{lang}.html"

#         json_path = os.path.join(source_json_base, category_folder, law_folder, filename)
#         html_path = os.path.join(html_base_path, category_folder, law_folder, html_filename)
#         output_path = os.path.join(output_base_path, category_folder, law_folder, html_filename)

#         if not os.path.exists(json_path):
#             errors.append((celex_id, lang, "JSON file not found", json_path))
#             continue
#         if not os.path.exists(html_path):
#             errors.append((celex_id, lang, "HTML file not found", html_path))
#             continue

#         with open(json_path, "r") as f:
#             olmocr_data = json.load(f)

#         with open(html_path, "r") as f:
#             pymupdf_html = f.read()

#         page_splits = re.split(r"(<h2>Page (\d+)</h2>)", pymupdf_html)
#         page_map = {}
#         for i in range(1, len(page_splits), 3):
#             header = page_splits[i]
#             page_num = int(page_splits[i + 1])
#             content = page_splits[i + 2]
#             page_map[page_num] = header + content

#         markdown_tables = {}
#         for page in olmocr_data["pages"]:
#             if page["is_table"]:
#                 table = extract_markdown_table(page["natural_text"])
#                 if table:
#                     markdown_tables[page["page_number"]] = table

#         final_html = ""
#         for page_num in sorted(page_map.keys()):
#             page_html = page_map[page_num]
#             if page_num in markdown_tables:
#                 updated_html = replace_first_table_block(page_html, markdown_tables[page_num])
#                 final_html += updated_html
#             else:
#                 final_html += page_html

#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
#         with open(output_path, "w") as f:
#             f.write(final_html)

#         processed.append(output_path)

#     except Exception as e:
#         errors.append((celex_id, lang, str(e), ""))

# # Output results
# processed_df = pd.DataFrame({"Output File": processed})
# errors_df = pd.DataFrame(errors, columns=["CELEX ID", "Language", "Error", "Path"])

# # Save results locally for inspection
# processed_df.to_csv("processed_merged_files.csv", index=False)
# errors_df.to_csv("merge_errors.csv", index=False)

# print("Processing complete.")
# print(f"Processed files saved to: processed_merged_files.csv")
# print(f"Errors (if any) saved to: merge_errors.csv")





import os
import json
import re
from pathlib import Path
from tqdm import tqdm

def extract_markdown_tables(json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tables = {}
    for page in data.get("pages", []):
        if page.get("is_table", False):
            page_number = page.get("page_number")
            text = page.get("natural_text", "")
            if isinstance(text, str):
                match = re.search(r"((\|.+?\|(\s*\n))+)", text)
                if match:
                    tables[page_number] = match.group(1).strip()
    return tables

def replace_table_blocks(html_content, markdown_table):
    blocks = re.findall(r"((?:<p .*?</p>\s*){5,})", html_content, re.DOTALL)
    if blocks:
        return html_content.replace(blocks[0], f"<pre><code>{markdown_table}</code></pre>\n", 1)
    else:
        return html_content + f"<pre><code>{markdown_table}</code></pre>\n"

def merge_json_html(json_dir, html_dir, output_dir):
    json_dir = Path(json_dir)
    html_dir = Path(html_dir)
    output_dir = Path(output_dir)

    all_html_files = list(html_dir.rglob("*.html"))
    results = []

    for html_file in tqdm(all_html_files, desc="Merging files"):
        rel_path = html_file.relative_to(html_dir)
        json_file = json_dir / rel_path.with_suffix(".json")
        output_file = output_dir / rel_path

        if not json_file.exists():
            print(f"⚠️ JSON file not found for: {html_file}")
            continue

        try:
            markdown_tables = extract_markdown_tables(json_file)

            with open(html_file, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Split HTML into pages
            page_splits = re.split(r"(<h2>Page (\d+)</h2>)", html_content)
            page_map = {}
            for i in range(1, len(page_splits), 3):
                header = page_splits[i]
                page_num = int(page_splits[i + 1])
                content = page_splits[i + 2]
                page_map[page_num] = header + content

            final_html = ""
            for page_num in sorted(page_map):
                page_html = page_map[page_num]
                if page_num in markdown_tables:
                    updated = replace_table_blocks(page_html, markdown_tables[page_num])
                    final_html += updated
                else:
                    final_html += page_html

            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(final_html)

            results.append(str(output_file))

        except Exception as e:
            print(f"Error processing {html_file}: {e}")

    return results

merge_json_html(
    json_dir="/ltstorage/home/4baba/EUR_lex/converted_json",
    html_dir="/ltstorage/home/4baba/EUR_lex/converted_html_2024/pymupdf_deleted_image_tags",
    output_dir="/ltstorage/home/4baba/EUR_lex/merged_olmocr_pymupdf"
)