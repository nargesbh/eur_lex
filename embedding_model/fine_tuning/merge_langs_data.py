#!/usr/bin/env python3
import os
import json
import argparse
from typing import Dict, Any, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except Exception as e:
                print(f"[WARN] Skipping line {i} in {path}: {e}")
    return data


def extract_doc_id_from_query_path(query_path: str) -> str:
    """
    Extract document ID like '32007D0205' or '21993A0216(02)' from:
      .../32007D0205_EN.jsonl
      .../32007D0205_LV.jsonl
      .../21993A0216(02)_EN.jsonl
    Strategy: take basename and split before the first underscore.
    """
    base = os.path.basename(query_path)  # e.g. "32007D0205_EN.jsonl"
    if "_" in base:
        return base.split("_")[0]
    # Fallback: strip extension
    return os.path.splitext(base)[0]


def build_pos_map(examples: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Build doc_id -> positive_path map from a list of examples.
    """
    mapping: Dict[str, str] = {}
    for ex in examples:
        qp = ex.get("query_path") or ""
        pos = ex.get("positive_path") or ""
        if not qp or not pos:
            continue
        doc_id = extract_doc_id_from_query_path(qp)
        if not doc_id:
            continue
        # If there are duplicates, first one wins (fine for our purpose)
        if doc_id not in mapping:
            mapping[doc_id] = pos
    return mapping


def merge_union(en_path: str, lv_path: str, out_path: str) -> None:
    print(f"Loading EN train file from: {en_path}")
    en_examples = read_jsonl(en_path)
    print(f"Loaded {len(en_examples)} EN examples")

    print(f"Loading LV train file from: {lv_path}")
    lv_examples = read_jsonl(lv_path)
    print(f"Loaded {len(lv_examples)} LV examples")

    # doc_id -> positive TXT path
    en_pos_map = build_pos_map(en_examples)
    lv_pos_map = build_pos_map(lv_examples)

    print(f"Indexed {len(en_pos_map)} EN doc_ids and {len(lv_pos_map)} LV doc_ids")

    num_written = 0

    with open(out_path, "w", encoding="utf-8") as out_f:

        # ----- 1) Write all EN examples -----
        for ex in en_examples:
            qp = ex.get("query_path") or ""
            pos_en = ex.get("positive_path") or ""
            if not qp or not pos_en:
                continue

            doc_id = extract_doc_id_from_query_path(qp)
            pos_lv = lv_pos_map.get(doc_id)

            # Build positive_path list: always include EN, add LV if exists
            if pos_lv and pos_lv != pos_en:
                pos_list = [pos_en, pos_lv]
            else:
                pos_list = [pos_en]

            merged = {
                "query_text": ex.get("query_text", ""),
                "query_path": qp,
                "positive_path": pos_list,
            }
            out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
            num_written += 1

        # ----- 2) Write all LV examples -----
        for ex in lv_examples:
            qp = ex.get("query_path") or ""
            pos_lv = ex.get("positive_path") or ""
            if not qp or not pos_lv:
                continue

            doc_id = extract_doc_id_from_query_path(qp)
            pos_en = en_pos_map.get(doc_id)

            # Build positive_path list: always include LV, add EN if exists
            if pos_en and pos_en != pos_lv:
                pos_list = [pos_lv, pos_en]
            else:
                pos_list = [pos_lv]

            merged = {
                "query_text": ex.get("query_text", ""),
                "query_path": qp,
                "positive_path": pos_list,
            }
            out_f.write(json.dumps(merged, ensure_ascii=False) + "\n")
            num_written += 1

    print(f"Wrote {num_written} merged examples to {out_path}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge EN + LV train.jsonl files into one file with UNION of rows.\n"
            "Each output line keeps its own query_text/query_path and has "
            "positive_path as a list containing EN and/or LV txt paths."
        )
    )
    parser.add_argument("--en-train", required=True, help="Path to English train.jsonl")
    parser.add_argument("--lv-train", required=True, help="Path to Latvian train.jsonl")
    parser.add_argument("--out", required=True, help="Output merged train.jsonl")

    args = parser.parse_args()
    merge_union(args.en_train, args.lv_train, args.out)


if __name__ == "__main__":
    main()


# python merge_langs_data.py \
# --en-train /ltstorage/home/4baba/EUR_lex/embedding_model/fine_tuning/tunning_data/EN/EN_Qwen3-Embedding-4B/train.jsonl \
# --lv-train /ltstorage/home/4baba/EUR_lex/embedding_model/fine_tuning/tunning_data/LV/Qwen3_4B/train.jsonl \
# --out /ltstorage/home/4baba/EUR_lex/embedding_model/fine_tuning/tunning_data/LV-EN/Qwen3_4B/train.jsonl


# python merge_langs_data.py \
# --en-train /ltstorage/home/4baba/EUR_lex/embedding_model/fine_tuning/tunning_data/EN/EN_Qwen3-Embedding-4B/test.jsonl \
# --lv-train /ltstorage/home/4baba/EUR_lex/embedding_model/fine_tuning/tunning_data/LV/Qwen3_4B/test.jsonl \
# --out /ltstorage/home/4baba/EUR_lex/embedding_model/fine_tuning/tunning_data/LV-EN/Qwen3_4B/test.jsonl

# python merge_langs_data.py \
# --en-train /ltstorage/home/4baba/EUR_lex/embedding_model/fine_tuning/tunning_data/EN/EN_Qwen3-Embedding-4B/val.jsonl \
# --lv-train /ltstorage/home/4baba/EUR_lex/embedding_model/fine_tuning/tunning_data/LV/Qwen3_4B/val.jsonl \
# --out /ltstorage/home/4baba/EUR_lex/embedding_model/fine_tuning/tunning_data/LV-EN/Qwen3_4B/val.jsonl