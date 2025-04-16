import os
import glob
import json
import base64
import gc
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from math import ceil

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import fitz  # PyMuPDF

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts.anchor import get_anchor_text

# === Prompt builder ===
def build_finetuning_prompt_with_markdown_tables(anchor_text: str):
    return (
        "You are an intelligent document parser.\n\n"
        "Return the content of this page as plain text.\n"
        "If the page contains a table or list of values, format it using markdown (e.g., with | for columns and --- for headers).\n"
        "Otherwise, use regular text with line breaks.\n"
        "Only return content that is clearly visible in the image. Do not hallucinate.\n\n"
        + anchor_text
    )

# === Process a single PDF ===
def process_pdf(pdf_path, model, processor, device):
    total_pages = len(fitz.open(pdf_path))
    document_output = {
        "source_pdf": pdf_path,
        "pages": []
    }

    for page_number in range(1, total_pages + 1):
        try:
            image_base64 = render_pdf_to_base64png(pdf_path, page_number, target_longest_image_dim=1024)
            anchor_text = get_anchor_text(pdf_path, page_number, pdf_engine="pdfreport", target_length=4000)
            prompt = build_finetuning_prompt_with_markdown_tables(anchor_text)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

            inputs = processor(
                text=[text],
                images=[main_image],
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for (key, value) in inputs.items()}

            output = model.generate(
                **inputs,
                temperature=0.8,
                max_new_tokens=1024,
                num_return_sequences=1,
                do_sample=True,
            )

            prompt_length = inputs["input_ids"].shape[1]
            new_tokens = output[:, prompt_length:]
            raw_output = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

            try:
                parsed_once = json.loads(raw_output)
                if isinstance(parsed_once, str):
                    parsed_output = json.loads(parsed_once)
                else:
                    parsed_output = parsed_once
                parsed_output["page_number"] = page_number
                document_output["pages"].append(parsed_output)
            except Exception:
                document_output["pages"].append({
                    "page_number": page_number,
                    "natural_text": raw_output
                })

        except Exception as e:
            document_output["pages"].append({
                "page_number": page_number,
                "error": str(e)
            })

    return document_output

# === Define input and output folders ===
category_paths = {
    "category10": "/home/4baba/EUR_lex/pdfs_2024/category10/*/*.pdf",
    "category19": "/home/4baba/EUR_lex/pdfs_2024/category19/*/*.pdf",
}
output_folder_base = "/home/4baba/EUR_lex/converted_json"

# === Collect all PDF jobs ===
pdf_jobs = []
for category, pattern in category_paths.items():
    pdf_files = sorted(glob.glob(pattern))
    
    for pdf_file in pdf_files:
        # Get path relative to the category root
        relative_path = os.path.relpath(pdf_file, f"/home/4baba/EUR_lex/pdfs_2024/{category}")

        # Replace .pdf with .json, but keep directory structure
        json_relative_path = os.path.splitext(relative_path)[0] + ".json"

        # Create full output path
        output_json_path = os.path.join(output_folder_base, category, json_relative_path)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

        if not os.path.exists(output_json_path):
            pdf_jobs.append((pdf_file, output_json_path))

print(f"Found {len(pdf_jobs)} PDFs to process.")

# === Load model once ===
print(" Loading model and processor...")
torch.cuda.set_device(1)  # Use GPU 1 (A100)
device = torch.device("cuda")


model = Qwen2VLForConditionalGeneration.from_pretrained(
    "allenai/olmOCR-7B-0225-preview",
    torch_dtype=torch.bfloat16,
).to(device).eval()


processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
print(" Model loaded.")

# === Batched processing ===
batch_size = 30
total_batches = ceil(len(pdf_jobs) / batch_size)

for batch_index in range(total_batches):
    batch_jobs = pdf_jobs[batch_index * batch_size:(batch_index + 1) * batch_size]
    print(f"\n Starting batch {batch_index + 1} of {total_batches}")

    for pdf_file, output_path in tqdm(batch_jobs, desc=f"📚 Batch {batch_index + 1}"):
        try:
            output_data = process_pdf(pdf_file, model, processor, device)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f" Failed to process {pdf_file}: {e}")

    # === Clean up memory between batches ===
    torch.cuda.empty_cache()
    gc.collect()

print("\n All PDFs processed and saved.")
