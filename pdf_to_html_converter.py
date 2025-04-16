import os
import re
import logging
from tqdm import tqdm
import fitz  # PyMuPDF
import pandas as pd
import pdfplumber
from typing import List
from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
import html

logging.basicConfig(level=logging.INFO)

# ---------------------------
# Tool: Docling
# Full document extraction using OCR and structure-aware parser
# ---------------------------
class DoclingTool:
    def __init__(self):
        self.pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            ocr_options=EasyOcrOptions()
        )
        self.pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options,
                    backend=DoclingParseV2DocumentBackend
                )
            }
        )

    def convert(self, pdf_file: str) -> str:
        try:
            result = self.doc_converter.convert(pdf_file)
            document = result.document
            html = document.export_to_html()
            if not html.strip():
                logging.warning(f"[Docling] {os.path.basename(pdf_file)} is empty")
            return html
        except Exception as e:
            logging.error(f"[Docling] Error: {e}")
            return ""


# ---------------------------
# Tool: PyMuPDF (text only)
# Extracts plain text using HTML layout from PyMuPDF
# No table detection included
# ---------------------------
class PyMuPDFTool:
    def convert(self, pdf_file: str) -> str:
        try:
            doc = fitz.open(pdf_file)
            html = "<html><body>\n"
            for i, page in enumerate(doc):
                html += f"<h2>Page {i + 1}</h2>\n"
                html += page.get_text("html") + "\n"
            html += "</body></html>"
            return html
        except Exception as e:
            logging.error(f"[PyMuPDF] Error: {e}")
            return ""


# ---------------------------
# Tool: PyMuPDF with tables
# Same as PyMuPDFTool but includes structured tables at the end of each page
# Extracts tables using PyMuPDF's find_tables and converts them with pandas
# ---------------------------
class PyMuPDFWithTablesTool:
    def convert(self, pdf_file: str) -> str:
        try:
            doc = fitz.open(pdf_file)
            html = "<html><body>\n"
            for i, page in enumerate(doc):
                html += f"<h2>Page {i + 1}</h2>\n"
                html += page.get_text("html") + "\n"
                try:
                    tables = page.find_tables(strategy="text")
                    for j, table in enumerate(tables):
                        df = pd.DataFrame(table.extract())
                        html += f"<h3>Structured Table {j + 1}</h3>\n"
                        html += df.to_html(index=False, border=1) + "\n"
                except Exception as e:
                    logging.warning(f"[PyMuPDFTables] Table error on page {i + 1}: {e}")
            html += "</body></html>"
            return html
        except Exception as e:
            logging.error(f"[PyMuPDFTables] Error: {e}")
            return ""


# ---------------------------
# Tool: Hybrid pdfplumber + fitz
# Uses pdfplumber for precise text layout extraction
# Uses PyMuPDF for reliable table detection and rendering
# Best of both: fine text + structured tables
# ---------------------------
class PdfPlumberHybridTool:
    def convert(self, pdf_file: str) -> str:
        try:
            html = "<html><body>\n"
            with pdfplumber.open(pdf_file) as plumber_pdf, fitz.open(pdf_file) as pymupdf_pdf:
                for i, (plumber_page, pymupdf_page) in enumerate(zip(plumber_pdf.pages, pymupdf_pdf), start=1):
                    html += f"<h2>Page {i}</h2>\n"
                    text = plumber_page.extract_text()
                    html += f"<div style='white-space: pre-wrap;'>{text}</div>\n" if text else "<div><i>No text found.</i></div>\n"
                    try:
                        tables = pymupdf_page.find_tables(strategy="text")
                        for j, table in enumerate(tables):
                            df = pd.DataFrame(table.extract())
                            html += f"<h3>Table {j + 1}</h3>\n" + df.to_html(index=False, border=1)
                    except Exception as e:
                        logging.warning(f"[Hybrid] Table error on page {i}: {e}")
            html += "</body></html>"
            return html
        except Exception as e:
            logging.error(f"[Hybrid] Error: {e}")
            return ""

class PyMuPDFHybridOrderedTool:
    def convert(self, pdf_file: str) -> str:
        try:
            doc = fitz.open(pdf_file)
            html_out = "<html><body>\n"

            for i, page in enumerate(doc):
                html_out += f"<h2>Page {i + 1}</h2>\n"
                elements = []

                # --- Extract Text Blocks ---
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if block["type"] == 0:  # Text block
                        y0 = block["bbox"][1]
                        paragraph = ""
                        for line in block["lines"]:
                            line_text = " ".join([span["text"] for span in line["spans"]])
                            paragraph += line_text.strip() + " "
                        if paragraph.strip():
                            paragraph_html = f"<p>{html.escape(paragraph.strip())}</p>"
                            elements.append((y0, paragraph_html))

                # --- Extract All Tables (No Filtering) ---
                try:
                    tables = page.find_tables(strategy="text")
                    for j, table in enumerate(tables):
                        y0 = table.bbox[1]
                        df = pd.DataFrame(table.extract())
                        table_html = f"<h3>Table {j + 1}</h3>\n" + df.to_html(index=False, border=1)
                        elements.append((y0, table_html))
                except Exception as e:
                    logging.warning(f"[PyMuPDFHybridOrdered] Table error on page {i + 1}: {e}")

                # --- Merge Text + Tables by Layout Order ---
                elements.sort(key=lambda x: x[0])  # Sort by vertical position
                for _, content in elements:
                    html_out += content + "\n"

            html_out += "</body></html>"
            return html_out

        except Exception as e:
            logging.error(f"[PyMuPDFHybridOrdered] Error: {e}")
            return ""


# ---------------------------
# Tool: Unstructured
# Extracts elements and structures them semantically into HTML
# Includes basic layout classification (headings, paragraphs, etc.)
# ---------------------------
class UnstructuredTool:
    def __init__(self, language="eng"):
        self.language = language

    def extract_elements(self, pdf_file: str) -> List[Element]:
        try:
            return partition_pdf(
                filename=pdf_file,
                strategy="hi_res",
                infer_table_structure=True,
                model_name="yolox",
                languages=[self.language],
            )
        except Exception as e:
            logging.error(f"[Unstructured] Error: {e}")
            return []

    def convert(self, pdf_file: str) -> str:
        elements = self.extract_elements(pdf_file)
        html_parts = []
        for el in elements:
            try:
                if hasattr(el.metadata, "text_as_html") and el.metadata.text_as_html:
                    html_parts.append(el.metadata.text_as_html)
                else:
                    tag = self._tag_for_element(el)
                    html_parts.append(f"<{tag}>{el.text.strip()}</{tag}>")
            except Exception as e:
                logging.warning(f"[Unstructured] Element error: {e}")
        return "<html><body>\n" + "\n<br/>".join(html_parts) + "\n</body></html>"

    def _tag_for_element(self, el: Element) -> str:
        cat = el.category.lower()
        return {
            "title": "h1", "header": "h2", "footer": "footer",
            "caption": "figcaption", "list": "ul", "code": "code",
            "email": "address", "address": "address",
            "page": "hr", "break": "hr"
        }.get(cat, "p")

# ---------------------------
# Language extraction for Unstructured
# ---------------------------
def extract_tesseract_lang_from_filename(filename: str) -> str:
    code_map = {
        "BG": "bul", "CS": "ces", "DA": "dan", "DE": "deu", "EL": "ell", "EN": "eng",
        "ES": "spa", "ET": "est", "FI": "fin", "FR": "fra", "GA": "gle", "HR": "hrv",
        "HU": "hun", "IT": "ita", "LT": "lit", "LV": "lav", "MT": "mlt", "NL": "nld",
        "PL": "pol", "PT": "por", "RO": "ron", "SK": "slk", "SL": "slv", "SV": "swe",
    }
    match = re.search(r"_([A-Z]{2})\\.pdf$", os.path.basename(filename))
    return code_map.get(match.group(1), "eng") if match else "eng"

# ---------------------------
# Tool registry
# ---------------------------
TOOL_REGISTRY = {
    "docling": DoclingTool(),
    "pymupdf": PyMuPDFTool(),
    "pymupdf_tables": PyMuPDFWithTablesTool(),
    "pdfplumber_mix": PdfPlumberHybridTool(),
    "pymupdf_hybrid": PyMuPDFHybridOrderedTool(),
}

# ---------------------------
# Main converter wrapper
# ---------------------------
def convert_pdf_to_html(pdf_path: str, tool_name: str, output_path: str):
    if tool_name == "unstructured":
        lang = extract_tesseract_lang_from_filename(pdf_path)
        tool = UnstructuredTool(language=lang)
    else:
        tool = TOOL_REGISTRY.get(tool_name)

    if not tool:
        raise ValueError(f"Unsupported tool: {tool_name}")

    html_content = tool.convert(pdf_path)
    if html_content:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logging.info(f"Saved: {output_path}")
    else:
        logging.warning(f"No output: {pdf_path}")

# ---------------------------
# Batch runner
# ---------------------------
from tqdm import tqdm

from tqdm import tqdm

def run_batch_for_tool(pdf_dirs: List[str], output_root: str, tool: str):
    logging.info(f"Starting batch conversion with: {tool}\n")

    all_pdfs = []
    for pdf_root in pdf_dirs:
        for root, _, files in os.walk(pdf_root):
            for file in files:
                if file.endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    # Include category10/category11 in rel_path
                    rel_path = os.path.relpath(pdf_path, start=os.path.commonprefix(pdf_dirs))
                    all_pdfs.append((pdf_path, rel_path))

    for pdf_path, rel_path in tqdm(all_pdfs, desc=f"Converting with {tool}"):
        output_path = os.path.join(output_root, tool, os.path.splitext(rel_path)[0] + ".html")
        convert_pdf_to_html(pdf_path, tool, output_path)


if __name__ == "__main__":
    PDF_DIRS = [
        "/home/4baba/EUR_lex/pdfs_2024/category10",
        "/home/4baba/EUR_lex/pdfs_2024/category19",
    ]
    OUTPUT_ROOT = "/home/4baba/EUR_lex/converted_html_2024"
    SELECTED_TOOL = "pymupdf_hybrid"

    run_batch_for_tool(PDF_DIRS, OUTPUT_ROOT, SELECTED_TOOL)

