import os
import re
import logging
import time
import argparse
import csv
from pathlib import Path
from typing import List
from collections import defaultdict

import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from tqdm import tqdm
from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------
# Docling Tool
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
            html_out = document.export_to_html()
            if not html_out.strip():
                logging.warning(f"[Docling] {os.path.basename(pdf_file)} is empty")
            return html_out
        except Exception as e:
            logging.error(f"[Docling] Error: {e}")
            return ""


# ---------------------------
# PyMuPDF Tool (basic HTML)
# ---------------------------
class PyMuPDFTool:
    def convert(self, pdf_file: str) -> str:
        try:
            doc = fitz.open(pdf_file)
            html_out = "<html><body>\n"
            for i, page in enumerate(doc):
                html_out += f"<h2>Page {i + 1}</h2>\n"
                html_out += page.get_text("html") + "\n"
            html_out += "</body></html>"
            return html_out
        except Exception as e:
            logging.error(f"[PyMuPDF] Error: {e}")
            return ""


# ---------------------------
# PyMuPDF with Tables
# ---------------------------
class PyMuPDFWithTablesTool:
    def convert(self, pdf_file: str) -> str:
        try:
            doc = fitz.open(pdf_file)
            html_out = "<html><body>\n"
            for i, page in enumerate(doc):
                html_out += f"<h2>Page {i + 1}</h2>\n"
                html_out += page.get_text("html") + "\n"
                try:
                    tables = page.find_tables(strategy="text")
                    for j, table in enumerate(tables):
                        df = pd.DataFrame(table.extract())
                        html_out += f"<h3>Structured Table {j + 1}</h3>\n"
                        html_out += df.to_html(index=False, border=1) + "\n"
                except Exception as e:
                    logging.warning(f"[PyMuPDFTables] Table error on page {i + 1}: {e}")
            html_out += "</body></html>"
            return html_out
        except Exception as e:
            logging.error(f"[PyMuPDFTables] Error: {e}")
            return ""


# ---------------------------
# PdfPlumber + PyMuPDF Hybrid
# ---------------------------
class PdfPlumberHybridTool:
    def convert(self, pdf_file: str) -> str:
        try:
            html_out = "<html><body>\n"
            with pdfplumber.open(pdf_file) as plumber_pdf, fitz.open(pdf_file) as pymupdf_pdf:
                for i, (plumber_page, pymupdf_page) in enumerate(zip(plumber_pdf.pages, pymupdf_pdf), start=1):
                    html_out += f"<h2>Page {i}</h2>\n"
                    text = plumber_page.extract_text()
                    html_out += f"<div style='white-space: pre-wrap;'>{text}</div>\n" if text else "<div><i>No text found.</i></div>\n"
                    try:
                        tables = pymupdf_page.find_tables(strategy="text")
                        for j, table in enumerate(tables):
                            df = pd.DataFrame(table.extract())
                            html_out += f"<h3>Table {j + 1}</h3>\n" + df.to_html(index=False, border=1)
                    except Exception as e:
                        logging.warning(f"[Hybrid] Table error on page {i}: {e}")
            html_out += "</body></html>"
            return html_out
        except Exception as e:
            logging.error(f"[Hybrid] Failed to convert {pdf_file}: {e}")
            return ""


# ---------------------------
# PyMuPDF Hybrid Ordered
# ---------------------------
class PyMuPDFHybridOrderedTool:
    def convert(self, pdf_file: str) -> str:
        try:
            doc = fitz.open(pdf_file)
            html_out = "<html><body>\n"
            for i, page in enumerate(doc):
                html_out += f"<h2>Page {i + 1}</h2>\n"
                elements = []
                blocks = page.get_text("dict")["blocks"]
                for block in blocks:
                    if block["type"] == 0:
                        y0 = block["bbox"][1]
                        paragraph = ""
                        for line in block["lines"]:
                            line_text = " ".join([span["text"] for span in line["spans"]])
                            paragraph += line_text.strip() + " "
                        if paragraph.strip():
                            elements.append((y0, f"<p>{paragraph.strip()}</p>"))
                try:
                    tables = page.find_tables(strategy="text")
                    for j, table in enumerate(tables):
                        y0 = table.bbox[1]
                        df = pd.DataFrame(table.extract())
                        elements.append((y0, f"<h3>Table {j + 1}</h3>\n" + df.to_html(index=False, border=1)))
                except Exception as e:
                    logging.warning(f"[PyMuPDFHybridOrdered] Table error on page {i + 1}: {e}")
                elements.sort(key=lambda x: x[0])
                for _, content in elements:
                    html_out += content + "\n"
            html_out += "</body></html>"
            return html_out
        except Exception as e:
            logging.error(f"[PyMuPDFHybridOrdered] Error: {e}")
            return ""


# ---------------------------
# Unstructured Tool (page number version only)
# ---------------------------
class UnstructuredTool:
    def __init__(self, language="eng"):
        self.language = language

    def extract_elements(self, pdf_file: str):
        try:
            return partition_pdf(
                filename=pdf_file,
                strategy="hi_res",
                infer_table_structure=True,
                model_name="yolox",
                languages=[self.language],
            )
        except Exception as e:
            logging.error(f"[Unstructured] Failed to extract elements from {pdf_file}: {e}")
            return []

    def _tag_for_element(self, el: Element) -> str:
        cat = el.category.lower()
        return {
            "title": "h1", "header": "h2", "footer": "footer",
            "caption": "figcaption", "list": "ul", "code": "code",
            "email": "address", "address": "address",
            "page": "hr", "break": "hr"
        }.get(cat, "p")

    def convert(self, pdf_file: str) -> str:
        elements = self.extract_elements(pdf_file)
        pages = defaultdict(list)
        for el in elements:
            page_num = getattr(el.metadata, "page_number", 1)
            pages[page_num].append(el)
        html_parts = ["<html><body>"]
        for page_num in sorted(pages.keys()):
            html_parts.append(f"<h2>Page {page_num}</h2>")
            for el in pages[page_num]:
                try:
                    if hasattr(el.metadata, "text_as_html") and el.metadata.text_as_html:
                        html_parts.append(el.metadata.text_as_html)
                    else:
                        tag = self._tag_for_element(el)
                        html_parts.append(f"<{tag}>{el.text.strip()}</{tag}>")
                except Exception as e:
                    logging.warning(f"[Unstructured] Error in element conversion: {e}")
        html_parts.append("</body></html>")
        return "\n".join(html_parts)


# ---------------------------
# Tool Registry
# ---------------------------
TOOL_REGISTRY = {
    "docling": DoclingTool(),
    "pymupdf": PyMuPDFTool(),
    "pymupdf_tables": PyMuPDFWithTablesTool(),
    "hybrid": PdfPlumberHybridTool(),
    "pymupdf_hybrid": PyMuPDFHybridOrderedTool(),
    "unstructured": UnstructuredTool()
}


# ---------------------------
# Conversion Logic
# ---------------------------
def convert_single_pdf(pdf_file: Path, output_dir: Path, tool_name: str, timing_writer=None, skip_existing=False):
    tool = TOOL_REGISTRY[tool_name]
    rel_path = pdf_file.with_suffix(".html").name
    output_path = output_dir / tool_name / rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and output_path.exists():
        logging.info(f"[{tool_name}] Skipping existing: {output_path}")
        return
    logging.info(f"[{tool_name}] Processing: {pdf_file}")
    start_time = time.time()
    html_content = tool.convert(str(pdf_file))
    duration = time.time() - start_time
    if html_content.strip():
        output_path.write_text(html_content, encoding="utf-8")
        if timing_writer:
            timing_writer.writerow({"Filepath": str(pdf_file), "method": tool_name, "time": duration})
        logging.info(f"[{tool_name}] Saved: {output_path}")
    else:
        logging.warning(f"[{tool_name}] Empty output for {pdf_file}")


def main():
    parser = argparse.ArgumentParser(description="PDF to HTML conversion using multiple tools")
    parser.add_argument("--input", required=True, help="Path to PDF file or directory")
    parser.add_argument("--output", required=True, help="Output root directory")
    parser.add_argument("--tools", nargs="+", required=True, choices=list(TOOL_REGISTRY.keys()), help="Tools to run")
    parser.add_argument("--timing-csv", help="Path to timing CSV file")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already processed files")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output)
    pdf_files = [input_path] if input_path.is_file() else list(input_path.rglob("*.pdf"))

    timing_writer = None
    csv_file = None
    if args.timing_csv:
        timing_csv_path = Path(args.timing_csv)
        timing_csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_exists = timing_csv_path.exists()
        csv_file = open(timing_csv_path, mode="a", newline="", encoding="utf-8")
        timing_writer = csv.DictWriter(csv_file, fieldnames=["Filepath", "method", "time"])
        if not csv_exists:
            timing_writer.writeheader()

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        for tool_name in args.tools:
            convert_single_pdf(pdf_file, output_root, tool_name, timing_writer, args.skip_existing)

    if csv_file:
        csv_file.close()


if __name__ == "__main__":
    main()
