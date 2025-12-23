# EUR-Lex Embedding Fine-Tuning and Retrieval

This repository contains the code used to (1) preprocess EUR-Lex documents, (2) build training data and contrastive fine-tuning, (3) create a ChromaDB vector index, and (4) evaluate metadata-to-document retrieval.

The repository includes multiple conversion/evaluation utilities (e.g., alternative conversion routes and quality checks). In the final pipeline used for the paper, we rely on **olmOCR for PDF-to-text extraction** and produce **structured JSONL** as the main intermediate representation for the embedding and retrieval experiments.

---

## 1) Preprocessing and dataset preparation (root-level scripts)

These scripts prepare the dataset and run quality checks. They support different conversion and evaluation steps; the core goal is to produce clean, structured representations of legal acts for downstream retrieval.

### Core extraction / preparation
- **`extract_and_save_jsonl.py`**  
  Produces structured JSONL documents from processed sources. This is the main preprocessing output used by the embedding pipeline.

### Quality evaluation
- **`json_evaluation_content2.py`**  
  Evaluates extracted JSON content against a reference representation (Lexical Content Similarity).
---

## 2) Embedding and retrieval (`embedding_model/`)

All code related to embedding models, fine-tuning, indexing, and retrieval evaluation lives under `embedding_model/`.

### 2.1 Data preparation for retrieval and training
- **`extract_metadata.py`**  
  Extracts the metadata block used as the *query* for metadata-to-document retrieval.

- **`json_to_txt_converter.py`**  
  Converts structured documents into plain-text representations used for indexing and retrieval.

---

## 3) Fine-tuning (`embedding_model/fine_tuning/`)

This folder contains scripts for constructing training data and running contrastive fine-tuning.

- **`1_create_fineTuning_pair.py`**  
  Creates contrastive training examples by pairing metadata queries with their matching documents (positives) and adding non-matching documents as negatives.

- **`2_split_jsonl.py`**  
  Splits the prepared dataset into training/validation/test partitions for controlled experiments.

- **`merge_langs_data.py`**  
  Combines data across languages to enable multilingual training runs.

- **`fine_tune_full.py`**  
  Runs monolingual fine-tuning for a selected language/model configuration.

- **`multi_lang_fine_tune.py`**  
  Runs multilingual fine-tuning using merged multi-language training data.

---

## 4) Retrieval evaluation (`embedding_model/test_chromadb/`)

This folder runs retrieval experiments against a ChromaDB index and summarizes Top-k accuracy.

- **`create_chroma_db.py`**  
  Builds a ChromaDB vector index for a document collection using a selected fine tuned model. This index is used for retrieval experiments.

- **`test_chromadb.py`**  
  Executes retrieval by embedding queries, searching the vector index, and storing ranked retrieval outputs.

- **`test_result_analysis.py`**  
  Aggregates retrieval outputs and computes metrics (e.g., Top-1/Top-3/Top-5 accuracy), producing results used for tables/figures.

---

## External tools

- **`nougat/`** and **`olmocr/`**: third-party tools used for **PDF-to-text conversion** during document preprocessing.  
- **`eur-lex-sum/`**: utilities for **scraping** documents and metadata from the EUR-Lex website.


---

## Data availability

The dataset used in this work will be released after the paper review process.  
Due to its size, the full dataset could not be uploaded as part of the OpenReview submission and is therefore not publicly accessible during review. Upon acceptance, we will make the data available via an appropriate hosting platform and update this repository with access instructions.


---


## License

This project is released under the **MIT License**. See the `LICENSE` file for details.
