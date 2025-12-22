# EUR-Lex Embedding Fine-Tuning and Retrieval

This repository contains the code used to preprocess EUR-Lex legal documents, fine-tune embedding models on legislative text, and evaluate metadata-to-document retrieval performance.

The preprocessing pipeline converts EUR-Lex PDFs into structured JSONL files. This includes PDF-to-HTML conversion,  extraction of textual content, and quality checks against ground-truth HTML. These steps are implemented in the scripts located in the root directory.

All embedding-related code is contained in the `embedding_model/` directory. This includes scripts for extracting metadata and creating ChromaDB vector indices for contrastive learning.

The `embedding_model/fine_tuning/` folder contains the scripts used to construct query–document pairs and perform monolingual and multilingual contrastive fine-tuning of embedding models.

The `embedding_model/test_chromadb/` folder contains scripts for evaluating retrieval performance using a ChromaDB vector index under different search settings.

External tools used for document processing and scraping are included under nougat/, olmocr/, and eur-lex-sum/. File paths were generalized to prepare the code for submission, and trained model checkpoints are not included. edit this part nougat/, olmocr/ are two tools i used for pdf to txt conversion and eur-lex-sum/ is for scrapping the website
