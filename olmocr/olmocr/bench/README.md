# olmOCR-Bench

We develop olmOCR-Bench in order to automatically and effectively evaluate document-level OCR of various tools.

olmOCR-Bench works by testing various "facts" about document pages at the PDF-level.
Our intention is that each "fact" is very simple, unambiguous, and machine-checkable. For example, once your document has been OCRed, we may check that a particular sentence appears somewhere on the page.

We stay away from soft metrics like edit distance comparisons, because they may assign lower scores for parses of the document that differ from the reference, but may in fact still be correct. For example, on a document containing multiple distinct articles: you want the text of each article to be grouped together, but the relative order of the two articles may not be critical. Also, some documents may have critical details, like switching x and y in an equation that can make all the difference in understanding, but would appear as just a single character edit in an edit-distance metric.

olmOCR-bench operates on single page PDFs directly. We make this choice because PDFs do preserve some digital metadata and information which may be helpful to some OCR systems. Almost any other format can be converted to a PDF, but not the reverse, so we try to preserve these original documents where possible.

## Benchmark Principles

As we created olmOCR-bench, we also kept a few general rules in mind:

- We expect your OCR system to output a plain-text Unicode document in a reading order that would be considered natural.
- Documents from the benchmark should fit on a standard A4 piece of paper and still be readable to a human.
- Markdown syntax is allowed, but ignored. Ex. if we are looking for the word "enlightenment" to appear on a page, and your system outputs "**\*\*enlightenment\*\***" in Markdown bold, that still counts. 
- olmOCR-bench is not position sensitive, ex. we check that a sentence or math equation appears anywhere on a page. The exception to this is header/footer tests where we want to find simple page numbers appearing in the first or last few characters of a page.
- Tables can be in either Markdown syntax, or as an html `<table>`.
- Math equations must render with [Katex](https://katex.org/) and be delimeted with $, $$, \\(, or \\[. 
- Math equations are not position sensitive either, so if we are checking for 
$ 3x^2 $ to appear on a page, then outputting $ \int_a^b{ 3x ^ 2dx} $ counts.
- We normalize all Unicode to NFC before running the benchmark, so if your OCR model outputs é vs e + ◌́ then either way should not affect your benchmark score.
- We normalize all the different variants of hyphens to the ascii -, all the variants of double quoets to ascii " and all variants of single quotes/apostrophes to ascii '. You should score the same on the benchmark if you output - vs —
- All facts checked about documents are either pass/fail. We want it to be very clear if your OCR system fails a test, and if so, what output would make it pass.

## olmOCR-Bench Fact classes

- Text presence
  - This task makes sure that a given small piece of text (ex. 1-3 sentence level) is present within
    a parsed document. Soft/fuzzy matching is allowed, as well as specifying if the text must be in the first N or last N characters of the document. Case sensitive by default.
- Text absense
  - This task makes sure that a given piece of next does NOT appear in the OCR'ed version of a document. We generally want our OCR systems to filter out content like headers/footers/page numbers from documents. The same fuzzy matching as in Text Presence tests is allowed.
- Natural Reading Order
  - This task ensures that blocks of text which are present have a defined order relative to one another. For example,
  on a document that contains multiple news articles on one page, you'd want to see that the first sentence of the 
  first article appears after the heading of that article. But, you may be okay with swapping the order of those 
  two articles.
- Table Accuracy
  - Both Markdown and HTML based tables are supported. These tests check that a cell with a given text exists somewhere in the table, and that its neighbors have certain properties. Ex. A cell exists on this page with text "4.5%" and above that is a cell with the text "2.4%"
- Math Formula Accuracy
  - We render a given Latex style equation using Katex in a headless browser. And then see if it exists anywhere in the final OCRed document. Matching is performed on a relative symbol level, ex. in "\f\relax{x} = \int_{-\infty}^\infty
    x^2dx" we check that a ∫ appears to the left of a x, x appears to the left of dx, etc...
  
## Downloading and running the benchmark

Currently the full benchmark data is located here, but it's private until we are done reviewing and checking all of the tests:
https://huggingface.co/datasets/allenai/olmOCR-bench

To run a benchmark, first install the bench requirements
```bash
conda create -n olmocr python=3.11
conda activate olmocr

git clone https://github.com/allenai/olmocr.git
cd olmocr

pip install -e .[bench]

# Now clone the benchmark data
git clone https://huggingface.co/datasets/allenai/olmOCR-bench
```

Convert your documents
```bash
# convert using a single OCR-engine, see the olmocr/bench/runners directory for options
python -m olmocr.bench.convert olmocr_pipeline --dir ./olmOCR-bench/bench_data

# or use convert_all.sh to run OCR with many common frameworks all at once, API keys will be required
./olmocr/bench/scripts/convert_all.sh
```

Now run the benchmark
```bash
python -m olmocr.bench.benchmark --dir ./olmOCR-bench/bench_data
```

## Previewing the benchmark questions

We have an internal data annotation tool that can be used to review the questions in the benchmark, and make edits.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/dd24fd88-a642-4379-b5a1-9911717bf5b1" />


```bash
python -m olmocr.bench.review_app --port 5000 --debug ./olmOCR-bench/bench_data/multi_column.jsonl --force
```

## How were the tests made

Several categories of tests have been made so far:
1. arxiv_math -> We downloaded recent math papers from arxiv, filtered to those which had a single tex source file, and a rendered pdf, using https://github.com/allenai/olmocr/blob/main/olmocr/bench/miners/download_math.py. Then we matched up the text on a pdf page to the location in the tex source mostly likely to match to it using a dynamic programming matching algorithm in https://github.com/allenai/olmocr/blob/main/olmocr/bench/miners/mine_math.py. From there, Latex equations from the matching page were then parsed out, and we checked they rendered in Katex before adding them as test cases. We did a final quick scan over the data manually to remove any cases where the Latex parsing may have failed egregiously.
2. headers_footers -> We sampled documents from our internal crawled PDF repository. (The same from which olmOCR-mix was derived, though the likelyhood of duplicates is low, as there are 200M+ pdfs in this set). Then we used [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) to identify regions of the pages which were marked as headers/footers using the abandon category. We then got the text of those headers/footers regions by extracting them out and prompting Gemini, and we added them as test cases which should be absent. Manual review was then performed to remove mistakenly filtered text, and to set conditions such as limiting the search area to the first N or last N characters. Ex. if a page number "5" appears on the bottom a page, you want to test that your OCR system does not output a "5" in the last 20 characters of the page, but "5" could apepar earlier if in the actual body text.
3. table_tests -> We sampled documents from our internal crawled PDF repository, and found those which had tables using gemini-flash-2.0. https://github.com/allenai/olmocr/blob/main/olmocr/bench/miners/mine_tables_gemini.py On pages that had tables, we then further asked gemini-flash-2.0 to tell us the relationships between randomly chosen cells. Those tests were then manually checked.
4. multi_column -> We sampled documents from our internal crawled PDF repository manually, to find documents which had multi-column layouts and multiple articles on one page. Then, we used claude-sonnet-3.7 to render those pages to html, and from that html, we extracted text segments which were before/after one another. Then we manually reviewed each entry.
5. old_scans -> We sampled documents from the Library of Congress which contained handwriting or typewritten text. Then we priortized creating rules that check for reading order. (TODO)
6. old_scans_math -> We found old math textbooks in the public domain from the Internet Archive. We then extracted random pages from them, OCRed them, filtered down to pages which contained equations, and picked several random equations from each page to use as test cases. We then manually checked each test case to see that it was accurate capturing what was on the page.
7. long_tiny_text -> We found documents from the Internet Archive which contained a large amount of dense small print on a single page. Ex. pages from a dictionary, or pages of references from an academic paper. We then generated test cases using an LLM, and verified them manually.


## TODO List for release
 - [ ] Check all tests for duplicates
 - [ ] Make absense tests not case sensitive by default
 - [ ] Check that we have URLs for all tests
 - [ ] Write a script to verify that all baseline tests that actually have weird unicodes have exemptions
 - [X] Review math equations in old_scans_math.jsonl using chat gpt script
 - [X] Add test category of long_texts which are still ~1 standard printed page, but with dense/small text
 - [ ] Review multicolumn_tests, make sure they are correct, clean, and don't have order tests between regions
 - [ ] Run automated check of multicolumn tests for: #1 sub/super scripts #2 max diffs calibrations #3 mixing across different distinct regions of text 
 - [X] Remove [] and other special symbols from old_scans
 - [X] Full review of old_scans, somehow, chatgpt or prolific
 - [X] Adjust scoring to weight each test category equally in final score distribution
 - [X] Double check marker inline math outputs
 - [ ] Run against final set of comparison tools, and check list of all-pass and all-fail tests
