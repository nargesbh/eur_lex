import os
import logging
import requests
from time import time, sleep
from tqdm import tqdm

from extractors.celexdocument_list import celex_main  # gets list of CELEX IDs


def fetch_and_save_html(lang, celex_id, save_dir):
    """
    Fetch and save the official HTML version from EUR-Lex.
    """
    url = f'https://eur-lex.europa.eu/legal-content/{lang}/TXT/HTML/?uri=CELEX:{celex_id}'
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        html_content = response.text

        if 'The requested document does not exist.' in html_content:
            logging.warning(f"[{celex_id}][{lang}] HTML not found.")
            return

        os.makedirs(save_dir, exist_ok=True)
        html_path = os.path.join(save_dir, f"{celex_id}_{lang}.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logging.info(f"[{celex_id}][{lang}] HTML saved.")
    except Exception as e:
        logging.warning(f"[{celex_id}][{lang}] Failed to fetch HTML: {e}")


def fetch_and_save_pdf(lang, celex_id, save_dir):
    """
    Fetch and save the PDF version from EUR-Lex.
    """
    url_pdf = f'https://eur-lex.europa.eu/legal-content/{lang}/TXT/PDF/?uri=CELEX:{celex_id}'
    try:
        response = requests.get(url_pdf, timeout=15)
        if 'The requested document does not exist.' in response.text:
            raise Exception("PDF not available")

        os.makedirs(save_dir, exist_ok=True)
        pdf_path = os.path.join(save_dir, f"{celex_id}_{lang}.pdf")
        with open(pdf_path, 'wb') as f:
            f.write(response.content)

        logging.info(f"[{celex_id}][{lang}] PDF saved.")
    except Exception as e:
        logging.warning(f"[{celex_id}][{lang}] Failed to fetch PDF: {e}")


def run_fetch(years, domains):
    langs = ['BG', 'ES', 'CS', 'DA', 'DE', 'ET', 'EL', 'EN', 'FR',
            'GA', 'HR', 'IT', 'LV', 'LT', 'HU', 'MT', 'NL', 'PL',
            'PT', 'RO', 'SK', 'SL', 'FI', 'SV'
             ]

    base_dir = os.getcwd()

    # Create a top-level directory for PDFs and HTMLs
    pdf_base_dir = os.path.join(base_dir, "pdfs_2024")
    html_base_dir = os.path.join(base_dir, "htmls_2024")
    os.makedirs(pdf_base_dir, exist_ok=True)
    os.makedirs(html_base_dir, exist_ok=True)

    logs_path = os.path.join(base_dir, "Logs_Official_HTML_Download.log")
    logging.basicConfig(filename=logs_path,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%d-%b-%y %H:%M:%S')

    start_time = time()

    for domain in domains:
        domain_str = f'{domain:02d}'
        
        # Create category directories for PDFs and HTMLs
        pdf_category_dir = os.path.join(pdf_base_dir, f"category{domain_str}")
        html_category_dir = os.path.join(html_base_dir, f"category{domain_str}")
        os.makedirs(pdf_category_dir, exist_ok=True)
        os.makedirs(html_category_dir, exist_ok=True)

        base_url = f'https://eur-lex.europa.eu/search.html?name=browse-by%3Alegislation-in-force&type=named&displayProfile=allRelAllConsDocProfile&qid=1651004540876&CC_1_CODED={domain_str}'

        for year in years:
            print(f"Fetching domain {domain_str}, year {year}")
            final_url = base_url + f"&DD_YEAR={year}"
            celex_ids = celex_main(final_url)

            # Fetch and save PDF and HTML for each CELEX ID
            for celex_id in tqdm(celex_ids):
                # Create a law directory for each CELEX ID in the category directory
                law_pdf_dir = os.path.join(pdf_category_dir, f"law{celex_id}")
                law_html_dir = os.path.join(html_category_dir, f"law{celex_id}")
                os.makedirs(law_pdf_dir, exist_ok=True)
                os.makedirs(law_html_dir, exist_ok=True)

                # Save HTML and PDF for each language
                for lang in langs:
                    fetch_and_save_html(lang, celex_id, law_html_dir)
                    fetch_and_save_pdf(lang, celex_id, law_pdf_dir)
                    sleep(1)  # Avoid rate limiting

    end_time = time()
    logging.info(f"Execution completed in {end_time - start_time:.2f} seconds")
    
def run_fetch_for_subgroup(pdf_base_dir, html_base_dir):
    langs = ['BG', 'ES', 'CS', 'DA', 'DE', 'ET', 'EL', 'EN', 'FR',
             'GA', 'HR', 'IT', 'LV', 'LT', 'HU', 'MT', 'NL', 'PL',
             'PT', 'RO', 'SK', 'SL', 'FI', 'SV']

    category_num = 15  # Main domain
    subgroup_code = 1510

    os.makedirs(pdf_base_dir, exist_ok=True)
    os.makedirs(html_base_dir, exist_ok=True)

    logs_path = os.path.join(os.getcwd(), f"Logs_Category{category_num}_{subgroup_code}.log")
    logging.basicConfig(filename=logs_path,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%d-%b-%y %H:%M:%S')

    start_time = time()

    base_url = f'https://eur-lex.europa.eu/search.html?name=browse-by%3Alegislation-in-force&type=named&displayProfile=allRelAllConsDocProfile&CC_1_CODED={category_num}&CC_2_CODED={subgroup_code}'

    for year in range(2025, 2026):
        print(f"Fetching category {category_num}, subgroup {subgroup_code}, year {year}")
        final_url = base_url + f"&DD_YEAR={year}"
        celex_ids = celex_main(final_url)

        if not celex_ids:
            logging.info(f"No CELEX IDs found for year {year}")
            continue
        print(f"[run_fetch] Found {len(celex_ids)} CELEX IDs for year {year}")
        for celex_id in tqdm(celex_ids):
            pdf_dir = os.path.join(pdf_base_dir, f"{year}", f"law{celex_id}")
            html_dir = os.path.join(html_base_dir, f"{year}", f"law{celex_id}")
            os.makedirs(pdf_dir, exist_ok=True)
            os.makedirs(html_dir, exist_ok=True)

            for lang in langs:
                print(f"[run_fetch] Fetching HTML for CELEX: {celex_id}, lang: {lang}")
                fetch_and_save_html(lang, celex_id, html_dir)
                print(f"[run_fetch] Fetching PDF for CELEX: {celex_id}, lang: {lang}")
                fetch_and_save_pdf(lang, celex_id, pdf_dir)
                sleep(1)

    end_time = time()
    logging.info(f"Execution completed in {end_time - start_time:.2f} seconds")


# if __name__ == "__main__":
#     selected_years = [2024]  # Fetching laws for 2024
#     selected_domains = range(10, 21)  # Domains 10 to 20
#     run_fetch(selected_years, selected_domains)

if __name__ == "__main__":
    pdf_output = "/ltstorage/home/4baba/EUR_lex/category15/pdfs_category15"
    html_output = "/ltstorage/home/4baba/EUR_lex/category15/htmls_category15"
    run_fetch_for_subgroup(pdf_output, html_output)


