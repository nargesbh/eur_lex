import json
import traceback
from pathlib import Path
from spellchecker import SpellChecker
from tqdm import tqdm

# === Config Paths ===
json_input_root = Path("")
json_output_root = Path("")
dict_dir = Path("")

# === SpellChecker Cache ===
spellcheckers = {}

def get_spellchecker(language_code: str) -> SpellChecker:
    if language_code in spellcheckers:
        return spellcheckers[language_code]

    dict_path = dict_dir / f"{language_code}.json"
    if not dict_path.exists():
        return None

    with open(dict_path, "r", encoding="utf-8") as f:
        word_freqs = json.load(f)

    spell = SpellChecker(language=None)
    words = [w for w, freq in word_freqs.items() if freq >= 5]
    spell.word_frequency.load_words(words)
    spellcheckers[language_code] = spell
    return spell

def correct_text(text: str, spell: SpellChecker, file_label: str) -> str:
    corrected_words = []
    words = text.split()

    for word in tqdm(words, desc=f"Spellchecking {file_label}", leave=False, unit="word"):
        if word.isalpha():
            corrected = spell.correction(word)
            corrected_words.append(corrected if corrected else word)
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)

def spellcheck_json(json_file: Path, output_file: Path, spell: SpellChecker):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated_pages = []
    for idx, page in enumerate(data.get("pages", [])):
        page_copy = page.copy()
        natural = page.get("natural_text", "")

        if isinstance(natural, str):
            try:
                parsed = json.loads(natural)
                if isinstance(parsed, dict) and "natural_text" in parsed:
                    corrected = correct_text(parsed["natural_text"], spell, f"{json_file.name} (page {idx+1} inner)")
                    parsed["natural_text"] = corrected
                    page_copy["natural_text"] = json.dumps(parsed, ensure_ascii=False)
                else:
                    corrected = correct_text(natural, spell, f"{json_file.name} (page {idx+1})")
                    page_copy["natural_text"] = corrected
            except json.JSONDecodeError:
                corrected = correct_text(natural, spell, f"{json_file.name} (page {idx+1} raw)")
                page_copy["natural_text"] = corrected
        updated_pages.append(page_copy)

    data["pages"] = updated_pages
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_file}")

def main():
    all_json_files = list(json_input_root.rglob("*.json"))

    for json_file in tqdm(all_json_files, desc="Processing files", unit="file"):
        relative_path = json_file.relative_to(json_input_root)
        output_file = json_output_root / relative_path

        if output_file.exists():
            continue

        lang_code = json_file.stem.split("_")[-1].upper()
        spell = get_spellchecker(lang_code)
        if not spell:
            print(f"No dictionary for {lang_code}, skipping {json_file}")
            continue

        try:
            spellcheck_json(json_file, output_file, spell)
        except Exception as e:
            print(f"Failed for {json_file}: {e}")
            traceback.print_exc()

    print("Spellchecking complete.")

if __name__ == "__main__":
    main()
