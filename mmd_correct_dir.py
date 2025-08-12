import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
mmd_dir = Path("/ltstorage/shares/datasets/eu/category15/nougat_mmd")
pdf_base_dir = Path("/ltstorage/shares/datasets/eu/category15/pdfs_category15")
output_base = Path("/ltstorage/shares/datasets/eu/category15/nougat_correct_path")

# Go through all .mmd files
mmd_files = list(mmd_dir.glob("*.mmd"))
for mmd_file in tqdm(mmd_files, desc="Copying MMD files to structured paths"):
    name = mmd_file.stem  # e.g., "31975D0437_EN"
    if "_" not in name:
        print(f"Skipped malformed name: {mmd_file.name}")
        continue

    lang = name.split("_")[-1]
    base = name.rsplit("_", 1)[0]

    # Precompute expected relative path
    expected_rel_path = None
    match = list(pdf_base_dir.rglob(f"{base}_{lang}.pdf"))
    if match:
        pdf_path = match[0]
        expected_rel_path = pdf_path.relative_to(pdf_base_dir).with_suffix(".mmd")
        target_path = output_base / expected_rel_path

        # Skip if already exists
        if target_path.exists():
            print(f"Already exists: {target_path}")
            continue

        # Copy
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mmd_file, target_path)
        print(f"Copied: {mmd_file.name} → {target_path}")
    else:
        print(f"No match found for: {mmd_file.name}")
