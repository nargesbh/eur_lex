import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import boto3
from tqdm import tqdm

# Configuration
BUCKET = "ai2-llm"
PREFIX = "pretraining-data/sources/soldni-open-access-books/v0/pipeline/results"
OUTPUT_FILENAME = "all_completed_files.txt"


def process_file(key: str):
    """
    Process a single S3 file given by its key.
    Reads a jsonl file from S3, decodes each line,
    extracts the 'Source-File' from the 'metadata' field,
    and returns a list of these source file strings.
    """
    # Create a new S3 client in the worker thread (thread-safe)
    s3 = boto3.client("s3")
    extracted_lines = []
    try:
        response = s3.get_object(Bucket=BUCKET, Key=key)
        for raw_line in response["Body"].iter_lines():
            try:
                # Decode the line from bytes to text
                line_str = raw_line.decode("utf-8")
            except UnicodeDecodeError as e:
                print(f"Skipping a line in {key} due to decode error: {e}")
                continue
            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as e:
                print(f"Skipping a malformed json line in {key}: {e}")
                continue
            # Extract 'Source-File' from metadata if present
            metadata = data.get("metadata", {})
            source_file = metadata.get("Source-File")
            if source_file:
                extracted_lines.append(source_file)
    except Exception as e:
        print(f"Error processing file {key}: {e}")
    return extracted_lines


def main():
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=BUCKET, Prefix=PREFIX)

    # Gather all S3 object keys under the specified prefix
    keys = []
    for page in page_iterator:
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            keys.append(obj["Key"])

    print(f"Found {len(keys)} files to process.")

    # Open the output file for writing
    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as output_file:
        # Create a thread pool to process files concurrently.
        # Adjust max_workers based on your environment and workload.
        with ProcessPoolExecutor() as executor:
            # Submit all processing jobs and map each future to its key
            future_to_key = {executor.submit(process_file, key): key for key in keys}
            # Use tqdm to wrap the as_completed iterator for progress display
            for future in tqdm(as_completed(future_to_key), total=len(future_to_key), desc="Processing files"):
                try:
                    source_files = future.result()
                    # Write each extracted line to the output file as soon as the future completes
                    for source in source_files:
                        output_file.write(source + "\n")
                    # Optionally flush after each completed task
                    output_file.flush()
                except Exception as e:
                    key = future_to_key[future]
                    print(f"Exception occurred for file {key}: {e}")

    print(f"Finished writing the source file names to {OUTPUT_FILENAME}")


if __name__ == "__main__":
    main()
