# A data preparation pipeline for documents to JSONL (for vectorization / RAG)

import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from unstructured.partition.auto import partition
from unstructured.chunking.basic import chunk_elements

# Load env variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# CONFIG
DOCS_DIR = Path("docs")
OUTPUT_DIR = Path("processed")
VECTOR_STORE_NAME = "UnstructuredDocs"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def process_doc_to_jsonl(input_path: Path, output_path: Path):
    """
    Parse a document into JSONL chunks with metadata for better RAG.
    """
    elements = partition(filename=str(input_path))
    chunks = chunk_elements(elements, max_characters=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            obj = {
                "text": chunk.text,
                "metadata": {
                    "filename": input_path.name,
                    "type": chunk.category,
                    "page": getattr(chunk.metadata, "page_number", None),
                },
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run_pipeline():
    """
    Process all documents in DOCS_DIR into JSONL files under OUTPUT_DIR.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    for file in DOCS_DIR.iterdir():
        if file.is_file():
            out_file = OUTPUT_DIR / f"{file.name}.jsonl"
            process_doc_to_jsonl(file, out_file)
            print(f"Processed {file.name} → {out_file}")


if __name__ == "__main__":
    run_pipeline()
    print("✅ All documents processed into JSONL format")
