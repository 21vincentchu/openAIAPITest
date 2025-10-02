"""
Data preparation pipeline for documents to JSONL format.

This pipeline:
1. Processes documents (PDF, DOCX, etc.) using Unstructured
2. Chunks content with overlap for better RAG retrieval
3. Extracts equipment context (model numbers, types) from document headers
4. Uploads processed files to OpenAI vector store for semantic search

Configuration is managed via constants at module level.
"""
import os
import json
import re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from unstructured.partition.auto import partition
from unstructured.chunking.basic import chunk_elements

os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

# Load env variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# CONFIG
DOCS_DIR = Path("docs")
OUTPUT_DIR = Path("processed")
VECTOR_STORE_NAME = "processed_docs"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

def extract_equipment_context(elements):
    """
    Extract equipment identifiers from document headers to improve RAG retrieval.

    Scans the first 5 elements for:
    - Model numbers (e.g., "Model HOM-1107", "MA6BA6")
    - Equipment types (e.g., "Fume Hood", "RIE")

    Args:
        elements: List of document elements from Unstructured parser

    Returns:
        str: Formatted context string to prepend to chunks, or empty string if none found
    """
    # Combine text from document header
    first_elements = elements[:5]
    full_text = " ".join([elem.text for elem in first_elements if hasattr(elem, 'text')])

    context_parts = []

    # Extract model numbers using combined pattern
    model_pattern = r'(?:Model\s+)?([A-Z]{2,}[-\s]?\d{2,}[A-Z0-9]*)\b'
    matches = re.findall(model_pattern, full_text, re.IGNORECASE)
    if matches:
        context_parts.extend([f"Model: {m}" for m in matches[:2]])

    # Extract equipment type
    equipment_keywords = [
        'Fume Hood', 'RIE', 'Reactive Ion Etching', 'Hotplate',
        'Mask Aligner', 'Supreme Air', 'Plasmalab', 'Isotemp'
    ]

    for keyword in equipment_keywords:
        if keyword.lower() in full_text.lower():
            context_parts.append(f"Equipment: {keyword}")
            break

    return " | ".join(context_parts) + "\n\n" if context_parts else ""


def process_doc_to_json(input_path: Path, output_path: Path):
    """
    Parse a document into JSONL chunks with metadata.

    Args:
        input_path: Path to input document (PDF, DOCX, etc.)
        output_path: Path for output JSONL file

    Process:
        1. Parse document into elements using Unstructured
        2. Extract equipment context from header
        3. Chunk content with overlap (configured via CHUNK_SIZE/CHUNK_OVERLAP)
        4. Prepend context to each chunk for better retrieval
        5. Write JSONL with text + metadata (filename, type, page)
    """
    elements = partition(filename=str(input_path))
    equipment_context = extract_equipment_context(elements)
    chunks = chunk_elements(elements, max_characters=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            chunk_data = {
                "text": equipment_context + chunk.text,
                "metadata": {
                    "filename": input_path.name,
                    "type": chunk.category,
                    "page": getattr(chunk.metadata, "page_number", None),
                },
            }
            f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")

def run_pipeline():
    """
    Process all documents in DOCS_DIR into JSONL files.

    Creates OUTPUT_DIR if needed, then processes each file.
    Skips directories and handles errors gracefully.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    doc_files = [f for f in DOCS_DIR.iterdir() if f.is_file()]

    for i, file in enumerate(doc_files, 1):
        try:
            out_file = OUTPUT_DIR / f"{file.name}.jsonl"
            process_doc_to_json(file, out_file)
            print(f"[{i}/{len(doc_files)}] Processed {file.name} → {out_file.name}")
        except Exception as e:
            print(f"[{i}/{len(doc_files)}] ERROR processing {file.name}: {e}")
    
def upload_to_existing_vector_store():
    """
    Upload processed JSONL files to OpenAI vector store for semantic search.

    Process:
        1. Find vector store by name (VECTOR_STORE_NAME)
        2. Upload all JSONL files from OUTPUT_DIR
        3. Create file batch to attach files to vector store

    Requires vector store to exist - create it in OpenAI dashboard first.
    """
    try:
        # Find vector store by name
        vector_stores = client.vector_stores.list()
        target_store = next(
            (store for store in vector_stores.data if store.name == VECTOR_STORE_NAME),
            None
        )

        if not target_store:
            print(f"Error: Vector store '{VECTOR_STORE_NAME}' not found.")
            print("Create it in the OpenAI dashboard first.")
            return

        print(f"Found vector store: {target_store.id}")

        # Get JSONL files
        files = list(OUTPUT_DIR.glob("*.jsonl"))
        if not files:
            print("No processed files found to upload.")
            return

        # Upload files to OpenAI
        uploaded_files = []
        for i, f in enumerate(files, 1):
            print(f"[{i}/{len(files)}] Uploading {f.name}...")
            with open(f, "rb") as file_data:
                upload = client.files.create(file=file_data, purpose="assistants")
                uploaded_files.append(upload.id)
                print(f"  → File ID: {upload.id}")

        # Attach to vector store
        file_batch = client.vector_stores.file_batches.create(
            vector_store_id=target_store.id,
            file_ids=uploaded_files
        )

        print(f"\n✓ Uploaded {len(uploaded_files)} files to '{VECTOR_STORE_NAME}'")
        print(f"  Vector store: {target_store.id}")
        print(f"  Batch ID: {file_batch.id}")

    except Exception as e:
        print(f"Error: {e}")
        print("You can manually upload files in the OpenAI dashboard.")

if __name__ == "__main__":
    run_pipeline()
    print("All documents processed into JSON format")
    
    upload_to_existing_vector_store()
    print("Uploaded files to vector store")
