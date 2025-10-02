# A data preparation pipeline for documents to JSON (for vectorization / RAG)

import os
import json
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
    Extract equipment names, model numbers, and key context from document start.
    Returns a context string to prepend to chunks.
    """
    # Get first few elements (title, intro paragraphs)
    first_elements = elements[:5] if len(elements) >= 5 else elements
    full_text = " ".join([elem.text for elem in first_elements if hasattr(elem, 'text')])

    # Extract key patterns
    context_parts = []

    # Look for model numbers (e.g., "Model HOM-1107", "MA6BA6", "Oxford-80")
    import re
    model_patterns = [
        r'Model\s+([A-Z0-9-]+)',
        r'model\s+([A-Z0-9-]+)',
        r'\b([A-Z]{2,}[-\s]?\d{2,}[A-Z0-9]*)\b'
    ]

    for pattern in model_patterns:
        matches = re.findall(pattern, full_text)
        if matches:
            context_parts.extend([f"Model: {m}" for m in matches[:2]])
            break

    # Look for equipment types (Fume Hood, RIE, Hotplate, etc.)
    equipment_keywords = [
        'Fume Hood', 'RIE', 'Reactive Ion Etching', 'Hotplate',
        'Mask Aligner', 'Supreme Air', 'Plasmalab', 'Isotemp'
    ]

    for keyword in equipment_keywords:
        if keyword.lower() in full_text.lower():
            context_parts.append(f"Equipment: {keyword}")
            break

    if context_parts:
        return " | ".join(context_parts) + "\n\n"
    return ""


def process_doc_to_json(input_path: Path, output_path: Path):
    """
    Parse a document into JSONL chunks with metadata for better RAG.
    Uses domain-aware chunking with equipment context prepended.
    """
    elements = partition(filename=str(input_path))

    # Extract equipment context from document
    equipment_context = extract_equipment_context(elements)

    chunks = chunk_elements(elements, max_characters=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    # Write as JSONL (one JSON object per line)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            # Prepend equipment context to each chunk
            chunk_text = equipment_context + chunk.text

            chunk_data = {
                "text": chunk_text,
                "metadata": {
                    "filename": input_path.name,
                    "type": chunk.category,
                    "page": getattr(chunk.metadata, "page_number", None),
                },
            }
            f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")

def run_pipeline():
    """
    Process all documents in DOCS_DIR into JSON files under OUTPUT_DIR.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    for file in DOCS_DIR.iterdir():
        if file.is_file():
            out_file = OUTPUT_DIR / f"{file.name}.jsonl"
            process_doc_to_json(file, out_file)
            print(f"Processed {file.name} → {out_file}")
    
def upload_to_existing_vector_store():
    '''
    Uploads the processed JSON files to the existing 'processed_docs' vector store
    '''
    try:
        # Find the existing vector store by name
        vector_stores = client.vector_stores.list()
        target_store = None
        
        for store in vector_stores.data:
            if store.name == VECTOR_STORE_NAME:
                target_store = store
                break
        
        if not target_store:
            print(f"Error: Could not find vector store named '{VECTOR_STORE_NAME}'")
            print("Please make sure you've created it in the OpenAI dashboard first.")
            return
        
        print(f"Found existing vector store: {target_store.id}")
        
        # Get processed JSONL files
        files = [OUTPUT_DIR / f for f in os.listdir(OUTPUT_DIR) if f.endswith(".jsonl")]
        if not files:
            print("No processed files found to upload.")
            return
        
        # Upload files to OpenAI
        uploaded_files = []
        for f in files:
            print(f"Uploading {f.name} ...")
            with open(f, "rb") as file_data:
                upload = client.files.create(file=file_data, purpose="assistants")
                uploaded_files.append(upload.id)
                print(f"  File ID: {upload.id}")

        # Attach uploaded files to the existing vector store
        file_batch = client.vector_stores.file_batches.create(
            vector_store_id=target_store.id,
            file_ids=uploaded_files
        )

        print(f"\nSuccessfully uploaded {len(uploaded_files)} files to vector store '{VECTOR_STORE_NAME}'")
        print(f"Vector store ID: {target_store.id}")
        print(f"File batch ID: {file_batch.id}")
        
    except Exception as e:
        print(f"Error uploading to vector store: {e}")
        print("You can manually add the files to your vector store in the OpenAI dashboard.")

if __name__ == "__main__":
    run_pipeline()
    print("All documents processed into JSON format")
    
    upload_to_existing_vector_store()
    print("Uploaded files to vector store")
