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
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def process_doc_to_json(input_path: Path, output_path: Path):
    """
    Parse a document into JSON chunks with metadata for better RAG.
    """
    elements = partition(filename=str(input_path))
    chunks = chunk_elements(elements, max_characters=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    # Convert chunks to list of dictionaries
    chunks_data = [
        {
            "text": chunk.text,
            "metadata": {
                "filename": input_path.name,
                "type": chunk.category,
                "page": getattr(chunk.metadata, "page_number", None),
            },
        }
        for chunk in chunks
    ]
    
    # Write as JSON
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)



def run_pipeline():
    """
    Process all documents in DOCS_DIR into JSON files under OUTPUT_DIR.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    for file in DOCS_DIR.iterdir():
        if file.is_file():
            out_file = OUTPUT_DIR / f"{file.name}.json" 
            process_doc_to_json(file, out_file)          
            print(f"Processed {file.name} → {out_file}")

def upload_to_vector_store():
    '''
    Uploads the processed JSON files to a new vectorstore in openAi
    '''
    
    #create vector store
    vector_store = client.vector_stores.create(name=VECTOR_STORE_NAME)
    
    #get processed JSON Files
    files = [OUTPUT_DIR / f for f in os.listdir(OUTPUT_DIR) if f.endswith(".json")]
    if not files:
        print("No processed files found to upload.")
        return
    
    #upload files
    uploaded_files = []
    for f in files:
        print(f"Uploading {f.name} ...")
        with open(f, "rb") as file_data:
            upload = client.files.create(file=file_data, purpose="assistants")
            uploaded_files.append(upload.id)

    # Attach uploaded files to vector store
    client.vector_stores.file_batches.create(
        vector_store_id=vector_store.id,
        file_ids=uploaded_files
    )

    print(f"Uploaded {len(uploaded_files)} files to vector store '{VECTOR_STORE_NAME}'")
    
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
        
        # Get processed JSON files
        files = [OUTPUT_DIR / f for f in os.listdir(OUTPUT_DIR) if f.endswith(".json")]
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
