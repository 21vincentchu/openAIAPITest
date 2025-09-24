# A data preperation pipeline for documents to be turned into JSON's

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from unstructured.partition.auto import partition
from unstructured.chunking.basic import chunk_elements

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

#CONFIG
DOCS_DIR = "docs"
OUTPUT_DIR = "processed"
VECTOR_STORE_NAME = "UnstructuredDocs"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def process_doc_to_jsonl(input_path, output_path, filename):
    """
    Parse documents in the docs folder to JSONL with metadata for better RAG
    input: input path, output path, filename
    output: JSONL 
    """    
    
    elements = partition(filename=input_path)
    chunks = chunk_elements(elements, max_characters=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            obj = {
                "text": chunk.text,
                "metadata": {
                    "filename": filename,
                    "type": chunk.category,
                    "page": getattr(chunk.metadata, "page_number", None),
                },
            }
            f.write(json.dumps(obj) + "\n")

