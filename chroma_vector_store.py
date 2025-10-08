# ChromaDB-based vector store implementation
# Modern vector database with persistent storage and advanced querying

import os
import json
import chromadb
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Tuple

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# CONFIG
PROCESSED_DIR = Path("processed")
CHROMA_DIR = Path("chroma_db")  # Persistent storage directory
COLLECTION_NAME = "processed_docs"
EMBEDDING_MODEL = "text-embedding-3-large"
TOP_K = 5  # Number of results to return


def get_chroma_client():
    """Get ChromaDB client with persistent storage."""
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_embedding(text: str) -> List[float]:
    """Get embedding from OpenAI API."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def load_processed_documents() -> List[Dict]:
    """Load all processed JSONL documents."""
    documents = []
    for jsonl_file in PROCESSED_DIR.glob("*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    documents.append(json.loads(line))
    return documents


def create_vector_store():
    """
    Create ChromaDB vector store from processed documents.
    Stores embeddings and metadata in a persistent ChromaDB collection.
    """
    print("Loading processed documents...")
    documents = load_processed_documents()

    if not documents:
        print("No documents found in processed directory!")
        return

    # Initialize ChromaDB
    chroma_client = get_chroma_client()

    # Delete existing collection if it exists
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except:
        pass

    # Create new collection
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Processed documents with embeddings"}
    )

    # Prepare data for batch insertion
    ids = []
    texts = []
    embeddings = []
    metadatas = []

    for i, doc in enumerate(documents):
        if i % 10 == 0:
            print(f"  Processing {i}/{len(documents)}...")

        text = doc["text"]
        doc_id = f"doc_{i}"

        ids.append(doc_id)
        texts.append(text)
        embeddings.append(get_embedding(text))
        metadatas.append(doc["metadata"])

    # Batch insert to ChromaDB
    print(f"Inserting {len(ids)} documents into ChromaDB...")
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"✓ Vector store created with {len(ids)} documents")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Storage: {CHROMA_DIR}")


def search_vector_store(query: str, top_k: int = TOP_K) -> List[Tuple[str, Dict, float]]:
    """
    Search the ChromaDB vector store for relevant documents.
    Returns list of (text, metadata, similarity_score) tuples.
    """
    # Initialize ChromaDB
    chroma_client = get_chroma_client()

    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except:
        raise FileNotFoundError(f"Collection '{COLLECTION_NAME}' not found. Run create_vector_store() first.")

    # Get query embedding
    query_embedding = get_embedding(query)

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Convert results to our format
    # ChromaDB returns distances (lower is better), we convert to similarity
    output = []
    if results['ids'] and len(results['ids'][0]) > 0:
        for i in range(len(results['ids'][0])):
            text = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            # Convert L2 distance to similarity score (inverse relationship)
            # For L2 distance, smaller is better, so we invert it
            similarity = 1.0 / (1.0 + distance)
            output.append((text, metadata, similarity))

    return output


def query_with_context(question: str) -> tuple[str, float]:
    """
    Answer a question using the ChromaDB vector store for context.
    Returns (answer, total_latency_in_seconds).
    Latency includes vector search + API call time.
    """
    import time
    start_time = time.time()

    # Get relevant context
    results = search_vector_store(question, top_k=5)

    print(f"Found {len(results)} relevant chunks:")
    for i, (text, meta, score) in enumerate(results, 1):
        print(f"  {i}. {meta['filename']} (similarity: {score:.3f})")

    # Build context from results
    context = "\n\n".join([text for text, _, _ in results])

    # Simplified prompt
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer in 1-2 sentences."

    # Get response from OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer concisely in 1-2 sentences."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    latency = time.time() - start_time
    return response.choices[0].message.content, latency


def get_collection_info():
    """Get information about the ChromaDB collection."""
    chroma_client = get_chroma_client()
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        count = collection.count()
        print(f"Collection: {COLLECTION_NAME}")
        print(f"Document count: {count}")
        print(f"Storage location: {CHROMA_DIR}")
    except Exception as e:
        print(f"Collection not found: {e}")


if __name__ == "__main__":
    import sys
    import csv

    if len(sys.argv) > 1:
        arg = sys.argv[1]

        # Check if it's info command
        if arg == "info":
            get_collection_info()
        # Check if it's a file path
        elif arg.endswith('.txt') and Path(arg).exists():
            # Read questions from file
            print(f"Reading questions from {arg}...\n")
            with open(arg, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]

            print(f"Found {len(questions)} questions\n")
            print("=" * 80)

            # Open CSV file for writing
            log_file = "chroma_query_results.csv"
            with open(log_file, 'w', encoding='utf-8', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(["Question", "Answer", "Latency (s)"])

                for i, question in enumerate(questions, 1):
                    print(f"\nQuestion {i}/{len(questions)}: {question}")
                    answer, latency = query_with_context(question)
                    print(f"\nAnswer: {answer}")
                    print(f"Latency: {latency:.3f}s")
                    print("=" * 80)

                    # Write to CSV file
                    csv_writer.writerow([question, answer, f"{latency:.3f}"])

            print(f"\nResults saved to {log_file}")
        else:
            # Single question mode
            question = " ".join(sys.argv[1:])
            answer, latency = query_with_context(question)
            print(f"\nAnswer: {answer}")
            print(f"Latency: {latency:.3f}s")
    else:
        # Create vector store
        create_vector_store()