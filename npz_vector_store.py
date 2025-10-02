# Local NPZ-based vector store implementation
# Alternative to OpenAI's vector store using numpy arrays

import os
import json
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Tuple
from datetime import datetime

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# CONFIG
PROCESSED_DIR = Path("processed")
NPZ_FILE = Path("vector_store.npz")
EMBEDDING_MODEL = "text-embedding-3-large"
TOP_K = 5  # Number of results to return


def load_processed_documents() -> List[Dict]:
    """Load all processed JSONL documents."""
    documents = []
    for jsonl_file in PROCESSED_DIR.glob("*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    documents.append(json.loads(line))
    return documents


def get_embedding(text: str) -> np.ndarray:
    """Get embedding from OpenAI API."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return np.array(response.data[0].embedding)


def create_vector_store():
    """
    Create NPZ vector store from processed documents.
    Stores embeddings and metadata in a single .npz file.
    """
    print("Loading processed documents...")
    documents = load_processed_documents()

    texts = []
    embeddings = []
    metadata = []

    for i, doc in enumerate(documents):
        if i % 10 == 0:
            print(f"  Processing {i}/{len(documents)}...")

        text = doc["text"]
        texts.append(text)
        embeddings.append(get_embedding(text))
        metadata.append(doc["metadata"])

    # Convert to numpy arrays
    embeddings_array = np.array(embeddings)

    print(f"Saving to {NPZ_FILE}...")
    # Save as compressed npz
    np.savez_compressed(
        NPZ_FILE,
        embeddings=embeddings_array,
        texts=texts,
        metadata=json.dumps(metadata)
    )

    print(f"✓ Vector store created: {NPZ_FILE}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Cache for vector store data
_vector_store_cache = None

def search_vector_store(query: str, top_k: int = TOP_K) -> List[Tuple[str, Dict, float]]:
    """
    Search the NPZ vector store for relevant documents.
    Returns list of (text, metadata, similarity_score) tuples.
    """
    global _vector_store_cache

    if not NPZ_FILE.exists():
        raise FileNotFoundError(f"Vector store not found: {NPZ_FILE}")

    # Load vector store (with caching)
    if _vector_store_cache is None:
        data = np.load(NPZ_FILE, allow_pickle=True)
        _vector_store_cache = {
            "embeddings": data["embeddings"],
            "texts": data["texts"],
            "metadata": json.loads(str(data["metadata"]))
        }

    embeddings = _vector_store_cache["embeddings"]
    texts = _vector_store_cache["texts"]
    metadata = _vector_store_cache["metadata"]

    # Get query embedding
    query_embedding = get_embedding(query)

    # Vectorized cosine similarity calculation
    # Normalize query embedding
    query_norm = query_embedding / np.linalg.norm(query_embedding)

    # Normalize all embeddings
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Calculate all similarities at once
    similarities = np.dot(embeddings_norm, query_norm)

    # Get top k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # Return top k results
    results = []
    for idx in top_indices:
        results.append((texts[idx], metadata[idx], float(similarities[idx])))

    return results


def query_with_context(question: str) -> tuple[str, float]:
    """
    Answer a question using the NPZ vector store for context.
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

    # Build context from results - simplified
    context = "\n\n".join([text for text, _, _ in results])

    # Simplified prompt
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer in 1-2 sentences."

    # Get response from OpenAI - using gpt-4o-mini for speed
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        arg = sys.argv[1]

        # Check if it's a file path
        if arg.endswith('.txt') and Path(arg).exists():
            # Read questions from file
            print(f"Reading questions from {arg}...\n")
            with open(arg, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f if line.strip()]

            print(f"Found {len(questions)} questions\n")
            print("=" * 80)

            # Open log file for writing
            log_file = "npz_query_results.txt"
            with open(log_file, 'w', encoding='utf-8') as log:
                log.write(f"Question\tAnswer\tLatency (s)\n")

                for i, question in enumerate(questions, 1):
                    print(f"\nQuestion {i}/{len(questions)}: {question}")
                    answer, latency = query_with_context(question)
                    print(f"\nAnswer: {answer}")
                    print(f"Latency: {latency:.3f}s")
                    print("=" * 80)

                    # Write to log file (tab-separated for easy spreadsheet import)
                    log.write(f"{question}\t{answer}\t{latency:.3f}\n")

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
