# Local NPZ-based vector store implementation
# Alternative to OpenAI's vector store using numpy arrays

import os
import json
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Tuple

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# CONFIG
PROCESSED_DIR = Path("processed")
NPZ_FILE = Path("vector_store.npz")
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 5  # Number of results to return


def load_processed_documents() -> List[Dict]:
    """Load all processed JSON documents."""
    documents = []
    for json_file in PROCESSED_DIR.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            documents.extend(chunks)
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

    if not documents:
        print("No documents found in processed/ directory")
        return

    print(f"Found {len(documents)} chunks")
    print("Generating embeddings...")

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
    print(f"  Shape: {embeddings_array.shape}")
    print(f"  Size: {NPZ_FILE.stat().st_size / 1024:.2f} KB")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_vector_store(query: str, top_k: int = TOP_K) -> List[Tuple[str, Dict, float]]:
    """
    Search the NPZ vector store for relevant documents.
    Returns list of (text, metadata, similarity_score) tuples.
    """
    if not NPZ_FILE.exists():
        raise FileNotFoundError(f"Vector store not found: {NPZ_FILE}")

    # Load vector store
    data = np.load(NPZ_FILE, allow_pickle=True)
    embeddings = data["embeddings"]
    texts = data["texts"]
    metadata = json.loads(str(data["metadata"]))

    # Get query embedding
    query_embedding = get_embedding(query)

    # Calculate similarities
    similarities = []
    for i, emb in enumerate(embeddings):
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((i, sim))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top k results
    results = []
    for idx, score in similarities[:top_k]:
        results.append((texts[idx], metadata[idx], score))

    return results


def query_with_context(question: str) -> tuple[str, float]:
    """
    Answer a question using the NPZ vector store for context.
    Returns (answer, latency_in_seconds).
    """
    import time
    start_time = time.time()

    print(f"\nSearching vector store for: {question}")

    # Get relevant context
    results = search_vector_store(question)

    print(f"Found {len(results)} relevant chunks:")
    for i, (text, meta, score) in enumerate(results, 1):
        print(f"  {i}. {meta['filename']} (similarity: {score:.3f})")

    # Build context from results
    context = "\n\n---\n\n".join([text for text, _, _ in results])

    # Create prompt with context
    prompt = f"""Based on the context below, answer the question in 1-2 brief sentences.

Context:
{context}

Question: {question}"""

    # Get response from OpenAI
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "developer", "content": "You are a cleanroom lab assistant. Always answer in 1-2 sentences maximum. Be direct and concise."},
            {"role": "user", "content": prompt}
        ],
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

            for i, question in enumerate(questions, 1):
                print(f"\nQuestion {i}/{len(questions)}: {question}")
                answer, latency = query_with_context(question)
                print(f"\nAnswer: {answer}")
                print(f"Latency: {latency:.3f}s")
                print("=" * 80)
        else:
            # Single question mode
            question = " ".join(sys.argv[1:])
            answer, latency = query_with_context(question)
            print(f"\nAnswer: {answer}")
            print(f"Latency: {latency:.3f}s")
    else:
        # Create vector store
        create_vector_store()
