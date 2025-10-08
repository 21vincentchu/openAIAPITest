# Switch to ChromaDB for better performance
from chroma_vector_store import search_vector_store
import time
from datetime import datetime

def read_questions_in(fileName):
    try:
        with open(fileName, "r", encoding="utf-8") as file:
            questions = []
            for line in file.readlines():
                if line.strip():
                    questions.append(line.strip())
            return questions

    except Exception as e:
        print(f"error reading file: {e}")
        return []

def query_vector_store(questions_file, vector_store_file):
    # Read questions
    questions = read_questions_in(questions_file)

    print(f"Found {len(questions)} questions to process\n")
    print("=" * 80)

    # Process each question
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}/{len(questions)}: {question}")

        try:
            start = time.time()
            results = search_vector_store(question, top_k=2)
            end = time.time()

            print(f"\nResults (took {end-start:.3f} seconds):")
            print("-" * 80)

            for j, (text, metadata, score) in enumerate(results, 1):
                print(f"\nResult {j} (similarity: {score:.4f}):")
                print(f"Source: {metadata['filename']}")

                print(text)
                print("-" * 80)

        except Exception as e:
            print(f"Error processing question: {e}")
            continue

    print(f"\nCompleted processing all {len(questions)} questions!")

if __name__ == "__main__":
    questions_file = "/Users/vinnychu/Code/OpenAI/openAIAPITest/test_questions1.txt"
    vector_store_file = "vector_store.npz"
    query_vector_store(questions_file, vector_store_file)
