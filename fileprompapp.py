from openai import OpenAI
from dotenv import load_dotenv
import os
import time
from datetime import datetime

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def write_to_file(question, answer, latency):
    """Write the conversation to a text file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open("chat_history_filereadin.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] Latency: {latency:.3f}s\n")
        f.write(f"Question: {question}\n")
        f.write(f"Answer: {answer}\n")
        f.write("-" * 80 + "\n\n")

def process_questions_from_file():
    filename = "test_questions1.txt"
    
    try:
        # Read all questions from file
        with open(filename, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"Found {len(questions)} questions in {filename}")
        print("-" * 50)
        
        # Process each question
        for i, question in enumerate(questions, 1):
            print(f"\nProcessing question {i}/{len(questions)}")
            print(f"Question: {question}")
            
            try:
                start = time.time()
                print("Generating Response... ", flush=True)
                
                # Stream the response and collect it
                response_text = ""
                print("AI: ", end="", flush=True)
                with client.responses.stream(
                    model="gpt-5-nano",
                    prompt={
                        "id": "pmpt_68c33ff1d7a08196b83707127f95f9900d114a3e54143e10",
                        "version": "4"
                    },
                    input=question,
                ) as stream:
                    for event in stream:
                        if event.type == "response.output_text.delta":
                            print(event.delta, end="", flush=True)
                            response_text += event.delta
                
                # Stop timing after response is complete
                end = time.time()
                print(f"\nTotal API Latency: {end-start:.3f} seconds")
                print("-" * 80)
                
                # Write to file
                write_to_file(question, response_text, end-start)
                
            except Exception as e:
                print(f"Error processing question: {e}")
                continue
        
        print(f"\nCompleted processing all {len(questions)} questions!")
        
    except FileNotFoundError:
        print(f"Error: {filename} not found in current directory")
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    process_questions_from_file()