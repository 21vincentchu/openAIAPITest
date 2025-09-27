from openai import OpenAI
from dotenv import load_dotenv
import os
import time
from datetime import datetime

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def write_to_file(question, answer, latency):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open("chat_history_filereadin.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] Latency: {latency:.3f}s\n")
        f.write(f"Question: {question}\n")
        f.write(f"Answer: {answer}\n\n")

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
    
def process_questions_from_file(filename):
    
    questions = read_questions_in(filename)

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
                model="gpt-5-mini",
                prompt={
                    "id": "pmpt_68c33ff1d7a08196b83707127f95f9900d114a3e54143e10",
                    "version": "9"
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
if __name__ == "__main__":
    file_path = "/Users/vinnychu/Code/OpenAI/openAIAPITest/test_questions1.txt"
    process_questions_from_file(file_path)
