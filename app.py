from openai import OpenAI
from dotenv import load_dotenv
import time
import os
from datetime import datetime

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def write_to_file(question,answer,latency):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open("chat_history.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] Latency: {latency:.3f}s\n")
        f.write(f"Question: {question}\n")
        f.write(f"Answer: {answer}\n")
        f.write("-" * 80 + "\n\n")
    

def chat_with_ai():
    print("AI Chat Session Started (type 'quit' or 'exit' to end)")
    print("-" * 50)
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check if user wants to quit
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
  
        try:
            start = time.time()
            print("Generating Response... ", flush=True)
            
            # Stream the response
            response_text = ""
            print("AI: ", end="", flush=True)
            with client.responses.stream(
                model="gpt-5-nano",
                prompt={
                    "id": "pmpt_68c33ff1d7a08196b83707127f95f9900d114a3e54143e10",
                    "version": "4"
                },
                input=user_input,
            ) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        print(event.delta, end="", flush=True)
                        response_text += event.delta
                        
                end = time.time()
                print(f"Total API Latency: {end-start:.3f} seconds")
                print()
                
                #write to file
                write_to_file(user_input,response_text,end-start)
                
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")
            
if __name__ == "__main__":
    chat_with_ai()