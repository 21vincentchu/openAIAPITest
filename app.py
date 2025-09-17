from openai import OpenAI
from dotenv import load_dotenv
import time
import os

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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
            
        # Skip empty inputs
        if not user_input:
            continue
            
        try:
            start = time.time()
            print("Generating Response... ", flush=True)
            
            # Stream the response
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
                        end = time.time()
                print(f"Total API Latency: {end-start:.3f} seconds")
                print()
                
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")
            
if __name__ == "__main__":
    chat_with_ai()