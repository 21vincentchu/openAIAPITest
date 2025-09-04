from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key = os.getenv('OPENAI_API_KEY')
)

response = client.responses.create(
  model="o4-mini-2025-04-16",
  input="very briefly and concisely describe python for me",
  store=True,
)

print(response.output_text)