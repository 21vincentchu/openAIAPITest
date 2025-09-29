# TLDR
- API testing for calling OPENAI
- Testing with our prompts
- Testing with our vectorStore
- Testing latency of our calls
- Testing accuracy of our calls

### Currently 
- working with unstructured to parse through documents and turning into JSONL
- Uploading to our OpenAI vector store

### Future
- Use LangChain
- Use gemini-flash
- Use ChromaDB for vector storing

### To-Do's
- configure data pipelineing with unstructured
- test the files for conversions
- ensure the files are created into a new vector store

### Setup
- make an openAI platform account and grab an API key
- make a .env file 
- download requirements
- NOTE: I developed on macOS with homebrew package manager, double check the prompt_file_app.py for some pathing issues
- USE .gitignore 
    - do NOT post our documents online, keep them offline
    - do NOT post our API keys
