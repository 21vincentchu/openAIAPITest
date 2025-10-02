# TLDR
- API testing for calling OPENAI
- Testing with our prompts
- Testing with our vectorStore
- Testing latency of our calls
- Testing accuracy of our calls

### Currently 
= preprocessing documents into JSONL files for quick reads
- have a .NPZ vector file
- querying and responding < 3 seconds

### Future
- Use LangChain
- Use gemini-flash
- Use ChromaDB for vector storing

### To-Do's
- implement chromaDB
- configure embeddings and rerankings for chromaDB
- Create documentation

### Setup
- make an openAI platform account and grab an API key
- make a .env file 
- download requirements
- NOTE: I developed on macOS with homebrew package manager, double check the prompt_file_app.py for some pathing issues
- USE .gitignore 
    - do NOT post our documents online, keep them offline
    - do NOT post our API keys
