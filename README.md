# RAG-Summarization (Local/Ollama + Chroma)

## Quick start
1. Create venv: `py -3.12 -m venv .venv && .\.venv\Scripts\Activate`
2. Install deps: `pip install -r requirements.txt`
3. Run local script: `python app.py --file "docs\YourDoc.pdf" --rebuild`
4. (Optional API) `uvicorn server:app --reload`

## EC2 (docker-compose)
- `docker compose up -d --build`
- `docker exec -it ollama ollama pull qwen2:7b`

## Notes
- Data stays local: Ollama + Chroma + HF embeddings
- Do NOT commit: `.venv/`, `chroma_store/`, `ollama_models/`
