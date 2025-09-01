# requirements (install once):
# pip install -U langchain langchain-community langchain-ollama langchain-chroma langchain-huggingface chromadb pypdf pymupdf sentence-transformers tqdm

import os, glob, argparse, shutil
from tqdm import trange
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser

# ---- Config (edit if needed) ----
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
GEN_MODEL       = "qwen2:7b"              # you already have this pulled
PERSIST_DIR     = "chroma_store"
K               = 4                       # top-k passages for context
CHUNK_SIZE      = 1500
CHUNK_OVERLAP   = 100

def find_paths(single_file: str | None):
    if single_file:
        if not os.path.exists(single_file):
            raise SystemExit(f"[ERROR] File not found: {single_file}")
        return [single_file]
    return glob.glob("docs/**/*.pdf", recursive=True) + glob.glob("docs/**/*.txt", recursive=True)

def load_docs(paths):
    print(f"[INFO] Found {len(paths)} file(s)")
    if not paths:
        raise SystemExit("[ERROR] No PDFs/TXTs under docs/ (or file path invalid)")

    docs = []
    for p in paths:
        try:
            if p.lower().endswith(".pdf"):
                try:
                    docs.extend(PyMuPDFLoader(p).load())    # best text extractor
                except Exception:
                    docs.extend(PyPDFLoader(p).load())      # fallback
            else:
                docs.extend(TextLoader(p, encoding="utf-8").load())
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
    print(f"[INFO] Loaded {len(docs)} docs before splitting")
    return docs

def chunk_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunks")
    for i, c in enumerate(chunks[:3]):
        snippet = (c.page_content or "").strip().replace("\n", " ")
        print(f"[PREVIEW {i}]: {snippet[:200]}{'...' if len(snippet)>200 else ''}")
    return chunks

def build_or_load_index(chunks, rebuild: bool):
    # Fast local CPU embeddings (no HTTP calls)
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if rebuild and os.path.exists(PERSIST_DIR):
        print(f"[INFO] Rebuilding: removing existing index at {PERSIST_DIR}")
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)

    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)

    # Only add if empty/new
    is_empty = not (os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR))
    if is_empty:
        texts = [c.page_content for c in chunks]
        metas = [c.metadata      for c in chunks]
        BATCH = 256
        print(f"[INFO] Indexing {len(texts)} texts into {PERSIST_DIR} ...")
        for i in trange(0, len(texts), BATCH, desc="Indexing"):
            bt = texts[i:i+BATCH]
            bm = metas[i:i+BATCH]
            keep = [(t, m) for t, m in zip(bt, bm) if (t or "").strip() != ""]
            if keep:
                k_texts, k_metas = zip(*keep)
                vectordb.add_texts(texts=list(k_texts), metadatas=list(k_metas))
        print("[INFO] Index build complete.")
    else:
        print(f"[INFO] Using existing Chroma index at {PERSIST_DIR}")

    return vectordb

def make_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": K})
    llm = ChatOllama(model=GEN_MODEL, base_url=OLLAMA_BASE_URL, num_ctx=1024)
    prompt = ChatPromptTemplate.from_template(
        "Using ONLY the provided context, write a concise bullet list of key points, "
        "then list action items.\n\n"
        "Question: {question}\n\nContext:\n{context}"
    )
    def fmt(ds): return "\n\n".join(d.page_content for d in ds)
    chain = (
        {"context": retriever | fmt, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain, retriever

def main():
    parser = argparse.ArgumentParser(description="Local RAG summarizer (Ollama + Chroma)")
    parser.add_argument("--file", help="Path to a single PDF/TXT to index (optional)")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of the Chroma index")
    parser.add_argument("--q", default="Summarize the key points across my Huawei docs and list action items.",
                        help="Question to ask")
    args = parser.parse_args()

    paths = find_paths(args.file)
    docs  = load_docs(paths)
    chunks = chunk_docs(docs)
    vectordb = build_or_load_index(chunks, rebuild=args.rebuild)
    chain, retriever = make_chain(vectordb)

    # Pre-flight: ensure retrieval returns something
    sample = retriever.invoke(args.q)
    print(f"[INFO] Retriever returned {len(sample)} doc(s)")
    if not sample:
        raise SystemExit("[ERROR] Retrieval is empty. If PDFs are image-only, convert to .txt or use OCR.")

    print("\n===== ANSWER =====\n")
    print(chain.invoke(args.q))

if __name__ == "__main__":
    main()
