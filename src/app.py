# requirements (install once):
# pip install -U langchain langchain-community langchain-ollama langchain-chroma langchain-huggingface chromadb pypdf pymupdf sentence-transformers tqdm

import os, glob, argparse, shutil
from pathlib import Path
from tqdm import trange
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser

# Optional reranker imports (only used if --rerank is set)
try:
    from sentence_transformers import CrossEncoder
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    HAS_RERANK = True
except Exception:
    HAS_RERANK = False

# ---- Defaults (can be overridden via CLI) ----
BASE_DIR        = Path(__file__).resolve().parent
DOCS_DIR        = BASE_DIR / "docs"                    # expects files under src/docs
PERSIST_DIR     = BASE_DIR / "chroma_store"            # persisted DB lives next to app.py
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
GEN_MODEL       = "qwen2:7b"                            # already pulled

def find_paths(single_file: str | None):
    if single_file:
        p = Path(single_file)
        if not p.exists():
            raise SystemExit(f"[ERROR] File not found: {single_file}")
        return [str(p)]
    return [str(p) for p in DOCS_DIR.rglob("*.pdf")] + [str(p) for p in DOCS_DIR.rglob("*.txt")]

def load_docs(paths):
    print(f"[INFO] Found {len(paths)} file(s)")
    if not paths:
        raise SystemExit("[ERROR] No PDFs/TXTs under docs/ (or file path invalid)")

    docs = []
    for p in paths:
        try:
            if p.lower().endswith(".pdf"):
                try:
                    docs.extend(PyMuPDFLoader(p).load())    # strong extractor
                except Exception:
                    docs.extend(PyPDFLoader(p).load())      # fallback
            else:
                docs.extend(TextLoader(p, encoding="utf-8").load())
        except Exception as e:
            print(f"[WARN] Failed to load {p}: {e}")
    print(f"[INFO] Loaded {len(docs)} docs before splitting")

    # Debug: how many are empty pre-split
    empties = sum(1 for d in docs if not (d.page_content or "").strip())
    if empties:
        print(f"[WARN] {empties} doc(s) appear empty (image-only PDFs?). Consider OCR or .txt conversion.")
    return docs

def chunk_docs(docs, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Split into {len(chunks)} chunks")
    for i, c in enumerate(chunks[:3]):
        snippet = (c.page_content or "").strip().replace("\n", " ")
        print(f"[PREVIEW {i}]: {snippet[:200]}{'...' if len(snippet)>200 else ''}")

    # Debug: count empty chunks
    empty_chunks = sum(1 for c in chunks if not (c.page_content or "").strip())
    if empty_chunks:
        print(f"[WARN] {empty_chunks} / {len(chunks)} chunk(s) are empty; they won't be indexed.")
    return chunks

def _collection_count(vdb) -> int:
    try:
        return vdb._collection.count()
    except Exception:
        return -1

def build_or_load_index(chunks, rebuild: bool, persist_dir: Path, embed_model: str):
    persist_dir = str(persist_dir)  # Chroma expects str

    emb = HuggingFaceEmbeddings(
        model_name=embed_model,
        # BGE works best with normalized vectors + cosine
        encode_kwargs={"normalize_embeddings": True} if "bge" in embed_model.lower() else {}
    )

    if rebuild and os.path.exists(persist_dir):
        print(f"[INFO] Rebuilding: removing existing index at {persist_dir}")
        shutil.rmtree(persist_dir, ignore_errors=True)

    # Use explicit cosine space to match normalized embeddings
    vectordb = Chroma(
        collection_name="huawei_docs",
        persist_directory=persist_dir,
        embedding_function=emb,
        collection_metadata={"hnsw:space": "cosine"},
    )

    cnt = _collection_count(vectordb)
    need_index = cnt <= 0
    if need_index:
        texts = [c.page_content for c in chunks]
        metas = [c.metadata      for c in chunks]
        BATCH = 256
        print(f"[INFO] Indexing {len(texts)} texts into {persist_dir} ...")
        added = 0
        for i in trange(0, len(texts), BATCH, desc="Indexing"):
            bt = texts[i:i+BATCH]
            bm = metas[i:i+BATCH]
            keep = [(t, m) for t, m in zip(bt, bm) if (t or "").strip()]
            if keep:
                k_texts, k_metas = zip(*keep)
                vectordb.add_texts(texts=list(k_texts), metadatas=list(k_metas))
                added += len(k_texts)
        # Note: langchain_chroma.Chroma persists automatically when persist_directory is set.
        print(f"[INFO] Index build complete. Added {added} texts.")
    else:
        print(f"[INFO] Using existing Chroma index at {persist_dir} (count={cnt})")

    # Final sanity
    cnt_after = _collection_count(vectordb)
    print(f"[DEBUG] Chroma collection count = {cnt_after}")
    if cnt_after == 0:
        print("[ERROR] Vector store is empty even after build. Run with --rebuild and ensure docs have text (OCR if PDFs are scanned).")
        raise SystemExit(1)

    return vectordb

def make_retriever(vectordb, k: int, use_rerank: bool):
    # Simple, reliable baseline first
    base = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": max(1, k)})
    if use_rerank:
        if not HAS_RERANK:
            print("[WARN] --rerank requested but reranker deps not available; proceeding without rerank.")
            return base
        # Light, fast reranker (local CPU)
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=k)
        return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base)
    return base

def make_chain(retriever, gen_model: str, base_url: str, num_ctx: int):
    llm = ChatOllama(
        model=gen_model,
        base_url=base_url,
        num_ctx=num_ctx,
        temperature=0,      # deterministic
        num_predict=512     # concise output
    )
    prompt = ChatPromptTemplate.from_template(
        "You are a careful analyst. Use ONLY the provided context.\n"
        "If a detail is missing from the context, say so.\n\n"
        "Deliver:\n"
        "1) 5–8 bullet key points (verbatim facts), each with a bracketed citation like [S1], [S2] matching the order of retrieved chunks.\n"
        "2) 3–6 action items.\n\n"
        "Question: {question}\n\nContext:\n{context}"
    )
    def fmt(ds): return "\n\n".join((d.page_content or "") for d in ds)
    chain = (
        {"context": retriever | fmt, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain

def main():
    parser = argparse.ArgumentParser(description="Local RAG summarizer (Ollama + Chroma)")
    parser.add_argument("--file", help="Path to a single PDF/TXT to index (optional)")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of the Chroma index")
    parser.add_argument("--q", default="Summarize the key points across my Huawei docs and list action items.",
                        help="Question to ask")
    parser.add_argument("--k", type=int, default=4, help="Top-k chunks to use as context")
    parser.add_argument("--chunk-size", type=int, default=1500, help="Chunk size for splitting")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap for splitting")
    parser.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5",
                        help="HF embedding model (e.g., 'sentence-transformers/all-MiniLM-L6-v2' or 'BAAI/bge-small-en-v1.5')")
    parser.add_argument("--ctx", type=int, default=1536, help="Model context window to request")
    parser.add_argument("--rerank", action="store_true", help="Use local cross-encoder reranker for better retrieval")
    args = parser.parse_args()

    paths  = find_paths(args.file)
    docs   = load_docs(paths)
    chunks = chunk_docs(docs, args.chunk_size, args.chunk_overlap)
    vectordb = build_or_load_index(chunks, rebuild=args.rebuild, persist_dir=PERSIST_DIR, embed_model=args.embed_model)

    # Hard guard: if empty, exit early
    try:
        _cnt = _collection_count(vectordb)
        if _cnt == 0:
            raise SystemExit("[ERROR] Vector store is empty even after build. Use --rebuild and ensure docs contain text.")
    except Exception:
        pass

    retriever = make_retriever(vectordb, k=args.k, use_rerank=args.rerank)

    # Pre-flight: ensure retrieval returns something (use an easy query first)
    sanity_q = args.q or "What is cloud computing?"
    hits = retriever.invoke(sanity_q)
    print(f"[INFO] Retriever returned {len(hits)} doc(s)")
    if not hits:
        raise SystemExit("[ERROR] Retrieval is empty. If PDFs are image-only, convert to .txt or use OCR.")

    chain = make_chain(retriever, GEN_MODEL, OLLAMA_BASE_URL, num_ctx=args.ctx)

    answer = chain.invoke(args.q)
    print("\n===== ANSWER =====\n")
    print(answer if (isinstance(answer, str) and answer.strip()) else "[No text returned]")

    print("\n===== SOURCES =====")
    for i, d in enumerate(hits, 1):
        src  = d.metadata.get("source", "?")
        page = d.metadata.get("page", d.metadata.get("page_number", "?"))
        print(f"[S{i}] {os.path.basename(src)} (page {page})")

if __name__ == "__main__":
    main()
