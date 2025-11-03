import os
import uuid
from typing import List, Tuple

from dotenv import load_dotenv
from datasets import Dataset
import numpy as np

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import Langfuse
import chromadb


def read_source_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def simple_chunk(text: str, max_chars: int = 600) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    for p in paragraphs:
        if len(p) <= max_chars:
            chunks.append(p)
            continue
        start = 0
        while start < len(p):
            chunks.append(p[start : start + max_chars])
            start += max_chars
    return chunks


def ensure_collection(chroma_dir: str, name: str):
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        collection = client.get_collection(name)
    except Exception:
        collection = client.create_collection(name)
    return collection


def embed_texts(texts: List[str], embedding_model: str) -> List[List[float]]:
    vectors: List[List[float]] = []
    for t in texts:
        resp = genai.embed_content(model=embedding_model, content=t)
        vectors.append(resp["embedding"])  # type: ignore[index]
    return vectors


def upsert_documents(collection, texts: List[str], embeddings: List[List[float]]):
    ids = [str(uuid.uuid4()) for _ in texts]
    collection.upsert(documents=texts, embeddings=embeddings, ids=ids)


def retrieve_contexts(collection, question: str, embedding_model: str, k: int) -> List[str]:
    q_emb = embed_texts([question], embedding_model)[0]
    res = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "distances"])
    docs = res.get("documents", [[]])[0]
    return docs


def build_prompt(question: str, contexts: List[str]) -> str:
    context_block = "\n\n".join(f"- {c}" for c in contexts)
    return (
        "You are a helpful assistant. Answer the question using only the information in the context. "
        "If the answer is not contained in the context, say you don't know.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"
    )


def generate_answer(model: str, prompt: str) -> str:
    gm = genai.GenerativeModel(model)
    try:
        resp = gm.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception:
        # Fallback to a well-supported model if configured model is unavailable
        gm_fallback = genai.GenerativeModel("gemini-1.5-flash-002")
        resp = gm_fallback.generate_content(prompt)
        return (resp.text or "").strip()


def run_ragas_eval(question: str, contexts: List[str], answer: str, ground_truth: str, llm) -> dict:
    from ragas import evaluate
    from ragas.metrics import faithfulness, context_precision

    dataset = Dataset.from_dict(
        {
            "question": [question],
            "contexts": [contexts],
            "answer": [answer],
            "ground_truth": [ground_truth],
        }
    )
    result = evaluate(dataset, metrics=[faithfulness, context_precision], llm=llm)
    return {k: float(v) for k, v in result.items()}


def main():
    load_dotenv()

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    # Tolerate accidental quotes in .env
    if gemini_api_key:
        gemini_api_key = gemini_api_key.strip().strip('"').strip("'")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")

    langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    chroma_dir = os.getenv("CHROMA_DIR", ".chroma")
    top_k = int(os.getenv("TOP_K", "3"))

    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is required")
    if not (langfuse_public_key and langfuse_secret_key and langfuse_host):
        raise RuntimeError("LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST are required")

    genai.configure(api_key=gemini_api_key)
    # Ensure LangChain's Google wrapper sees the key even if env differs
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    langfuse = Langfuse(public_key=langfuse_public_key, secret_key=langfuse_secret_key, host=langfuse_host)
    # LLM for Ragas (Gemini via LangChain) with safe fallback
    try:
        ragas_llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=0)
    except Exception:
        ragas_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002", temperature=0)

    trace = langfuse.trace(name="minimal-rag", input={"note": "RAG demo start"})

    # Ingest
    source_text = read_source_text(os.path.join("data", "source.txt"))
    docs = simple_chunk(source_text)
    ingest_span = trace.span(name="ingest", input={"num_docs": len(docs)})
    collection = ensure_collection(chroma_dir, name="docs")
    embeddings = embed_texts(docs, embedding_model)
    upsert_documents(collection, docs, embeddings)
    ingest_span.end(output={"status": "ok"})

    # Retrieval
    question = "What are Langfuse and Ragas used for?"
    retrieval_span = trace.span(name="retrieval", input={"question": question, "top_k": top_k})
    contexts = retrieve_contexts(collection, question, embedding_model, top_k)
    retrieval_span.end(output={"contexts": contexts})

    # Generation
    generation_span = trace.span(name="generation")
    prompt = build_prompt(question, contexts)
    answer = generate_answer(gemini_model, prompt)
    generation_span.end(input={"prompt": prompt}, output={"answer": answer})

    # Ragas evaluation (with a tiny ground truth for demo purposes)
    ground_truth = (
        "Langfuse provides tracing and observability for LLM apps; Ragas evaluates RAG quality."
    )
    eval_span = trace.span(name="evaluation", input={"ground_truth": ground_truth})
    scores = run_ragas_eval(question, contexts, answer, ground_truth, llm=ragas_llm)
    eval_span.end(output={"scores": scores})

    trace.update(output={"answer": answer, "scores": scores})
    langfuse.flush()

    print("Answer:\n", answer)
    print("\nRagas scores:")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()



