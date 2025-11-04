import os
import uuid
from typing import List, Tuple

from dotenv import load_dotenv
from datasets import Dataset
import numpy as np

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
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


def embed_texts(texts: List[str], embedding_model: str, openai_client) -> List[List[float]]:
    vectors: List[List[float]] = []
    for t in texts:
        resp = openai_client.embeddings.create(model=embedding_model, input=t)
        vectors.append(resp.data[0].embedding)
    return vectors


def upsert_documents(collection, texts: List[str], embeddings: List[List[float]]):
    ids = [str(uuid.uuid4()) for _ in texts]
    collection.upsert(documents=texts, embeddings=embeddings, ids=ids)


def retrieve_contexts(collection, question: str, embedding_model: str, k: int, openai_client) -> List[str]:
    q_emb = embed_texts([question], embedding_model, openai_client=openai_client)[0]
    res = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "distances"])
    docs = res.get("documents", [[]])[0]
    return docs


def build_prompt(question: str, contexts: List[str]) -> str:
    context_block = "\n\n".join(f"- {c}" for c in contexts)
    return (
        "You are a precise assistant. Prefer the provided context when answering. "
        "If the context does not contain the answer, you may use well-established general knowledge, "
        "but explicitly state when you are answering beyond the provided context. Be concise.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"
    )


def generate_answer(model: str, prompt: str, openai_client) -> str:
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {e}")


def run_ragas_eval(question: str, contexts: List[str], answer: str, ground_truth: str, llm) -> dict:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    dataset = Dataset.from_dict(
        {
            "question": [question],
            "contexts": [contexts],
            "answer": [answer],
            "ground_truth": [ground_truth],
        }
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
    )
    return {k: float(v) for k, v in result.items()}


def main():
    load_dotenv()

    # OpenAI-only configuration
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        openai_api_key = openai_api_key.strip().strip('"').strip("'")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    openai_client = OpenAI(api_key=openai_api_key)
    print("Using OpenAI GPT models")

    langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    chroma_dir = os.getenv("CHROMA_DIR", ".chroma")
    top_k = int(os.getenv("TOP_K", "8"))
    max_k = int(os.getenv("MAX_K", "12"))
    chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))

    if not (langfuse_public_key and langfuse_secret_key and langfuse_host):
        raise RuntimeError("LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST are required")

    langfuse = Langfuse(public_key=langfuse_public_key, secret_key=langfuse_secret_key, host=langfuse_host)
    
    # LLM for Ragas (OpenAI only)
    ragas_llm = ChatOpenAI(model=openai_model, temperature=0)

    trace = langfuse.trace(name="minimal-rag", input={"note": "RAG demo start"})

    # Reduce noisy chromadb telemetry warnings
    os.environ["CHROMA_TELEMETRY"] = "false"

    # Ingest
    source_text = read_source_text(os.path.join("data", "source.txt"))
    docs = simple_chunk(source_text, max_chars=chunk_size)
    ingest_span = trace.span(name="ingest", input={"num_docs": len(docs)})
    collection = ensure_collection(chroma_dir, name="docs")
    embeddings = embed_texts(docs, embedding_model, openai_client=openai_client)
    upsert_documents(collection, docs, embeddings)
    ingest_span.end(output={"status": "ok"})

    # Retrieval
    question = "Summarize what Langfuse and Ragas are used for in two bullet points."
    # Use a higher k for retrieval to increase context breadth
    retrieval_k = max_k if max_k > top_k else top_k
    retrieval_span = trace.span(name="retrieval", input={"question": question, "top_k": retrieval_k})
    contexts = retrieve_contexts(
        collection,
        question,
        embedding_model,
        retrieval_k,
        openai_client=openai_client,
    )
    retrieval_span.end(output={"contexts": contexts})

    # Generation
    generation_span = trace.span(name="generation")
    prompt = build_prompt(question, contexts)
    model_name = openai_model
    answer = generate_answer(model_name, prompt, openai_client=openai_client)
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

    print("Question:\n", question)
    print("\nAnswer:\n", answer)
    print("\nRagas scores:")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()