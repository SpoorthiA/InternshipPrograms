import random
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

class Trace:
    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.spans: List[Dict[str, Any]] = []
        self.created_at = time.time()

    def add_span(self, name: str, input_data: Any, output_data: Any, meta: Optional[Dict[str, Any]] = None):
        span = {
            "name": name,
            "input": input_data,
            "output": output_data,
            "meta": meta or {},
            "timestamp": time.time()
        }
        self.spans.append(span)

    def to_dict(self):
        return {
            "trace_id": self.trace_id,
            "created_at": self.created_at,
            "spans": self.spans
        }

class TraceLogger:
    def __init__(self):
        self.traces: Dict[str, Trace] = {}

    def start_trace(self, trace_id: Optional[str] = None) -> Trace:
        if trace_id is None:
            trace_id = f"trace_{int(time.time()*1000)}_{random.randint(0,999)}"
        t = Trace(trace_id)
        self.traces[trace_id] = t
        return t

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        return self.traces.get(trace_id)

    def list_traces(self) -> List[str]:
        return list(self.traces.keys())

    def export_traces(self) -> Dict[str, Any]:
        return {tid: t.to_dict() for tid, t in self.traces.items()}

class SimpleDataset:
    def __init__(self):
        self.items: List[Dict[str, Any]] = []

    def add(self, question: str, ground_truth: Optional[str] = None):
        item = {
            "id": f"item_{len(self.items)+1}",
            "question": question,
            "ground_truth": ground_truth
        }
        self.items.append(item)

    def sample(self, n: int) -> List[Dict[str, Any]]:
        return random.sample(self.items, min(n, len(self.items)))

    def to_list(self):
        return self.items[:]

class Retriever:
    def __init__(self, documents: List[str]):
        self.documents = documents

    def retrieve(self, question: str, k: int = 3) -> List[str]:
        choices = list(self.documents)
        random.shuffle(choices)
        return choices[:k]

class Generator:
    def generate(self, question: str, contexts: List[str]) -> str:
        if "observe" in question.lower():
            return "Observability captures traces and helps debug model behavior."
        if "rag" in question.lower():
            return "RAG augments generation with retrieved context to reduce hallucinations."
        if "langfuse" in question.lower():
            return "Langfuse provides tracing, evaluation and prompt management for LLMs."
        if contexts:
            return f"Based on context: {contexts[0]}"
        return "I am not sure, please provide more information."

def simple_token_overlap(text_a: str, text_b: str) -> int:
    set_a = set(text_a.lower().split())
    set_b = set(text_b.lower().split())
    return len(set_a.intersection(set_b))

class Metrics:
    @staticmethod
    def faithfulness(answer: str, contexts: List[str]) -> float:
        if not contexts:
            return 0.0
        matches = sum(simple_token_overlap(answer, c) for c in contexts)
        denom = max(1, len(answer.split()))
        score = matches / denom
        return round(min(score, 1.0), 2)

    @staticmethod
    def relevance(question: str, answer: str) -> float:
        overlap = simple_token_overlap(question, answer)
        denom = max(1, len(set(question.lower().split())))
        score = overlap / denom
        return round(min(score, 1.0), 2)

    @staticmethod
    def context_precision(contexts: List[str], answer: str) -> float:
        if not contexts:
            return 0.0
        useful = sum(1 for c in contexts if simple_token_overlap(c, answer) > 0)
        return round(useful / len(contexts), 2)

    @staticmethod
    def context_recall(contexts: List[str], answer: str) -> float:
        total_words = sum(len(c.split()) for c in contexts)
        if total_words == 0:
            return 0.0
        covered = sum(sum(1 for w in c.split() if w.lower() in answer.lower()) for c in contexts)
        return round(min(covered / total_words, 1.0), 2)

class Evaluator:
    def __init__(self, retriever: Retriever, generator: Generator, logger: TraceLogger):
        self.retriever = retriever
        self.generator = generator
        self.logger = logger
        self.results: List[Dict[str, Any]] = []

    def evaluate_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        trace = self.logger.start_trace()
        q = item["question"]
        trace.add_span("receive_request", {"question": q}, None)
        contexts = self.retriever.retrieve(q)
        trace.add_span("retrieve", {"k": len(contexts)}, contexts)
        answer = self.generator.generate(q, contexts)
        trace.add_span("generate", {"question": q}, answer)
        f = Metrics.faithfulness(answer, contexts)
        r = Metrics.relevance(q, answer)
        p = Metrics.context_precision(contexts, answer)
        rec = Metrics.context_recall(contexts, answer)
        metrics = {"faithfulness": f, "relevance": r, "precision": p, "recall": rec}
        trace.add_span("metrics", None, metrics)
        res = {
            "id": item.get("id"),
            "question": q,
            "contexts": contexts,
            "answer": answer,
            "metrics": metrics,
            "trace_id": trace.trace_id
        }
        self.results.append(res)
        return res

    def evaluate_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        batch_results = []
        for it in items:
            batch_results.append(self.evaluate_item(it))
            time.sleep(0.05)
        return batch_results

    def summary(self) -> Dict[str, float]:
        if not self.results:
            return {}
        avg_f = round(sum(r["metrics"]["faithfulness"] for r in self.results) / len(self.results), 2)
        avg_r = round(sum(r["metrics"]["relevance"] for r in self.results) / len(self.results), 2)
        avg_p = round(sum(r["metrics"]["precision"] for r in self.results) / len(self.results), 2)
        avg_rec = round(sum(r["metrics"]["recall"] for r in self.results) / len(self.results), 2)
        return {
            "avg_faithfulness": avg_f,
            "avg_relevance": avg_r,
            "avg_precision": avg_p,
            "avg_recall": avg_rec
        }

def print_report(results: List[Dict[str, Any]], summary_stats: Dict[str, float]):
    print("\n=== EVALUATION REPORT ===")
    for r in results:
        print(f"ID: {r['id']}")
        print(f"Q: {r['question']}")
        print(f"A: {r['answer']}")
        ms = r["metrics"]
        print(f"Faithfulness: {ms['faithfulness']} Relevance: {ms['relevance']}")
        print(f"Precision: {ms['precision']} Recall: {ms['recall']}")
        print(f"Trace ID: {r['trace_id']}")
        print("-" * 40)
    print("=== SUMMARY ===")
    for k, v in summary_stats.items():
        print(f"{k}: {v}")
    print("=================\n")

def save_results(path: str, results: List[Dict[str, Any]], traces: Dict[str, Any], summary: Dict[str, float]):
    out = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": summary,
        "results": results,
        "traces": traces
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

def demo_run():
    docs = [
        "Langfuse captures nested traces for LLM calls and tools.",
        "RAG pipelines retrieve documents and then generate answers using them.",
        "Observability helps debugging and building evaluation datasets.",
        "Evaluation includes metrics like faithfulness and relevance.",
        "Prompt management and dataset versioning are part of LLM engineering."
    ]
    retriever = Retriever(docs)
    generator = Generator()
    logger = TraceLogger()
    evaluator = Evaluator(retriever, generator, logger)
    ds = SimpleDataset()
    ds.add("What is Langfuse and why use it?")
    ds.add("Explain RAG in simple terms.")
    ds.add("How does observability help model debugging?")
    ds.add("What metrics are useful for evaluating responses?")
    ds.add("How do you measure faithfulness?")
    items = ds.sample(5)
    results = evaluator.evaluate_batch(items)
    summary = evaluator.summary()
    print_report(results, summary)

if __name__ == "__main__":
    demo_run()