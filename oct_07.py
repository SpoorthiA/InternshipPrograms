import random
import time

class TraceLogger:
    def __init__(self):
        self.traces = []

    def log(self, step_name, input_data, output_data):
        log_entry = {
            "step": step_name,
            "input": input_data,
            "output": output_data,
            "timestamp": time.time()
        }
        self.traces.append(log_entry)

    def show(self):
        print("\n--- TRACE LOGS ---")
        for trace in self.traces:
            print(f"[{trace['step']}] -> {trace['output']}")
        print("------------------\n")

QUESTIONS = [
    "What is Langfuse used for?",
    "Explain Ragas in one line.",
    "What is the main goal of RAG evaluation?",
    "How do you check the faithfulness of an LLM response?",
    "Why is observability important in AI pipelines?"
]

DOCUMENTS = [
    "Langfuse is a tracing and observability tool for LLM applications.",
    "Ragas evaluates the quality of responses in retrieval-augmented generation.",
    "Faithfulness measures if the output is grounded in context.",
    "Observability ensures reliable debugging of model systems.",
    "RAG pipelines retrieve context documents and generate answers."
]

def retrieve_context(question):
    num_docs = random.randint(1, 3)
    retrieved = random.sample(DOCUMENTS, num_docs)
    return retrieved

def generate_answer(question, contexts):
    answer = ""
    if "Langfuse" in question:
        answer = "Langfuse helps observe and debug LLM pipelines."
    elif "Ragas" in question:
        answer = "Ragas provides metrics for evaluating RAG systems."
    elif "faithfulness" in question:
        answer = "Faithfulness means the output is consistent with provided data."
    elif "observability" in question:
        answer = "Observability allows tracking performance and debugging issues."
    else:
        answer = random.choice(contexts)
    return answer

def compute_faithfulness(answer, contexts):
    matches = 0
    for c in contexts:
        for word in c.split():
            if word.lower() in answer.lower():
                matches += 1
    total_words = len(answer.split()) + 1
    score = matches / total_words
    return round(score, 2)

def compute_relevance(question, answer):
    q_words = question.lower().split()
    a_words = answer.lower().split()
    overlap = len(set(q_words).intersection(a_words))
    score = overlap / (len(set(q_words)) + 1)
    return round(score, 2)

def compute_context_precision(contexts, answer):
    useful = 0
    for c in contexts:
        if any(word.lower() in answer.lower() for word in c.split()):
            useful += 1
    score = useful / len(contexts)
    return round(score, 2)

def compute_context_recall(contexts, answer):
    covered_words = 0
    total_words = 0
    for c in contexts:
        total_words += len(c.split())
        for word in c.split():
            if word.lower() in answer.lower():
                covered_words += 1
    score = covered_words / (total_words + 1)
    return round(score, 2)

class RAGEvaluator:
    def __init__(self, logger):
        self.logger = logger
        self.results = []

    def evaluate(self, question):
        contexts = retrieve_context(question)
        self.logger.log("retrieval", question, contexts)
        answer = generate_answer(question, contexts)
        self.logger.log("generation", question, answer)
        faithfulness = compute_faithfulness(answer, contexts)
        relevance = compute_relevance(question, answer)
        precision = compute_context_precision(contexts, answer)
        recall = compute_context_recall(contexts, answer)
        self.logger.log("metrics", question, {
            "faithfulness": faithfulness,
            "relevance": relevance,
            "precision": precision,
            "recall": recall
        })
        result = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "precision": precision,
            "recall": recall
        }
        self.results.append(result)

    def summary(self):
        print("\n=== Evaluation Summary ===")
        for r in self.results:
            print(f"Q: {r['question']}")
            print(f"A: {r['answer']}")
            print(f"Faithfulness: {r['faithfulness']}, Relevance: {r['relevance']}")
            print(f"Precision: {r['precision']}, Recall: {r['recall']}")
            print("-" * 50)

    def overall_stats(self):
        if not self.results:
            return
        avg_faith = sum(r['faithfulness'] for r in self.results) / len(self.results)
        avg_rel = sum(r['relevance'] for r in self.results) / len(self.results)
        avg_prec = sum(r['precision'] for r in self.results) / len(self.results)
        avg_rec = sum(r['recall'] for r in self.results) / len(self.results)
        print("\n=== Overall Metrics ===")
        print(f"Faithfulness Avg: {round(avg_faith, 2)}")
        print(f"Relevance Avg: {round(avg_rel, 2)}")
        print(f"Precision Avg: {round(avg_prec, 2)}")
        print(f"Recall Avg: {round(avg_rec, 2)}")
        print("=========================\n")

def main():
    logger = TraceLogger()
    evaluator = RAGEvaluator(logger)
    print("Starting RAG Evaluation Simulation...\n")
    for q in QUESTIONS:
        evaluator.evaluate(q)
        time.sleep(0.2)
    logger.show()
    evaluator.summary()
    evaluator.overall_stats()

if __name__ == "__main__":
    main()
