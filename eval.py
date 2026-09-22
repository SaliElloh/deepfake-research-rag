import csv, json, sys
import ollama
from query import embedder, collection
 
CONFIGS = [("base", "base-alpaca", False), ("base+rag", "base-alpaca", True),
           ("ft", "deepfake-ft", False), ("ft+rag", "deepfake-ft", True)]
 
RAG_PROMPT = """Answer the question directly and concisely, using ONLY the context provided below. Do not preface your answer with phrases like "the context shows" or "based on the provided information" — just give the answer. If the context doesn't contain enough information to answer, say so plainly — do not make up information.
 
Context:
 
{context}
 
Question: {question}"""
 
 
def retrieve(question):
    res = collection.query(query_embeddings=embedder.encode([question]).tolist(), n_results=3)
    return "\n\n---\n\n".join(res["documents"][0]), [m["source"] for m in res["metadatas"][0]]
 
 
def run():
    items = [json.loads(line) for line in open("eval_set.jsonl") if line.strip()]
    retrieved = {i["id"]: retrieve(i["question"]) for i in items}  
    with open("eval_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "config", "question", "reference_answer", "retrieval_hit", "answer", "score"])
        for name, model, use_rag in CONFIGS:  
            for i in items:
                context, sources = retrieved[i["id"]]
                prompt = RAG_PROMPT.format(context=context, question=i["question"]) if use_rag else i["question"]
                answer = ollama.generate(model=model, prompt=prompt,
                                         options={"temperature": 0.1, "seed": 42, "num_predict": 300})["response"].strip()
                w.writerow([i["id"], i["type"], name, i["question"], i["reference_answer"],
                            i["source"] in sources if use_rag else "", answer, ""])
                print(name, i["id"])
 
 
def summarize():
    rows = [r for r in csv.DictReader(open("eval_results.csv")) if r["score"].strip()]
    for name, _, _ in CONFIGS:
        ans = [int(r["score"]) for r in rows if r["config"] == name and r["type"] != "unanswerable"]
        una = [int(r["score"]) for r in rows if r["config"] == name and r["type"] == "unanswerable"]
        print(f"{name:<9} score {100 * sum(ans) / (2 * len(ans)):5.1f}%  "
              f"correct {ans.count(2)}/{len(ans)}  honest {una.count(2)}/{len(una)}")
    hits = [r["retrieval_hit"] == "True" for r in rows if r["config"] == "base+rag" and r["type"] != "unanswerable"]
    print(f"retrieval hit rate {sum(hits)}/{len(hits)}")
 
 
if __name__ == "__main__":
    {"run": run, "summarize": summarize}[sys.argv[1]]()