import json
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

corpus_dir = "./corpus"
documents = []
for filename in os.listdir(corpus_dir):
    if filename.endswith(".pdf"):
        path = os.path.join(corpus_dir, filename)
        documents.append({"text": extract_text_from_pdf(path), "source": filename})

# Use larger chunks for dataset generation — enough context to write a real Q&A pair
splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
passages = []
for doc in documents:
    for chunk in splitter.split_text(doc["text"]):
        if len(chunk) > 300:  # skip tiny/junk fragments
            passages.append(chunk)

print(f"Generating Q&A pairs from {len(passages)} passages...")

dataset = []
for i, passage in enumerate(passages):
    prompt = f"""Based on the following research paper excerpt, write ONE clear question and its answer. The answer must be fully supported by the excerpt. Respond ONLY in this exact format, nothing else:

Q: <question>
A: <answer>

Excerpt:
{passage}"""

    response = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt}])
    text = response["message"]["content"]

    if "Q:" in text and "A:" in text:
        q = text.split("Q:")[1].split("A:")[0].strip()
        a = text.split("A:")[1].strip()
        dataset.append({"instruction": q, "output": a})

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(passages)} done")

with open("training_data.jsonl", "w") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")

print(f"\nSaved {len(dataset)} Q&A pairs to training_data.jsonl")