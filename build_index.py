import os 
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb
from sentence_transformers import SentenceTransformer

import ollama 

def extract_text_from_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + '\n'

    return text



corpus_dir = "./corpus"

# Splitting 
documents = []


for filename in os.listdir(corpus_dir):
    if filename.endswith(".pdf"):
        path = os.path.join(corpus_dir, filename)
        text = extract_text_from_pdf(path)
        documents.append({'text': text, "source": filename})


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)


# Chunking
chunks = []

for doc in documents:
    doc_chunks = splitter.split_text(doc["text"])
    for i, chunk_text in enumerate(doc_chunks):
        chunks.append({
            "text": chunk_text,
            "source": doc['source'],
            "chunk_id": f"{doc['source']}_{i}"
        })


# embedding
embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name='deepfake_research')

texts = [c['text'] for c in chunks]
embeddings = embedder.encode(texts, show_progress_bar=True)

collection.add(
    documents=texts,
    embeddings=embeddings.tolist(),
    ids=[c['chunk_id'] for c in chunks],
    metadatas=[{'source': c['source']} for c in chunks],
)

# for i, (doc, meta) in enumerate(zip(results["documents"][0], results['metadatas'][0])):
#     print(f"\n--- Result {i+1} (source: {meta['source']}) ---")
#     print(doc[:300])


# try ollama

def answer_question(question, n_chunks=3):
    # retrieve chunks 
    query_embedding = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results = n_chunks
    )

    retrieved_chunks = results['documents'][0]
    sources = [meta['source'] for meta in results['metadatas'][0]]
    distances = results['distances'][0]

    context = "\n\n---\n\n".join(retrieved_chunks)

    # prompt 
    
    prompt = f"""Answer the question directly and concisely, using ONLY the context provided below. Do not preface your answer with phrases like "the context shows" or "based on the provided information" — just give the answer. If the context doesn't contain enough information to answer, say so plainly — do not make up information.

    Context:

    {context}

    Question: {question}


    Answer:
    
    """

    # send to the local LLM and get a response

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{'role': 'user', 'content': prompt}],
    )

    answer = response['message']['content']

    citations = [
        {'source': src, 'distance': round(dist, 3), "excerpt": chunk[:150]} for src, dist, chunk in zip(sources, distances, retrieved_chunks)]
    

    return answer, citations



def print_answer(question, n_chunks=3):
    answer, citations = answer_question(question , n_chunks)
    print(f"Question: {question}\n")
    print(f"Answer: {answer}\n")
    print("Citations:")

    for i, c in enumerate(citations):
        print(f"  [{i+1}] {c['source']} (distance: {c['distance']})")
        print(f"      \"{c['excerpt']}...\"")
    print()

print_answer("What features are used to detect audio deepfakes?")
