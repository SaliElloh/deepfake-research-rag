import os 
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb
from sentence_transformers import SentenceTransformer

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
    print(doc_chunks)
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

#  Checking:

query = "What features are used to detect audio deepfakes"
query_embedding = embedder.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)


for i, (doc, meta) in enumerate(zip(results["documents"][0], results['metadatas'][0])):
    print(f"\n--- Result {i+1} (source: {meta['source']}) ---")
    print(doc[:300])
