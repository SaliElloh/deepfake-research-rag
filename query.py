import chromadb
from sentence_transformers import SentenceTransformer
import ollama 


# embedding
embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name='deepfake_research')


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
        options={'temperature': 0.1},
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

if __name__ == "__main__":
    print_answer("What features are used to detect audio deepfakes?")
