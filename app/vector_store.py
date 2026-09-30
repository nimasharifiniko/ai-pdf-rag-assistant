import os
import sys
import chromadb

CHROMA_PATH = os.path.join("data", "chroma_db")


def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_or_create_collection(collection_name: str = "pdf_collection"):
    client = get_chroma_client()
    return client.get_or_create_collection(name=collection_name)


def reset_collection(collection_name: str = "pdf_collection"):
    client = get_chroma_client()
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    return client.get_or_create_collection(name=collection_name)


def add_chunks_to_store(chunks: list[dict], embeddings: list[list[float]]):
    if not chunks or not embeddings:
        return

    collection = reset_collection()

    ids = [chunk["chunk_id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [{"page_number": chunk["page_number"]} for chunk in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )


def search_similar_chunks(query_embedding: list[float], top_k: int = 3) -> list[dict]:
    collection = get_or_create_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    retrieved = []
    if results and results["documents"]:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0] if "distances" in results else [
            0] * len(docs)

        for doc, meta, dist in zip(docs, metas, distances):
            retrieved.append({
                "text": doc,
                "page_number": meta.get("page_number", 0),
                "distance": dist
            })

    return retrieved


if __name__ == "__main__":
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from app.pdf_processor import process_pdf_file
    from app.embeddings import generate_embeddings, generate_single_embedding

    sample_pdf = os.path.join("data", "sample.pdf")

    if os.path.exists(sample_pdf):
        chunks = process_pdf_file(sample_pdf)[:5]
        texts = [c["text"] for c in chunks]
        embeddings = generate_embeddings(texts)

        add_chunks_to_store(chunks, embeddings)
        print("Indexed 5 chunks into ChromaDB successfully.")

        query = "What is artificial intelligence?"
        q_embed = generate_single_embedding(query)
        matches = search_similar_chunks(q_embed, top_k=2)

        print(f"\nQuery: '{query}'")
        for i, match in enumerate(matches, 1):
            print(
                f"Match #{i} (Page {match['page_number']}): {match['text'][:100]}...")
