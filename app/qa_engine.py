import os
from openai import OpenAI
from dotenv import load_dotenv
from app.embeddings import generate_single_embedding
from app.vector_store import search_similar_chunks

load_dotenv()

_client = None


def _get_ai_client() -> OpenAI:
    global _client
    if _client is None:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        _client = OpenAI(base_url=base_url, api_key="ollama")
    return _client


RAG_SYSTEM_PROMPT = """
You are a precise AI document assistant. Your task is to answer the user's question ONLY using the provided retrieved context snippets from the document.

Strict Rules:
1. Base your answer strictly on the provided context. Do NOT use outside knowledge or hallucinate.
2. If the answer cannot be found in the context, explicitly state: "I couldn't find the answer to this question in the provided document."
3. Keep your response professional, clear, and concise.
"""


def generate_rag_response(query: str, top_k: int = 3) -> dict:
    model_name = os.getenv("AI_MODEL_NAME", "qwen2.5-coder:7b")

    # 1. Generate query embedding
    query_embedding = generate_single_embedding(query)

    # 2. Retrieve relevant context chunks from ChromaDB
    matches = search_similar_chunks(query_embedding, top_k=top_k)

    if not matches:
        return {
            "answer": "No relevant context found in the document.",
            "sources": []
        }

    # 3. Format context text and collect unique page sources
    context_str = ""
    source_pages = set()

    for i, match in enumerate(matches, start=1):
        context_str += f"\n--- Context Snippet {i} (Page {match['page_number']}) ---\n{match['text']}\n"
        source_pages.add(match["page_number"])

    user_prompt = f"RETRIEVED DOCUMENT CONTEXT:\n{context_str}\n\nUSER QUESTION: {query}"

    # 4. Generate answer via Ollama LLM
    try:
        client = _get_ai_client()
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        answer_text = response.choices[0].message.content.strip()

        return {
            "answer": answer_text,
            "sources": sorted(list(source_pages))
        }
    except Exception as e:
        return {
            "answer": f"Error generating response: {str(e)}",
            "sources": []
        }


if __name__ == "__main__":
    test_query = "What is the main topic of this document?"
    print(f"Testing RAG Engine with query: '{test_query}'\n")

    res = generate_rag_response(test_query, top_k=2)
    print("🤖 AI Answer:")
    print(res["answer"])
    print(f"\n📍 Source Pages: {res['sources']}")
