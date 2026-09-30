from fastembed import TextEmbedding

_model = None


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _model


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    model = _get_model()
    embeddings = model.embed(texts)
    return [e.tolist() for e in embeddings]


def generate_single_embedding(text: str) -> list[float]:
    results = generate_embeddings([text])
    return results[0] if results else []


if __name__ == "__main__":
    sample = [
        "AI in education and learning recommendations.",
        "How do algorithms process vector embeddings?"
    ]
    vectors = generate_embeddings(sample)
    print(f"Generated {len(vectors)} vectors | Dimension: {len(vectors[0])}")
