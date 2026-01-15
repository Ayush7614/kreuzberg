"""
RAG Pipeline (Kreuzberg Native)

Uses:
- Built-in chunking
- Built-in embeddings (FastEmbed)
- FAISS vector search
- Ollama LLM

Run:
python rag_pipeline.py --file ../../test_documents/extraction_test.docx
"""

import argparse
import asyncio
import faiss
import numpy as np
import ollama

from kreuzberg import extract_file, generate_embeddings
from kreuzberg.embeddings import EmbeddingConfig, EmbeddingModelType

TOP_K = 3


def build_vector_db(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors.astype("float32"))
    return index


def search(query_vec, index, chunks, k):
    _, ids = index.search(query_vec, k)
    return [chunks[i] for i in ids[0]]


def ask_llm(context, question, model):
    prompt = f"""
Answer ONLY using the context.

Context:
{context}

Question:
{question}

Answer:
"""

    res = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return res["message"]["content"]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--model", default="mistral:7b")
    args = parser.parse_args()

    print("\n🔹 Extracting document...")
    result = await extract_file(args.file)

    print("\n🔹 Chunking (Kreuzberg native)...")
    chunks = result.chunks
    print("Chunks:", len(chunks))

    print("\n🔹 Generating embeddings (FastEmbed)...")

    embed_config = EmbeddingConfig(
        model=EmbeddingModelType.preset("balanced"),
        normalize=True
    )

    vectors = generate_embeddings(chunks, embed_config)
    vectors = np.array(vectors)

    print("Embedding shape:", vectors.shape)

    print("\n🔹 Building vector DB...")
    index = build_vector_db(vectors)

    print("\nRAG system ready ✅")

    while True:
        q = input("\nAsk question (or exit): ")
        if q.lower() == "exit":
            break

        q_vec = generate_embeddings([q], embed_config)
        q_vec = np.array(q_vec).astype("float32")

        docs = search(q_vec, index, chunks, TOP_K)
        context = "\n\n".join(docs)

        answer = ask_llm(context, q, args.model)
        print("\n🤖 Answer:\n", answer)


if __name__ == "__main__":
    asyncio.run(main())
