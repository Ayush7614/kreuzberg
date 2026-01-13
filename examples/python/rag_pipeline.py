"""
RAG Pipeline Example (Kreuzberg + FAISS + Ollama)

------------------------------------------------
SETUP

1. Create virtual environment
   python3 -m venv venv
   source venv/bin/activate

2. Install dependencies
   pip install -e packages/python
   pip install onnxruntime faiss-cpu transformers ollama

3. Install & start Ollama
   brew install ollama
   ollama serve

4. Pull model
   ollama pull mistral

5. Download embedding model
   mkdir models
   cd models
   curl -L -o embed.onnx \
   https://huggingface.co/optimum/all-MiniLM-L6-v2/resolve/main/model.onnx

------------------------------------------------
RUN (DEFAULT)

python rag_pipeline.py

RUN (CUSTOM FILE)

python rag_pipeline.py \
  --file test_documents/extraction_test.docx \
  --model models/embed.onnx

------------------------------------------------
"""

import argparse
import asyncio
import numpy as np
import onnxruntime as ort
import faiss
from transformers import AutoTokenizer
import ollama
from kreuzberg import extract_file

# ---------------- DEFAULT CONFIG ----------------

DEFAULT_FILE = "test_documents/extraction_test.docx"
DEFAULT_MODEL = "models/embed.onnx"
DEFAULT_TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "mistral:7b"
TOP_K = 3

# -----------------------------------------------


def chunk_text(text, size=400, overlap=80):
    chunks = []
    start = 0

    while start < len(text):
        end = start + size
        chunks.append(text[start:end].strip())
        start += size - overlap

    return chunks


class Embedder:
    def __init__(self, model_path, tokenizer_name):
        self.session = ort.InferenceSession(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def embed(self, texts):
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="np"
        )

        ort_inputs = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "token_type_ids": tokens.get(
                "token_type_ids",
                np.zeros_like(tokens["input_ids"])
            )
        }

        outputs = self.session.run(None, ort_inputs)[0]
        mask = ort_inputs["attention_mask"]

        # Mean pooling
        masked = outputs * np.expand_dims(mask, -1)
        summed = masked.sum(axis=1)
        counts = np.clip(mask.sum(axis=1, keepdims=True), 1e-9, None)
        return summed / counts


def build_vector_db(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors.astype("float32"))
    return index


def search(query, embedder, index, chunks, k=3):
    q_vec = embedder.embed([query]).astype("float32")
    _, I = index.search(q_vec, k)
    return [chunks[i] for i in I[0]]


def ask_llm(context, question):
    prompt = f"""
Answer ONLY using the context.

Context:
{context}

Question:
{question}

Answer:
"""

    res = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    return res["message"]["content"]


async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file",
        default=DEFAULT_FILE,
        help=f"Document path (default: {DEFAULT_FILE})"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"ONNX model path (default: {DEFAULT_MODEL})"
    )

    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help="HF tokenizer name"
    )

    args = parser.parse_args()

    print("🔹 Extracting document...")
    r = await extract_file(args.file)

    print("\n🔹 Chunking...")
    chunks = chunk_text(r.content)
    print(f"Chunks: {len(chunks)}")

    print("\n🔹 Generating embeddings...")
    embedder = Embedder(args.model, args.tokenizer)
    vectors = embedder.embed(chunks)

    print("\n🔹 Building vector DB...")
    index = build_vector_db(vectors)

    print("\nRAG system ready ✅")

    while True:
        q = input("\nAsk question (or exit): ")
        if q.lower() == "exit":
            break

        docs = search(q, embedder, index, chunks, TOP_K)
        context = "\n\n".join(docs)

        answer = ask_llm(context, q)
        print("\n🤖 Answer:\n", answer)


if __name__ == "__main__":
    asyncio.run(main())
