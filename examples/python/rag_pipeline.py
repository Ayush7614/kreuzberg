"""
RAG Pipeline Example (Kreuzberg + ONNX + FAISS + Ollama)
------------------------------------------------------

SETUP

1. Create virtual environment
   python3 -m venv venv-rag
   source venv-rag/bin/activate

2. Install dependencies
   pip install --upgrade pip
   pip install -e packages/python
   pip install onnxruntime==1.22.0 faiss-cpu transformers ollama numpy

3. Verify installation
   python - <<EOF
   import kreuzberg, onnxruntime
   print("Kreuzberg:", kreuzberg.__version__)
   print("ONNX:", onnxruntime.__version__)
   EOF

4. Install & start Ollama
   brew install ollama
   ollama serve

5. Pull LLM model
   ollama pull mistral:7b

6. Download embedding model (ONNX)
   mkdir -p models
   cd models
   curl -L -o embed.onnx \
   https://huggingface.co/optimum/all-MiniLM-L6-v2/resolve/main/model.onnx
   cd ..

------------------------------------------------------

RUN

python rag_pipeline.py \
 --file ../../test_documents/extraction_test.docx \
 --llm mistral:7b \
 --onnx-model ../../models/embed.onnx

------------------------------------------------------

WHAT THIS DOES

• Uses Kreuzberg for document extraction  
• Uses Kreuzberg built-in chunking  
• Uses ONNX Runtime for embeddings  
• Stores vectors in FAISS  
• Sends retrieved context to Ollama LLM  

NOTE

Kreuzberg exposes EmbeddingConfig, but the Python binding
does NOT expose runtime embedding execution yet.
Therefore embeddings are executed directly using ONNX
(the same backend Kreuzberg uses internally).

------------------------------------------------------
"""

import argparse
import asyncio
import numpy as np
import faiss
import ollama
import onnxruntime as ort
from transformers import AutoTokenizer

from kreuzberg import (
    extract_file,
    ExtractionConfig,
    ChunkingConfig
)

# ---------------- CONFIG ----------------

TOP_K = 3
TOKENIZER = "sentence-transformers/all-MiniLM-L6-v2"

# --------------------------------------


class Embedder:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

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


def search(query, embedder, index, chunks, k):
    q_vec = embedder.embed([query]).astype("float32")
    _, I = index.search(q_vec, k)
    return [chunks[i] for i in I[0]]


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
    parser.add_argument("--llm", default="mistral:7b")
    parser.add_argument("--onnx-model", required=True)
    args = parser.parse_args()

    print("\n🔹 Extracting document (Kreuzberg native chunking)...")

    cfg = ExtractionConfig(
        chunking=ChunkingConfig(
            max_chars=400,
            max_overlap=80
        )
    )

    result = await extract_file(args.file, config=cfg)

    if not result.chunks:
        raise RuntimeError("No chunks returned by Kreuzberg")

    # ✅ Correct key
    chunks = [c["content"] for c in result.chunks]

    print("Chunks:", len(chunks))

    # ---------------- EMBEDDINGS ----------------

    print("\n🔹 Generating embeddings (ONNX)...")
    embedder = Embedder(args.onnx_model)
    vectors = embedder.embed(chunks)

    print("Embedding shape:", vectors.shape)

    # ---------------- VECTOR DB ----------------

    print("\n🔹 Building vector DB...")
    index = build_vector_db(vectors)

    print("\nRAG system ready ✅")

    while True:
        q = input("\nAsk question (or exit): ")
        if q.lower() == "exit":
            break

        docs = search(q, embedder, index, chunks, TOP_K)
        context = "\n\n".join(docs)

        answer = ask_llm(context, q, args.llm)
        print("\n🤖 Answer:\n", answer)


if __name__ == "__main__":
    asyncio.run(main())
