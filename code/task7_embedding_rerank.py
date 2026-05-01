import json
import os
import re
from pathlib import Path
from typing import Dict, List

import chromadb
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


EMBEDDING_API_URL = "http://127.0.0.1:8188/v1/embeddings"
EMBEDDING_MODEL = "Qwen3-Embedding-8B"
RERANKER_PATH = "/mnt/moark-models/bge-reranker-v2-m3"
DOCUMENT_PATH = "/mnt/moark-models/L1_exam/embedding_documents.txt"
OUTPUT_PATH = "/data/exam/reranking_results.json"
CHROMA_PATH = "/data/exam/chroma_task7"

QUERY = "请检索文档中与模力方舟模型部署认证、向量化和重排最相关的内容。"
TOP_K = 10
EMBEDDING_DIMENSIONS = 1024


def read_documents(path: str) -> str:
    if not Path(path).exists():
        raise FileNotFoundError(f"Document file not found: {path}")
    return Path(path).read_text(encoding="utf-8")


def split_text(text: str, min_chunks: int = 10, max_chars: int = 450) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue

        for start in range(0, len(paragraph), max_chars):
            piece = paragraph[start : start + max_chars].strip()
            if piece:
                chunks.append(piece)

    if len(chunks) < min_chunks:
        compact = re.sub(r"\s+", " ", text).strip()
        step = max(120, len(compact) // min_chunks)
        chunks = []
        for start in range(0, len(compact), step):
            piece = compact[start : start + step].strip()
            if piece:
                chunks.append(piece)
            if len(chunks) >= min_chunks:
                break

    if len(chunks) < min_chunks:
        raise RuntimeError(f"Only generated {len(chunks)} chunks, expected at least {min_chunks}")

    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    response = requests.post(
        EMBEDDING_API_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": texts,
            "dimensions": EMBEDDING_DIMENSIONS,
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    vectors = [item["embedding"] for item in payload["data"]]

    for vector in vectors:
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise RuntimeError(
                f"Embedding dimension mismatch: got {len(vector)}, expected {EMBEDDING_DIMENSIONS}"
            )

    return vectors


def build_chroma_collection(chunks: List[str], embeddings: List[List[float]]):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name="task7_embedding_documents")

    ids = [f"chunk-{i:03d}" for i in range(len(chunks))]
    metadatas = [{"source": DOCUMENT_PATH, "chunk_index": i} for i in range(len(chunks))]

    try:
        collection.delete(ids=ids)
    except Exception:
        pass

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return collection


def retrieve_top_k(collection, query: str, top_k: int = TOP_K) -> List[Dict]:
    query_embedding = embed_texts([query])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )

    candidates = []
    for rank, (doc, distance, metadata) in enumerate(
        zip(result["documents"][0], result["distances"][0], result["metadatas"][0]),
        start=1,
    ):
        candidates.append(
            {
                "rank": rank,
                "document": doc,
                "distance": distance,
                "metadata": metadata,
            }
        )
    return candidates


def load_reranker():
    if not Path(RERANKER_PATH).exists():
        raise FileNotFoundError(f"Reranker model path not found: {RERANKER_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(RERANKER_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def rerank(query: str, candidates: List[Dict]) -> List[Dict]:
    tokenizer, model = load_reranker()
    pairs = [[query, item["document"]] for item in candidates]
    inputs = tokenizer(
        pairs,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        scores = model(**inputs).logits.view(-1).float().cpu().tolist()

    reranked = []
    for item, score in zip(candidates, scores):
        enriched = dict(item)
        enriched["rerank_score"] = score
        reranked.append(enriched)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    for index, item in enumerate(reranked, start=1):
        item["rerank_rank"] = index
    return reranked


def main() -> None:
    os.makedirs(Path(OUTPUT_PATH).parent, exist_ok=True)

    print("Reading documents...")
    text = read_documents(DOCUMENT_PATH)
    chunks = split_text(text)
    print(f"chunks: {len(chunks)}")

    print("Embedding chunks with dimensions=1024...")
    embeddings = embed_texts(chunks)

    print("Writing vectors to Chroma...")
    collection = build_chroma_collection(chunks, embeddings)

    print("Retrieving Top-10...")
    top10 = retrieve_top_k(collection, QUERY, TOP_K)

    print("Reranking with bge-reranker-v2-m3...")
    reranked = rerank(QUERY, top10)

    output = {
        "query": QUERY,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "reranker_model": RERANKER_PATH,
        "document_path": DOCUMENT_PATH,
        "chunk_count": len(chunks),
        "top10_before_rerank": top10,
        "top10_after_rerank": reranked,
    }

    Path(OUTPUT_PATH).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
