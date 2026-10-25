"""Ask a question and find the 3 most similar chunks in Pinecone.

Steps:
    1. Embed the question with nomic-embed-text running locally in Ollama.
    2. Send that embedding to Pinecone to find the top 3 most similar chunks.
    3. Print each chunk with its similarity score.

Usage:
    python retrieval/search_chunks.py
"""

import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"
TOP_K = 3


def embed_question(question: str) -> list[float]:
    # nomic-embed-text needs the "search_query: " prefix for questions
    # (the chunks were embedded with "search_document: ").
    payload = json.dumps({"model": MODEL, "input": "search_query: " + question}).encode("utf-8")
    request = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.load(response)["embeddings"][0]


def main() -> None:
    # Read the Pinecone settings from .env and connect to the index.
    load_dotenv(PROJECT_ROOT / ".env")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(host=os.environ["PINECONE_HOST"])

    # Step 1: embed the user's question.
    question = input("Ask a question: ")
    question_embedding = embed_question(question)

    # Step 2: ask Pinecone for the chunks whose embeddings are closest to the question's.
    results = index.query(
        vector=question_embedding,
        top_k=TOP_K,
        namespace=os.environ["PINECONE_NAMESPACE"],
        include_metadata=True,  # return the chunk text and metadata, not just ids
    )

    # Step 3: print each chunk with its similarity score (cosine: higher = more similar).
    for rank, match in enumerate(results["matches"], start=1):
        print(f"\n#{rank}  score: {match['score']:.3f}  section: {match['metadata']['section']}")
        print(match["metadata"]["content"])


if __name__ == "__main__":
    main()
