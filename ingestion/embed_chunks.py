"""Embed chunk JSON files with nomic-embed-text running locally in Ollama, and save them as JSONL.

Each output line is one chunk: all of its original fields and metadata, plus the embedding
vector and the model that produced it.

Requires Ollama running locally with the model pulled:
    ollama pull nomic-embed-text

Usage:
    python ingestion/embed_chunks.py                    # embeds source/chunks/RAG demo.json
    python ingestion/embed_chunks.py path/to/file.json
    python ingestion/embed_chunks.py path/to/folder     # embeds every .json file in the folder

Output is written to source/embeddings/<chunks name>.jsonl
"""

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

# Resolve paths relative to the project root so the script works from any directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS = PROJECT_ROOT / "source" / "chunks" / "RAG demo.json"
OUTPUT_DIR = PROJECT_ROOT / "source" / "embeddings"

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"
# Number of chunks sent to Ollama per request.
BATCH_SIZE = 16
# nomic-embed-text is trained with task prefixes: documents use "search_document: ",
# and queries at retrieval time must use "search_query: " to match.
DOCUMENT_PREFIX = "search_document: "


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Send a batch of texts to Ollama and return one embedding per text, in order."""
    payload = json.dumps({"model": MODEL, "input": texts}).encode("utf-8")
    request = urllib.request.Request(OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)["embeddings"]
    except urllib.error.HTTPError as err:
        raise SystemExit(f"Ollama returned {err.code}: {err.read().decode()} "
                         f"(is the model pulled? run: ollama pull {MODEL})")
    except urllib.error.URLError as err:
        raise SystemExit(f"Could not reach Ollama at {OLLAMA_URL} ({err.reason}). Is `ollama serve` running?")


def embed_chunks(chunks: list[dict]) -> list[dict]:
    records = []
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        # Embed the contextual text (document + section header + content), not the bare content.
        embeddings = embed_texts([DOCUMENT_PREFIX + chunk["contextual_text"] for chunk in batch])
        for chunk, embedding in zip(batch, embeddings):
            records.append({**chunk, "embedding_model": MODEL, "embedding_dim": len(embedding),
                            "embedding": embedding})
        print(f"  embedded {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunk JSON file(s) with Ollama into JSONL.")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_CHUNKS,
                        help="A chunks JSON file or a folder containing chunk JSON files")
    args = parser.parse_args()

    input_path = args.input.resolve()
    if input_path.is_dir():
        chunk_paths = sorted(input_path.glob("*.json"))
    elif input_path.is_file():
        chunk_paths = [input_path]
    else:
        parser.error(f"Not found: {input_path}")

    if not chunk_paths:
        parser.error(f"No chunk JSON files found in {input_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for chunk_path in chunk_paths:
        chunks = json.loads(chunk_path.read_text(encoding="utf-8"))
        print(f"Embedding {chunk_path.name} ({len(chunks)} chunks) with {MODEL}")
        records = embed_chunks(chunks)
        output_path = OUTPUT_DIR / f"{chunk_path.stem}.jsonl"
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Embedded {chunk_path.name} -> {output_path} ({len(records)} records)")


if __name__ == "__main__":
    main()
