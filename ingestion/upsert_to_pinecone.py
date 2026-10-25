"""Upsert embedded chunks from JSONL files into a Pinecone index, with their metadata.

Each chunk becomes one vector: its chunk_id is the vector id, its embedding is the vector
values, and every other field (text, section, heading path, counts, ...) is stored as metadata.
Upserting is idempotent: re-running overwrites vectors with the same chunk_id.

Reads PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_HOST and PINECONE_NAMESPACE from .env.

Usage:
    python ingestion/upsert_to_pinecone.py                     # upserts source/embeddings/RAG demo.jsonl
    python ingestion/upsert_to_pinecone.py path/to/file.jsonl
    python ingestion/upsert_to_pinecone.py path/to/folder      # upserts every .jsonl file in the folder
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

# Resolve paths relative to the project root so the script works from any directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "source" / "embeddings" / "RAG demo.jsonl"

# Number of vectors sent per upsert request (Pinecone allows up to 1000 / 2MB per request).
BATCH_SIZE = 100
# These fields are the vector itself, so they are not repeated in the metadata.
NON_METADATA_FIELDS = {"chunk_id", "embedding"}


def load_records(jsonl_path: Path) -> list[dict]:
    with jsonl_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_vector(record: dict) -> dict:
    # Pinecone metadata accepts strings, numbers, booleans and lists of strings,
    # which covers every chunk field (heading_path is a list of strings).
    metadata = {k: v for k, v in record.items() if k not in NON_METADATA_FIELDS and v is not None}
    return {"id": record["chunk_id"], "values": record["embedding"], "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Upsert embedded chunk JSONL file(s) into Pinecone.")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_EMBEDDINGS,
                        help="An embeddings JSONL file or a folder containing JSONL files")
    args = parser.parse_args()

    input_path = args.input.resolve()
    if input_path.is_dir():
        jsonl_paths = sorted(input_path.glob("*.jsonl"))
    elif input_path.is_file():
        jsonl_paths = [input_path]
    else:
        parser.error(f"Not found: {input_path}")

    if not jsonl_paths:
        parser.error(f"No JSONL files found in {input_path}")

    load_dotenv(PROJECT_ROOT / ".env")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["PINECONE_INDEX_NAME"]
    namespace = os.environ.get("PINECONE_NAMESPACE", "")
    # Connecting by host skips a lookup call; fall back to the index name if no host is set.
    host = os.environ.get("PINECONE_HOST")
    index = pc.Index(host=host) if host else pc.Index(index_name)
    index_dim = index.describe_index_stats()["dimension"]

    for jsonl_path in jsonl_paths:
        records = load_records(jsonl_path)
        for record in records:
            if len(record["embedding"]) != index_dim:
                raise SystemExit(f"{record['chunk_id']} has {len(record['embedding'])} dims, "
                                 f"but index {index_name} expects {index_dim}")

        print(f"Upserting {jsonl_path.name} ({len(records)} vectors) into {index_name}/{namespace}")
        upserted = 0
        for start in range(0, len(records), BATCH_SIZE):
            batch = [to_vector(r) for r in records[start:start + BATCH_SIZE]]
            upserted += index.upsert(vectors=batch, namespace=namespace)["upserted_count"]
            print(f"  upserted {upserted}/{len(records)} vectors")

    stats = index.describe_index_stats()
    count = stats["namespaces"].get(namespace, {}).get("vector_count", 0)
    print(f"Done. Namespace '{namespace}' now holds {count} vectors "
          f"(stats can take a few seconds to catch up).")


if __name__ == "__main__":
    main()
