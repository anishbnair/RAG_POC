"""Split Markdown files into section-based, contextual chunks and save them as JSON.

Each Markdown heading starts a new chunk. Every chunk carries its document title and
heading path, and `contextual_text` prefixes that context to the content, so a chunk
still makes sense on its own when it is retrieved by itself.

Usage:
    python ingestion/chunk_markdown.py                  # chunks source/markdown/RAG demo.md
    python ingestion/chunk_markdown.py path/to/file.md
    python ingestion/chunk_markdown.py path/to/folder   # chunks every .md file in the folder

Output is written to source/chunks/<md name>.json
"""

import argparse
import json
import re
from pathlib import Path

# Resolve paths relative to the project root so the script works from any directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MD = PROJECT_ROOT / "source" / "markdown" / "RAG demo.md"
OUTPUT_DIR = PROJECT_ROOT / "source" / "chunks"

# Sections longer than this are split further on paragraph boundaries.
MAX_CHUNK_CHARS = 1500
# Paragraphs with fewer words than this are treated as noise (e.g. stray text read from map images).
MIN_PARAGRAPH_WORDS = 4

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
IMAGE_PLACEHOLDER = "<!-- image -->"


def parse_sections(markdown: str) -> list[dict]:
    """Group the Markdown into sections: a heading path plus the paragraphs under it."""
    sections = []
    heading_stack: list[tuple[int, str]] = []  # (level, title) of the headings currently open
    paragraphs: list[str] = []

    def flush() -> None:
        if paragraphs:
            sections.append({"heading_path": [t for _, t in heading_stack], "paragraphs": paragraphs.copy()})
            paragraphs.clear()

    for block in re.split(r"\n\s*\n", markdown):
        block = block.strip()
        match = HEADING_RE.match(block)
        if match:
            level, title = len(match.group(1)), match.group(2)
            if paragraphs:
                flush()
                # Close headings at the same or a deeper level, then open this one.
                heading_stack[:] = [(l, t) for l, t in heading_stack if l < level]
            elif heading_stack and heading_stack[-1][0] == level:
                # The previous same-level heading had no body (e.g. a title + subtitle pair),
                # so merge the two titles instead of dropping the first one.
                title = f"{heading_stack.pop()[1]} - {title}"
            else:
                heading_stack[:] = [(l, t) for l, t in heading_stack if l < level]
            heading_stack.append((level, title))
        elif block and block != IMAGE_PLACEHOLDER and len(block.split()) >= MIN_PARAGRAPH_WORDS:
            paragraphs.append(block)

    flush()
    return sections


def split_paragraphs(paragraphs: list[str], max_chars: int) -> list[str]:
    """Pack paragraphs into pieces of at most max_chars, never splitting a paragraph."""
    pieces, current = [], ""
    for para in paragraphs:
        if current and len(current) + 2 + len(para) > max_chars:
            pieces.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        pieces.append(current)
    return pieces


def chunk_markdown(md_path: Path) -> list[dict]:
    doc_title = md_path.stem
    chunks = []
    for section in parse_sections(md_path.read_text(encoding="utf-8")):
        heading_path = section["heading_path"] or ["Introduction"]  # text before the first heading
        section_title = heading_path[-1]
        pieces = split_paragraphs(section["paragraphs"], MAX_CHUNK_CHARS)

        for part, content in enumerate(pieces, start=1):
            # The context header is what makes the chunk "contextual" when embedded.
            context = f"Document: {doc_title}\nSection: {' > '.join(heading_path)}"
            if len(pieces) > 1:
                context += f" (part {part} of {len(pieces)})"
            chunks.append({
                "chunk_id": f"{doc_title}-{len(chunks):03d}",
                "source": md_path.name,
                "section": section_title,
                "heading_path": heading_path,
                "part": part,
                "total_parts": len(pieces),
                "content": content,
                "contextual_text": f"{context}\n\n{content}",
                "char_count": len(content),
                "word_count": len(content.split()),
            })
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk Markdown file(s) by section into JSON.")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_MD,
                        help="A Markdown file or a folder containing Markdown files")
    args = parser.parse_args()

    input_path = args.input.resolve()
    if input_path.is_dir():
        md_paths = sorted(input_path.glob("*.md"))
    elif input_path.is_file():
        md_paths = [input_path]
    else:
        parser.error(f"Not found: {input_path}")

    if not md_paths:
        parser.error(f"No Markdown files found in {input_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for md_path in md_paths:
        chunks = chunk_markdown(md_path)
        output_path = OUTPUT_DIR / f"{md_path.stem}.json"
        output_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Chunked {md_path.name} -> {output_path} ({len(chunks)} chunks)")


if __name__ == "__main__":
    main()
