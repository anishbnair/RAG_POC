"""Convert PDF files to Markdown using Docling.

Usage:
    python ingestion/pdf_to_markdown.py                 # converts source/pdf/RAG demo.pdf
    python ingestion/pdf_to_markdown.py path/to/file.pdf
    python ingestion/pdf_to_markdown.py path/to/folder  # converts every PDF in the folder

Output is written to source/markdown/<pdf name>.md
"""

import argparse
from pathlib import Path

from docling.document_converter import DocumentConverter

# Resolve paths relative to the project root so the script works from any directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PDF = PROJECT_ROOT / "source" / "pdf" / "RAG demo.pdf"
OUTPUT_DIR = PROJECT_ROOT / "source" / "markdown"


def convert_pdf(converter: DocumentConverter, pdf_path: Path, output_dir: Path) -> Path:
    # Docling parses the PDF layout (headings, paragraphs, tables, ...) into a document model.
    result = converter.convert(pdf_path)

    # Export that document model as Markdown text.
    # escape_html=False keeps "&" as "&" instead of "&amp;" (cleaner text for embeddings later).
    markdown = result.document.export_to_markdown(escape_html=False)

    output_path = output_dir / f"{pdf_path.stem}.md"
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Converted {pdf_path.name} -> {output_path} ({len(markdown)} chars)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PDF(s) to Markdown with Docling.")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_PDF,
                        help="A PDF file or a folder containing PDFs")
    args = parser.parse_args()

    input_path = args.input.resolve()
    if input_path.is_dir():
        pdf_paths = sorted(input_path.glob("*.pdf"))
    elif input_path.is_file():
        pdf_paths = [input_path]
    else:
        parser.error(f"Not found: {input_path}")

    if not pdf_paths:
        parser.error(f"No PDF files found in {input_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create the converter once; it loads the layout models, which is the slow part.
    converter = DocumentConverter()
    for pdf_path in pdf_paths:
        convert_pdf(converter, pdf_path, OUTPUT_DIR)


if __name__ == "__main__":
    main()
