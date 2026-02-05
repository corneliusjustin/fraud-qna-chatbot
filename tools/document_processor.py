import logging
import re
from pathlib import Path
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)

DATASET_DIR = Path(__file__).parent.parent / "dataset"
PDF_PATH = DATASET_DIR / "Understanding Credit Card Frauds.pdf"


def extract_pdf_pages(pdf_path: Path = PDF_PATH) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({
                "page_number": i + 1,
                "text": text,
                "source": pdf_path.name,
            })
    logger.info(f"Extracted {len(pages)} pages from {pdf_path.name}")
    return pages


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Overlap: keep the tail of the current chunk
            words = current_chunk.split()
            overlap_text = " ".join(words[-overlap // 5:]) if len(words) > overlap // 5 else current_chunk
            if len(overlap_text) > overlap:
                overlap_text = overlap_text[-overlap:]
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk = current_chunk + " " + sentence if current_chunk else sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def process_pdf(pdf_path: Path = PDF_PATH) -> list[dict]:
    pages = extract_pdf_pages(pdf_path)
    all_chunks = []
    chunk_id = 0

    for page_data in pages:
        page_chunks = chunk_text(page_data["text"])
        for i, chunk in enumerate(page_chunks):
            all_chunks.append({
                "id": f"chunk_{chunk_id}",
                "text": chunk,
                "metadata": {
                    "page_number": page_data["page_number"],
                    "chunk_index": i,
                    "source": page_data["source"],
                    "total_chunks_in_page": len(page_chunks),
                },
            })
            chunk_id += 1

    logger.info(f"Processed {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks
