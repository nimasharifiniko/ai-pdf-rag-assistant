import os
import pdfplumber


def extract_text_with_page_numbers(pdf_path: str) -> list:
    """
    Extracts text page-by-page from a PDF file.
    Returns a list of dicts containing page_number and extracted text.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ PDF file not found at path: {pdf_path}")

    pages_data = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    pages_data.append({
                        "page_number": i,
                        "text": text.strip()
                    })
        print(
            f"📄 Successfully extracted text from {len(pages_data)} page(s) using pdfplumber.")
        return pages_data
    except Exception as e:
        print(f"❌ Error reading PDF with pdfplumber: {e}")
        return []


def create_chunks_with_metadata(pages_data: list, chunk_size: int = 300, overlap: int = 50) -> list:
    """
    Splits extracted page texts into overlapping chunks while preserving page metadata.

    Args:
        pages_data: List of dicts with 'page_number' and 'text'.
        chunk_size: Target word count per chunk.
        overlap: Overlapping word count between consecutive chunks.

    Returns:
        List of dicts: [{'chunk_id': ..., 'text': ..., 'page_number': ...}]
    """
    chunks = []
    chunk_counter = 1

    for page in pages_data:
        page_num = page["page_number"]
        words = page["text"].split()

        if len(words) <= chunk_size:
            # Page text is smaller than chunk size, take as single chunk
            chunks.append({
                "chunk_id": f"p{page_num}_c{chunk_counter}",
                "text": " ".join(words),
                "page_number": page_num
            })
            chunk_counter += 1
        else:
            # Split words with sliding window overlap
            start = 0
            while start < len(words):
                end = start + chunk_size
                chunk_words = words[start:end]

                chunk_text = " ".join(chunk_words)
                chunks.append({
                    "chunk_id": f"p{page_num}_c{chunk_counter}",
                    "text": chunk_text,
                    "page_number": page_num
                })
                chunk_counter += 1

                # Move window forward by (chunk_size - overlap)
                start += (chunk_size - overlap)

    print(
        f"🧩 Processed PDF into {len(chunks)} chunk(s) (Chunk Size: {chunk_size} words, Overlap: {overlap} words).")
    return chunks


def process_pdf_file(pdf_path: str) -> list:
    """
    Full processing pipeline for a PDF: Extract pages -> Chunk with metadata.
    """
    pages_data = extract_text_with_page_numbers(pdf_path)
    if not pages_data:
        return []

    return create_chunks_with_metadata(pages_data)


# Self-testing block
if __name__ == "__main__":
    test_pdf_path = os.path.join("data", "sample.pdf")

    print("🛠️ Testing PDF Processor Module...\n")
    if os.path.exists(test_pdf_path):
        processed_chunks = process_pdf_file(test_pdf_path)

        print(f"\n✅ Total Chunks Generated: {len(processed_chunks)}\n")
        # Show first 3 chunks
        for idx, ch in enumerate(processed_chunks[:3], start=1):
            print(f"--- Chunk #{idx} (Page {ch['page_number']}) ---")
            print(f"ID: {ch['chunk_id']}")
            print(f"Text Snippet: {ch['text'][:150]}...")
            print("-" * 40)
    else:
        print(
            f"⚠️ Test skipped: Please place a sample PDF file at '{test_pdf_path}' to test.")
