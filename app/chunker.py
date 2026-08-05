from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(pages, chunk_size=500, chunk_overlap=100):
    """
    Split extracted PDF text into overlapping chunks.

    Args:
        pages (list): List of page dictionaries.
        chunk_size (int): Maximum characters per chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = []

    for page in pages:

        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:

            chunks.append(
                {
                    "page": page["page"],
                    "text": chunk
                }
            )

    return chunks