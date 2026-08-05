import fitz


def extract_text_from_pdf(pdf_path):
    """
    Extract text from each page of a digital PDF.

    Returns:
        List[dict]
    """

    pages = []

    document = fitz.open(pdf_path)

    for page_number in range(len(document)):

        page = document.load_page(page_number)

        text = page.get_text("text")

        pages.append(
            {
                "page": page_number + 1,
                "text": text.strip()
            }
        )

    document.close()

    return pages