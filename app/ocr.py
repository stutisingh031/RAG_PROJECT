import fitz
import pytesseract
from PIL import Image
import io


class OCRExtractor:

    def __init__(self):

        # Change this path if Tesseract is installed elsewhere
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

    def extract_text(self, pdf_path):

        document = fitz.open(pdf_path)

        full_text = ""

        for page_number in range(len(document)):

            page = document.load_page(page_number)

            # Convert page to image
            pix = page.get_pixmap(dpi=300)

            image_bytes = pix.tobytes("png")

            image = Image.open(io.BytesIO(image_bytes))

            # OCR
            page_text = pytesseract.image_to_string(image)

            full_text += page_text + "\n"

        document.close()

        return full_text