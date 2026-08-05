from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Folders
PDF_FOLDER = BASE_DIR / "data" / "pdfs"
JSON_FOLDER = BASE_DIR / "data" / "json"
CHUNK_FOLDER = BASE_DIR / "data" / "chunks"

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash"

# Qdrant
QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "rag_documents"

# Azure text-embedding-3-small
EMBEDDING_DIMENSION = 384