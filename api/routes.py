from fastapi import (
    APIRouter,
    UploadFile,
    File
)

import shutil
import uuid

from api.models import (
    QuestionRequest,
    AnswerResponse
)

from app.config import PDF_FOLDER
from app.embedder import Embedder
from app.sparse_embedder import SparseEmbedder
from app.retriever import Retriever
from app.llm import LLM
from app.vector_db import VectorDB
from app.pdf_parser import extract_text_from_pdf
from app.chunker import chunk_text


router = APIRouter()


# ----------------------------------------
# Load Models Once
# ----------------------------------------

embedder = Embedder()

sparse_embedder = SparseEmbedder()

retriever = Retriever()

llm = LLM()


# ----------------------------------------
# Upload PDF
# ----------------------------------------

@router.post("/upload")
def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):

        return {
            "message": "Only PDF files are allowed."
        }

    # Save uploaded PDF

    pdf_path = PDF_FOLDER / file.filename

    with open(pdf_path, "wb") as buffer:

        shutil.copyfileobj(file.file, buffer)

    print("PDF Saved Successfully")

    # Extract text

    pages = extract_text_from_pdf(pdf_path)

    # Chunk document

    chunks = chunk_text(pages)

    print(f"Total Chunks : {len(chunks)}")

    # Connect Vector DB

    db = VectorDB()

    # Index every chunk

    for chunk in chunks:

        dense_embedding = embedder.create_embedding(
            chunk["text"]
        )

        sparse_embedding = sparse_embedder.create_sparse_embedding(
            chunk["text"]
        )

        db.insert(

            point_id=str(uuid.uuid4()),

            dense_embedding=dense_embedding,

            sparse_embedding=sparse_embedding,

            payload={

                "page": chunk["page"],

                "text": chunk["text"],

                "source": file.filename

            }

        )

    return {

        "message": "Document indexed successfully.",

        "filename": file.filename,

        "chunks": len(chunks)

    }


# ----------------------------------------
# Chat API
# ----------------------------------------

@router.post(
    "/chat",
    response_model=AnswerResponse
)
def chat(request: QuestionRequest):

    question = request.question

    # Hybrid Search

    results = retriever.hybrid_search(

        query=question,

        top_k=3

    )

    if len(results) == 0:

        return AnswerResponse(

            answer="No relevant document found."

        )

    # Build Context

    context = "\n\n".join(

        result.payload["text"]

        for result in results

    )

    # Generate Answer

    answer = llm.answer_question(

        question=question,

        context=context

    )

    return AnswerResponse(

        answer=answer

    )