from app.config import PDF_FOLDER
from app.pdf_parser import extract_text_from_pdf
from app.chunker import chunk_text
from app.embedder import Embedder
from app.vector_db import VectorDB
from app.retriever import Retriever
from app.llm import LLM
from fastapi import FastAPI

from api.routes import router


app = FastAPI(

    title="Hybrid RAG API",

    description="Hybrid Search using Dense + Sparse Embeddings",

    version="1.0.0"

)

app.include_router(router)


def main():

    pdf_path = PDF_FOLDER / "sample.pdf"

    pages = extract_text_from_pdf(pdf_path)

    chunks = chunk_text(pages)

    embedder = Embedder()

    db = VectorDB()

    # Store vectors
    for idx, chunk in enumerate(chunks):

        embedding = embedder.create_embedding(chunk["text"])

        db.insert(
            point_id=idx,
            embedding=embedding,
            payload={
                "page": chunk["page"],
                "text": chunk["text"]
            }
        )

    print("\nDocuments indexed successfully.")
    print("-" * 60)

    question = input("Ask a question: ")

    question_embedding = embedder.create_embedding(question)

    retriever = Retriever()

    results = retriever.search(question_embedding)

    # Create context from retrieved chunks
    context = ""

    for result in results:
        context += result.payload["text"] + "\n\n"

    # Ask Gemini
    llm = LLM()

    answer = llm.answer_question(
        question,
        context
    )

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()