from app.config import PDF_FOLDER
from app.pdf_parser import extract_text_from_pdf
from app.chunker import chunk_text
from app.embedder import Embedder
from app.sparse_embedder import SparseEmbedder
from app.vector_db import VectorDB


def main():

    pdf_path = PDF_FOLDER / "sample.pdf"

    print("Reading PDF...")

    pages = extract_text_from_pdf(pdf_path)

    print("Chunking document...")

    chunks = chunk_text(pages)

    print(f"Total Chunks: {len(chunks)}")

    embedder = Embedder()

    sparse_embedder = SparseEmbedder()

    db = VectorDB()

    print("Generating embeddings and indexing...")

    for idx, chunk in enumerate(chunks):

        dense_embedding = embedder.create_embedding(
            chunk["text"]
        )

        sparse_embedding = sparse_embedder.create_sparse_embedding(
            chunk["text"]
        )

        db.insert(

            point_id=idx,

            dense_embedding=dense_embedding,

            sparse_embedding=sparse_embedding,

            payload={

                "page": chunk["page"],

                "text": chunk["text"]

            }

        )

    print("\nDocument indexed successfully.")


if __name__ == "__main__":
    main()