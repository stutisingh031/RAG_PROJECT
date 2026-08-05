from collections import defaultdict

from app.embedder import Embedder
from app.sparse_embedder import SparseEmbedder
from app.vector_db import VectorDB


class Retriever:

    def __init__(self):

        self.vector_db = VectorDB()

        self.embedder = Embedder()

        self.sparse_embedder = SparseEmbedder()

    # ---------------------------------
    # Dense Search
    # ---------------------------------

    def dense_search(self, query, top_k=3):

        dense_embedding = self.embedder.create_embedding(query)

        return self.vector_db.search_dense(

            dense_embedding=dense_embedding,

            top_k=top_k

        )

    # ---------------------------------
    # Sparse Search
    # ---------------------------------

    def sparse_search(self, query, top_k=3):

        sparse_embedding = self.sparse_embedder.create_sparse_embedding(query)

        return self.vector_db.search_sparse(

            sparse_embedding=sparse_embedding,

            top_k=top_k

        )

    # ---------------------------------
    # Reciprocal Rank Fusion (RRF)
    # ---------------------------------

    def reciprocal_rank_fusion(

        self,

        dense_results,

        sparse_results,

        k=60

    ):

        scores = defaultdict(float)

        documents = {}

        # Dense Ranking
        for rank, result in enumerate(dense_results, start=1):

            scores[result.id] += 1 / (k + rank)

            documents[result.id] = result

        # Sparse Ranking
        for rank, result in enumerate(sparse_results, start=1):

            scores[result.id] += 1 / (k + rank)

            documents[result.id] = result

        ranked_documents = sorted(

            scores.items(),

            key=lambda item: item[1],

            reverse=True

        )

        final_results = [

            documents[doc_id]

            for doc_id, _ in ranked_documents

        ]

        return final_results

    # ---------------------------------
    # Hybrid Search
    # ---------------------------------

    def hybrid_search(self, query, top_k=3):

        dense_results = self.dense_search(

            query=query,

            top_k=top_k

        )

        sparse_results = self.sparse_search(

            query=query,

            top_k=top_k

        )

        fused_results = self.reciprocal_rank_fusion(

            dense_results=dense_results,

            sparse_results=sparse_results

        )

        return fused_results[:top_k]