from fastembed import SparseTextEmbedding


class SparseEmbedder:

    def __init__(self):

        self.model = SparseTextEmbedding(
            model_name="Qdrant/bm25"
        )

    def create_sparse_embedding(self, text):

        embedding = next(
            self.model.embed([text])
        )

        return embedding