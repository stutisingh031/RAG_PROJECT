from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(self):

        self.model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5"
        )

        self.dimension = self.model.get_sentence_embedding_dimension()

    def create_embedding(self, text: str):

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()

    def get_dimension(self):

        return self.dimension