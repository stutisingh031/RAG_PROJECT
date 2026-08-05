from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    HnswConfigDiff,
    PointStruct,
    SparseVector,
)

from app.config import (
    QDRANT_URL,
    COLLECTION_NAME,
    EMBEDDING_DIMENSION,
)


class VectorDB:

    def __init__(self):

        self.client = QdrantClient(
            url=QDRANT_URL
        )

        self.collection_name = COLLECTION_NAME

        self.create_collection()

    # --------------------------------------------------
    # Create Collection
    # --------------------------------------------------

    def create_collection(self):

        collections = self.client.get_collections().collections

        collection_names = [
            collection.name
            for collection in collections
        ]

        if self.collection_name in collection_names:

            print("Collection already exists.")
            return

        self.client.create_collection(

            collection_name=self.collection_name,

            vectors_config={

                "dense": VectorParams(

                    size=EMBEDDING_DIMENSION,

                    distance=Distance.COSINE

                )

            },

            sparse_vectors_config={

                "sparse": SparseVectorParams()

            },

            hnsw_config=HnswConfigDiff(

                m=16,

                ef_construct=100

            )

        )

        print("Collection created successfully.")

    # --------------------------------------------------
    # Delete Collection
    # --------------------------------------------------

    def delete_collection(self):

        collections = self.client.get_collections().collections

        collection_names = [
            collection.name
            for collection in collections
        ]

        if self.collection_name in collection_names:

            self.client.delete_collection(

                collection_name=self.collection_name

            )

            print("Collection deleted.")

    # --------------------------------------------------
    # Insert Document
    # --------------------------------------------------

    def insert(

        self,

        point_id,

        dense_embedding,

        sparse_embedding,

        payload

    ):

        point = PointStruct(

            id=point_id,

            vector={

                "dense": dense_embedding,

                "sparse": {

                    "indices": sparse_embedding.indices.tolist(),

                    "values": sparse_embedding.values.tolist()

                }

            },

            payload=payload

        )

        self.client.upsert(

            collection_name=self.collection_name,

            points=[point]

        )

    # --------------------------------------------------
    # Dense Search
    # --------------------------------------------------

    def search_dense(

        self,

        dense_embedding,

        top_k=3

    ):

        response = self.client.query_points(

            collection_name=self.collection_name,

            using="dense",

            query=dense_embedding,

            limit=top_k

        )

        return response.points

    # --------------------------------------------------
    # Sparse Search
    # --------------------------------------------------

    def search_sparse(

        self,

        sparse_embedding,

        top_k=3

    ):

        sparse_query = SparseVector(

            indices=sparse_embedding.indices.tolist(),

            values=sparse_embedding.values.tolist()

        )

        response = self.client.query_points(

            collection_name=self.collection_name,

            using="sparse",

            query=sparse_query,

            limit=top_k

        )

        return response.points