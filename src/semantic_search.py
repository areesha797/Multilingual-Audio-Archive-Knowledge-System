
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(
        self,
        embeddings_path="data/processed/embeddings.npy",
        metadata_path="data/processed/embeddings_metadata.json",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        # Load stored embeddings
        self.embeddings = np.load(
            embeddings_path
        ).astype("float32")

        # Load chunk metadata
        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.metadata = json.load(file)

        # Load multilingual embedding model
        self.model = SentenceTransformer(
            model_name
        )

        # Create FAISS index
        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        # Add normalized embeddings to FAISS
        self.index.add(
            self.embeddings
        )

    def search(self, query, top_k=3):
        """Find transcript chunks using FAISS semantic search."""

        # Convert query into normalized embedding
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype("float32")

        # Search FAISS index
        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            # FAISS uses -1 when no result exists
            if index == -1:
                continue

            results.append(
                {
                    "chunk_id": self.metadata[index]["chunk_id"],
                    "text": self.metadata[index]["text"],
                    "start": self.metadata[index]["start"],
                    "end": self.metadata[index]["end"],
                    "similarity": float(score),
                }
            )

        return results


if __name__ == "__main__":

    search_engine = SemanticSearch()

    query = "What is this audio about?"

    results = search_engine.search(
        query
    )

    print("Search Query:")
    print(query)

    print("\nSearch Results:")

    for result in results:

        print(
            f"\nChunk ID: "
            f"{result['chunk_id']}"
        )

        print(
            f"Similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            f"Timestamp: "
            f"{result['start']:.2f}s - "
            f"{result['end']:.2f}s"
        )

        print(
            f"Text: "
            f"{result['text']}"
        )
