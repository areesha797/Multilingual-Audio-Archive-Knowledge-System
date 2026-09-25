import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class MultilingualEmbedder:
    def __init__(
        self,
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts):
        """Convert text chunks into normalized multilingual embeddings."""
        return self.model.encode(
            texts,
            normalize_embeddings=True,
        )

    def save_embeddings(self, embeddings, output_path):
        """Save embeddings as a NumPy file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(output_path, embeddings)

    def save_metadata(self, chunks, output_path):
        """Save chunk metadata for later semantic search."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = []

        for index, chunk in enumerate(chunks):
            metadata.append(
                {
                    "chunk_id": index,
                    "text": chunk["text"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                }
            )

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(
                metadata,
                file,
                ensure_ascii=False,
                indent=4,
            )


if __name__ == "__main__":
    input_path = "data/processed/audio_01_processed.json"
    embeddings_path = "data/processed/embeddings.npy"
    metadata_path = "data/processed/embeddings_metadata.json"

    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    chunks = data["chunks"]
    texts = [chunk["text"] for chunk in chunks]

    embedder = MultilingualEmbedder()

    embeddings = embedder.encode_texts(texts)

    embedder.save_embeddings(
        embeddings,
        embeddings_path,
    )

    embedder.save_metadata(
        chunks,
        metadata_path,
    )

    print("Number of chunks:", len(texts))
    print("Embedding shape:", embeddings.shape)
    print("Embedding dimension:", embeddings.shape[1])
    print(f"\nEmbeddings saved to: {embeddings_path}")
    print(f"Metadata saved to: {metadata_path}")