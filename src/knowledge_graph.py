import json
from pathlib import Path

import networkx as nx


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_audio(self, audio_id, language):
        """Add an audio recording to the knowledge graph."""
        self.graph.add_node(
            audio_id,
            type="audio",
            language=language,
        )

    def add_transcript_chunk(self, audio_id, chunk_id, text):
        """Add a transcript chunk and connect it to its audio."""
        chunk_node = f"{audio_id}_chunk_{chunk_id}"

        self.graph.add_node(
            chunk_node,
            type="transcript_chunk",
            text=text,
        )

        self.graph.add_edge(
            audio_id,
            chunk_node,
            relation="contains",
        )

        return chunk_node

    def add_entity(self, chunk_node, entity_text, entity_type):
        """Add an entity and connect it to a transcript chunk."""
        entity_node = f"{entity_type}:{entity_text}"

        self.graph.add_node(
            entity_node,
            type="entity",
            entity_type=entity_type,
            text=entity_text,
        )

        self.graph.add_edge(
            chunk_node,
            entity_node,
            relation="mentions",
        )

    def save_graph(self, output_path):
        """Save the knowledge graph as JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        graph_data = nx.node_link_data(self.graph)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(
                graph_data,
                file,
                ensure_ascii=False,
                indent=4,
            )

    def print_graph_summary(self):
        """Print basic graph statistics."""
        print("Knowledge Graph Summary")
        print("-----------------------")
        print("Nodes:", self.graph.number_of_nodes())
        print("Edges:", self.graph.number_of_edges())


if __name__ == "__main__":
    transcript_path = "data/processed/audio_01_processed.json"
    entities_path = "data/processed/audio_01_entities.json"
    output_path = "data/processed/knowledge_graph.json"

    with open(transcript_path, "r", encoding="utf-8") as file:
        transcript_data = json.load(file)

    with open(entities_path, "r", encoding="utf-8") as file:
        entities = json.load(file)

    knowledge_graph = KnowledgeGraph()

    audio_id = "audio_01"

    knowledge_graph.add_audio(
        audio_id,
        transcript_data["language"],
    )

    for chunk_index, chunk in enumerate(transcript_data["chunks"]):
        chunk_node = knowledge_graph.add_transcript_chunk(
            audio_id,
            chunk_index,
            chunk["text"],
        )

        for entity in entities:
            entity_text = entity["text"]

            if entity_text.lower() in chunk["text"].lower():
                knowledge_graph.add_entity(
                    chunk_node,
                    entity_text,
                    entity["label"],
                )

    knowledge_graph.save_graph(output_path)

    knowledge_graph.print_graph_summary()

    print(f"\nSaved to: {output_path}")