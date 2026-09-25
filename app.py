
import sys
from pathlib import Path

import streamlit as st

# Allow imports from the src folder
SRC_DIR = Path(__file__).parent / "src"
sys.path.append(str(SRC_DIR))

from transcription import AudioTranscriber
from preprocessing import TranscriptPreprocessor
from embeddings import MultilingualEmbedder
from semantic_search import SemanticSearch
from entity_extraction import EntityExtractor
from knowledge_graph import KnowledgeGraph


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="MKEA-Lite",
    page_icon="MKEA",
    layout="wide",
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).parent

AUDIO_DIR = BASE_DIR / "data" / "audio"
TRANSCRIPT_DIR = BASE_DIR / "data" / "transcripts"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

st.title("MKEA-Lite")
st.subheader("Multilingual Audio Archive Knowledge Engineering System")

st.write(
    "Convert audio recordings into searchable, structured knowledge "
    "using speech recognition, multilingual embeddings, entity extraction, "
    "and knowledge-graph representation."
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("System Pipeline")

st.sidebar.write(
    """
    1. Audio Upload
    2. Speech-to-Text
    3. Language Detection
    4. Transcript Processing
    5. Multilingual Embeddings
    6. Semantic Search
    7. Entity Extraction
    8. Knowledge Graph
    """
)


# ---------------------------------------------------------
# Upload audio
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an audio recording",
    type=["wav", "mp3", "m4a", "ogg", "flac"],
)


# ---------------------------------------------------------
# Process uploaded audio
# ---------------------------------------------------------

if uploaded_file is not None:

    audio_path = AUDIO_DIR / uploaded_file.name

    with open(audio_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    st.success(f"Audio uploaded: {uploaded_file.name}")

    if st.button("Process Audio"):

        # -------------------------------------------------
        # 1. Transcription
        # -------------------------------------------------

        with st.spinner("Transcribing audio with Whisper..."):

            transcriber = AudioTranscriber(model_name="tiny")

            transcription = transcriber.transcribe(
                str(audio_path)
            )

            transcript_path = (
                TRANSCRIPT_DIR
                / f"{audio_path.stem}.json"
            )

            transcriber.save_transcription(
                transcription,
                transcript_path,
            )

        # -------------------------------------------------
        # 2. Preprocessing
        # -------------------------------------------------

        with st.spinner("Processing transcript..."):

            preprocessor = TranscriptPreprocessor()

            processed_path = (
                PROCESSED_DIR
                / f"{audio_path.stem}_processed.json"
            )

            processed_data = preprocessor.process_file(
                transcript_path,
                processed_path,
            )

        # -------------------------------------------------
        # 3. Multilingual embeddings
        # -------------------------------------------------

        with st.spinner("Generating multilingual embeddings..."):

            embedder = MultilingualEmbedder()

            texts = [
                chunk["text"]
                for chunk in processed_data["chunks"]
            ]

            embeddings = embedder.encode_texts(texts)

            embeddings_path = (
                PROCESSED_DIR
                / f"{audio_path.stem}_embeddings.npy"
            )

            metadata_path = (
                PROCESSED_DIR
                / f"{audio_path.stem}_embeddings_metadata.json"
            )

            embedder.save_embeddings(
                embeddings,
                embeddings_path,
            )

            embedder.save_metadata(
                processed_data["chunks"],
                metadata_path,
            )

        # -------------------------------------------------
        # 4. Entity extraction
        # -------------------------------------------------

        with st.spinner("Extracting named entities..."):

            extractor = EntityExtractor()

            entities = extractor.extract_entities(
                processed_data["full_text"]
            )

            entities_path = (
                PROCESSED_DIR
                / f"{audio_path.stem}_entities.json"
            )

            extractor.save_entities(
                entities,
                entities_path,
            )

        # -------------------------------------------------
        # 5. Knowledge graph
        # -------------------------------------------------

        with st.spinner("Building knowledge graph..."):

            knowledge_graph = KnowledgeGraph()

            audio_id = audio_path.stem

            knowledge_graph.add_audio(
                audio_id,
                processed_data["language"],
            )

            for chunk_index, chunk in enumerate(
                processed_data["chunks"]
            ):

                chunk_node = (
                    knowledge_graph.add_transcript_chunk(
                        audio_id,
                        chunk_index,
                        chunk["text"],
                    )
                )

                for entity in entities:

                    if entity["text"].lower() in chunk["text"].lower():

                        knowledge_graph.add_entity(
                            chunk_node,
                            entity["text"],
                            entity["label"],
                        )

            graph_path = (
                PROCESSED_DIR
                / f"{audio_path.stem}_knowledge_graph.json"
            )

            knowledge_graph.save_graph(
                graph_path
            )

        # -------------------------------------------------
        # 6. Save session data
        # -------------------------------------------------

        st.session_state["processed"] = True
        st.session_state["audio_name"] = uploaded_file.name
        st.session_state["transcription"] = transcription
        st.session_state["processed_data"] = processed_data
        st.session_state["entities"] = entities
        st.session_state["graph"] = knowledge_graph
        st.session_state["embeddings_path"] = str(embeddings_path)
        st.session_state["metadata_path"] = str(metadata_path)

        st.success("Audio processing completed successfully.")


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

if st.session_state.get("processed", False):

    transcription = st.session_state["transcription"]
    processed_data = st.session_state["processed_data"]
    entities = st.session_state["entities"]
    knowledge_graph = st.session_state["graph"]

    st.divider()

    # -----------------------------------------------------
    # Overview
    # -----------------------------------------------------

    st.header("Recording Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Language",
            transcription["language"],
        )

    with col2:
        st.metric(
            "Transcript Chunks",
            len(processed_data["chunks"]),
        )

    with col3:
        st.metric(
            "Entities",
            len(entities),
        )

    with col4:
        st.metric(
            "Graph Nodes",
            knowledge_graph.graph.number_of_nodes(),
        )

    # -----------------------------------------------------
    # Transcript
    # -----------------------------------------------------

    st.header("Transcript")

    st.write(
        processed_data["full_text"]
    )

    # -----------------------------------------------------
    # Entities
    # -----------------------------------------------------

    st.header("Extracted Entities")

    if entities:

        for entity in entities:

            st.write(
                f"**{entity['text']}** "
                f"— {entity['label']}"
            )

    else:

        st.info(
            "No named entities were detected."
        )

    # -----------------------------------------------------
    # Semantic Search
    # -----------------------------------------------------

    st.header("Semantic Search")

    query = st.text_input(
        "Search the transcript using natural language"
    )

    if query:

        embeddings_path = st.session_state.get(
            "embeddings_path"
        )

        metadata_path = st.session_state.get(
            "metadata_path"
        )

        if embeddings_path and metadata_path:

            search_engine = SemanticSearch(
                embeddings_path=embeddings_path,
                metadata_path=metadata_path,
            )

            results = search_engine.search(
                query,
                top_k=3,
            )

            for result in results:

                st.markdown(
                    f"**Similarity:** "
                    f"{result['similarity']:.4f}"
                )

                st.write(
                    result["text"]
                )

                st.caption(
                    f"Timestamp: "
                    f"{result['start']:.2f}s - "
                    f"{result['end']:.2f}s"
                )

                st.divider()

        else:

            st.info(
                "Embedding index is not available."
            )

    # -----------------------------------------------------
    # Knowledge Graph
    # -----------------------------------------------------

    st.header("Knowledge Graph")

    st.write(
        f"Nodes: "
        f"{knowledge_graph.graph.number_of_nodes()}"
    )

    st.write(
        f"Relationships: "
        f"{knowledge_graph.graph.number_of_edges()}"
    )

    st.subheader("Graph Relationships")

    for source, target, data in knowledge_graph.graph.edges(
        data=True
    ):

        st.write(
            f"`{source}` "
            f"→ **{data.get('relation', 'related_to')}** → "
            f"`{target}`"
        )
