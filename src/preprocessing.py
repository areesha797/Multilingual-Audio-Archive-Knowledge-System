import json
import re
from pathlib import Path


class TranscriptPreprocessor:
    def clean_text(self, text):
        """Clean transcript text while preserving multilingual characters."""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def create_chunks(self, segments, max_words=80):
        """Group Whisper segments into manageable text chunks."""
        chunks = []
        current_text = []
        current_start = None
        current_end = None
        current_word_count = 0

        for segment in segments:
            text = self.clean_text(segment["text"])

            if not text:
                continue

            words = text.split()

            if current_start is None:
                current_start = segment["start"]

            current_text.append(text)
            current_end = segment["end"]
            current_word_count += len(words)

            if current_word_count >= max_words:
                chunks.append(
                    {
                        "text": " ".join(current_text),
                        "start": current_start,
                        "end": current_end,
                    }
                )

                current_text = []
                current_start = None
                current_end = None
                current_word_count = 0

        if current_text:
            chunks.append(
                {
                    "text": " ".join(current_text),
                    "start": current_start,
                    "end": current_end,
                }
            )

        return chunks

    def process_file(self, input_path, output_path):
        """Load Whisper JSON, preprocess it, and save structured chunks."""
        with open(input_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        chunks = self.create_chunks(data["segments"])

        processed_data = {
            "language": data["language"],
            "full_text": self.clean_text(data["text"]),
            "chunks": chunks,
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(
                processed_data,
                file,
                ensure_ascii=False,
                indent=4,
            )

        return processed_data


if __name__ == "__main__":
    input_path = "data/transcripts/audio_01.json"
    output_path = "data/processed/audio_01_processed.json"

    preprocessor = TranscriptPreprocessor()

    result = preprocessor.process_file(
        input_path,
        output_path,
    )

    print("Language:", result["language"])
    print("Number of chunks:", len(result["chunks"]))
    print("\nProcessed Text:")
    print(result["full_text"])
    print(f"\nSaved to: {output_path}")