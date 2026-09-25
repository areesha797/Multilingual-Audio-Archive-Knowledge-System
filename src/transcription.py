import json
from pathlib import Path

import whisper


class AudioTranscriber:
    def __init__(self, model_name="tiny"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path):
        result = self.model.transcribe(audio_path)

        return {
            "language": result["language"],
            "text": result["text"].strip(),
            "segments": result["segments"],
        }

    def save_transcription(self, result, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    audio_path = "data/audio/audio 01.ogg"
    output_path = "data/transcripts/audio_01.json"

    transcriber = AudioTranscriber()

    result = transcriber.transcribe(audio_path)
    transcriber.save_transcription(result, output_path)

    print("Detected Language:", result["language"])
    print("Transcript:")
    print(result["text"])
    print(f"\nSaved to: {output_path}")