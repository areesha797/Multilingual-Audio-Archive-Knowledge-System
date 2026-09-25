import json
from pathlib import Path

import spacy


class EntityExtractor:
    def __init__(self, model_name="en_core_web_sm"):
        self.nlp = spacy.load(model_name)

    def extract_entities(self, text):
        """Extract named entities from transcript text."""
        doc = self.nlp(text)

        entities = []

        for entity in doc.ents:
            entities.append(
                {
                    "text": entity.text,
                    "label": entity.label_,
                    "description": spacy.explain(entity.label_),
                }
            )

        return entities

    def save_entities(self, entities, output_path):
        """Save extracted entities as JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(
                entities,
                file,
                ensure_ascii=False,
                indent=4,
            )


if __name__ == "__main__":
    input_path = "data/processed/audio_01_processed.json"
    output_path = "data/processed/audio_01_entities.json"

    with open(input_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    extractor = EntityExtractor()

    entities = extractor.extract_entities(data["full_text"])

    extractor.save_entities(
        entities,
        output_path,
    )

    print("Extracted Entities:")

    if entities:
        for entity in entities:
            print(
                f"- {entity['text']} "
                f"({entity['label']}: {entity['description']})"
            )
    else:
        print("No named entities detected.")

    print(f"\nSaved to: {output_path}")