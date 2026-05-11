import argparse
from pathlib import Path

import tensorflow as tf

from src.config import BASE_DIR


class TextFakeNewsPredictor:
    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else BASE_DIR / "models" / "text_fake_news_model.keras"
        self.model = None

    def is_available(self) -> bool:
        return self.model_path.exists()

    def load(self) -> None:
        if not self.is_available():
            raise FileNotFoundError(f"Modèle introuvable : {self.model_path}")

        if self.model is None:
            self.model = tf.keras.models.load_model(self.model_path)

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            return {
                "enabled": False,
                "message": "empty_text",
            }

        if not self.is_available():
            return {
                "enabled": False,
                "message": "model_missing",
                "model_path": str(self.model_path),
            }

        self.load()

        inputs = tf.constant([text], dtype=tf.string)
        probability = float(self.model.predict(inputs, verbose=0)[0][0])

        if probability >= 0.70:
            label = "fake"
            risk = "élevé"
        elif probability >= 0.40:
            label = "uncertain"
            risk = "moyen"
        else:
            label = "real"
            risk = "faible"

        return {
            "enabled": True,
            "fake_probability": round(probability, 4),
            "real_probability": round(1.0 - probability, 4),
            "label": label,
            "risk": risk,
            "model_path": str(self.model_path),
        }


def main():
    parser = argparse.ArgumentParser(description="Test text fake news model")
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    predictor = TextFakeNewsPredictor()
    result = predictor.predict(args.text)

    print(result)


if __name__ == "__main__":
    main()