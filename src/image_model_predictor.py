import argparse
from pathlib import Path

import tensorflow as tf

from src.config import BASE_DIR

IMG_SIZE = 224


class ImageFakeNewsPredictor:
    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else BASE_DIR / "models" / "image_fake_news_model.keras"
        self.model = None

    def is_available(self) -> bool:
        return self.model_path.exists()

    def load(self):
        if not self.is_available():
            raise FileNotFoundError(f"Modèle image introuvable : {self.model_path}")

        if self.model is None:
            self.model = tf.keras.models.load_model(self.model_path)

    def preprocess(self, image_path: str):
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        image_bytes = tf.io.read_file(str(path))
        image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = tf.expand_dims(image, axis=0)

        return image

    def predict(self, image_path: str) -> dict:
        if not image_path:
            return {
                "enabled": False,
                "message": "empty_image_path",
            }

        if not self.is_available():
            return {
                "enabled": False,
                "message": "image_model_missing",
                "model_path": str(self.model_path),
            }

        try:
            self.load()
            image = self.preprocess(image_path)

            probability = float(self.model.predict(image, verbose=0)[0][0])

            if probability >= 0.70:
                label = "fake_associated_image"
                risk = "élevé"
            elif probability >= 0.40:
                label = "uncertain_image"
                risk = "moyen"
            else:
                label = "real_associated_image"
                risk = "faible"

            return {
                "enabled": True,
                "fake_probability": round(probability, 4),
                "real_probability": round(1.0 - probability, 4),
                "label": label,
                "risk": risk,
                "model_path": str(self.model_path),
            }

        except Exception as e:
            return {
                "enabled": False,
                "message": "image_prediction_error",
                "error": str(e),
            }


def main():
    parser = argparse.ArgumentParser(description="Tester modèle image fake/real")
    parser.add_argument("--image", type=str, required=True)

    args = parser.parse_args()

    predictor = ImageFakeNewsPredictor()
    result = predictor.predict(args.image)

    print(result)


if __name__ == "__main__":
    main()