from src.text_model_predictor import TextFakeNewsPredictor


class MLTextModelTool:
    """
    Outil IA basé sur TensorFlow.
    Il retourne une probabilité fake/real à partir du texte.
    """

    def __init__(self):
        self.predictor = TextFakeNewsPredictor()

    def predict(self, text: str) -> dict:
        try:
            return self.predictor.predict(text)
        except Exception as e:
            return {
                "enabled": False,
                "message": "ml_text_model_error",
                "error": str(e),
            }