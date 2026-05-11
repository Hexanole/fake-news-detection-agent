from src.image_model_predictor import ImageFakeNewsPredictor


class MLImageModelTool:
    """
    Outil IA image basé sur TensorFlow.
    Il estime si une image ressemble aux images associées à des posts fake/real.
    """

    def __init__(self):
        self.predictor = ImageFakeNewsPredictor()

    def predict(self, image_path: str) -> dict:
        try:
            return self.predictor.predict(image_path)
        except Exception as e:
            return {
                "enabled": False,
                "message": "ml_image_model_error",
                "error": str(e),
            }