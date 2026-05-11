from pathlib import Path
import pytesseract
from PIL import Image


class OCRTool:
    """
    Lit le texte présent dans une image.
    Nécessite Tesseract OCR installé sur Windows.
    """

    def extract_text(self, image_path: str) -> dict:
        path = Path(image_path)

        if not path.exists():
            return {
                "success": False,
                "error": f"Image not found: {image_path}",
                "text": ""
            }

        try:
            image = Image.open(path)
            text = pytesseract.image_to_string(image, lang="eng+fra")

            return {
                "success": True,
                "text": text.strip()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "text": ""
            }