from pathlib import Path
from PIL import Image
import imagehash


class ImageAnalysisTool:
    """
    Analyse simple de l'image :
    - existence
    - taille
    - format
    - hash perceptuel
    """

    def analyze(self, image_path: str) -> dict:
        path = Path(image_path)

        if not path.exists():
            return {
                "success": False,
                "error": f"Image not found: {image_path}"
            }

        try:
            image = Image.open(path).convert("RGB")

            width, height = image.size
            phash = str(imagehash.phash(image))
            ahash = str(imagehash.average_hash(image))

            suspicious_signals = []

            if width < 250 or height < 250:
                suspicious_signals.append("image_low_resolution")

            if width / height > 4 or height / width > 4:
                suspicious_signals.append("unusual_image_ratio")

            return {
                "success": True,
                "format": image.format,
                "width": width,
                "height": height,
                "phash": phash,
                "ahash": ahash,
                "suspicious_signals": suspicious_signals
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }