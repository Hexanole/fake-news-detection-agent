import argparse
import json
from agent.pipelines.verify_news_pipeline import VerifyNewsPipeline


def main():
    parser = argparse.ArgumentParser(description="Disinfo Agent - Fake News Verification V1")

    parser.add_argument("--text", type=str, default="", help="Texte ou claim à vérifier")
    parser.add_argument("--image", type=str, default="", help="Chemin vers l'image")
    parser.add_argument("--source", type=str, default="", help="URL, domaine ou compte source")
    parser.add_argument("--json", type=str, default="", help="Fichier JSON d'entrée")

    args = parser.parse_args()

    if args.json:
        with open(args.json, "r", encoding="utf-8") as f:
            data = json.load(f)

        text = data.get("text", "")
        image = data.get("image_path", "")
        source = data.get("source", "")
    else:
        text = args.text
        image = args.image
        source = args.source

    pipeline = VerifyNewsPipeline()
    report = pipeline.verify(text=text, image_path=image, source=source)

    print(json.dumps(report, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()