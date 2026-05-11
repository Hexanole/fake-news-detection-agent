import json
import shutil
import time
from pathlib import Path
from datetime import datetime

from src.config import BASE_DIR
from agent.pipelines.verify_news_pipeline import VerifyNewsPipeline


class AutoRunner:
    def __init__(self):
        self.inbox_dir = BASE_DIR / "inbox"
        self.processed_dir = BASE_DIR / "processed"
        self.failed_dir = BASE_DIR / "failed"
        self.logs_dir = BASE_DIR / "logs"

        self.inbox_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        self.failed_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        self.pipeline = VerifyNewsPipeline()

    def process_file(self, file_path: Path) -> None:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            text = data.get("text", "")
            image_path = data.get("image_path", "")
            source = data.get("source", "")

            report = self.pipeline.verify(
                text=text,
                image_path=image_path,
                source=source
            )

            print("=" * 80)
            print(f"Fichier traité : {file_path.name}")
            print(f"Risque : {report['decision']['risk_level']}")
            print(f"Score : {report['decision']['risk_score']}")
            print(f"Rapport : {report['report_path']}")
            print("=" * 80)

            destination = self.processed_dir / file_path.name
            shutil.move(str(file_path), str(destination))

        except Exception as e:
            print(f"Erreur avec {file_path.name}: {e}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            failed_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            destination = self.failed_dir / failed_name

            try:
                shutil.move(str(file_path), str(destination))
            except Exception:
                pass

            self.write_log(file_path.name, str(e))

    def write_log(self, filename: str, error: str) -> None:
        log_file = self.logs_dir / "errors.log"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat(timespec='seconds')}] {filename} -> {error}\n")

    def run_once(self) -> None:
        files = list(self.inbox_dir.glob("*.json"))

        if not files:
            print("Aucun fichier dans inbox/")

        for file_path in files:
            self.process_file(file_path)

    def run_forever(self, interval_seconds: int = 30) -> None:
        print("Agent automatique lancé.")
        print(f"Surveillance du dossier : {self.inbox_dir}")
        print("CTRL + C pour arrêter.")

        while True:
            self.run_once()
            time.sleep(interval_seconds)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Disinfo Agent Auto Runner")
    parser.add_argument("--watch", action="store_true", help="Surveiller inbox en continu")
    parser.add_argument("--interval", type=int, default=30, help="Intervalle en secondes")

    args = parser.parse_args()

    runner = AutoRunner()

    if args.watch:
        runner.run_forever(interval_seconds=args.interval)
    else:
        runner.run_once()


if __name__ == "__main__":
    main()