import csv
from datetime import datetime
from pathlib import Path

from src.config import BASE_DIR


class HistoryStorage:
    def __init__(self):
        self.history_dir = BASE_DIR / "outputs" / "history"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.history_dir / "analysis_history.csv"

    def save(self, report: dict) -> None:
        decision = report.get("decision", {})
        input_data = report.get("input", {})
        source_data = report.get("signals", {}).get("source_reputation", {})

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "text": input_data.get("text", ""),
            "source": input_data.get("source", ""),
            "domain": source_data.get("domain", ""),
            "risk_score": decision.get("risk_score", ""),
            "risk_level": decision.get("risk_level", ""),
            "label": decision.get("label", ""),
            "report_path": report.get("report_path", "")
        }

        file_exists = self.history_file.exists()

        with open(self.history_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

    def read_last(self, limit: int = 5) -> list[dict]:
        if not self.history_file.exists():
            return []

        with open(self.history_file, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        return rows[-limit:][::-1]