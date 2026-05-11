import json
from datetime import datetime
from src.config import REPORTS_DIR


class ReportTool:
    def save_report(self, report: dict) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPORTS_DIR / f"report_{timestamp}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        return str(path)