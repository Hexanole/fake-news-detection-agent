from agent.tools.factcheck_tool import FactCheckTool
from agent.tools.web_search_tool import WebSearchTool
from agent.tools.ocr_tool import OCRTool
from agent.tools.image_analysis_tool import ImageAnalysisTool
from agent.tools.source_reputation_tool import SourceReputationTool
from agent.tools.report_tool import ReportTool
from agent.tools.text_analysis_tool import TextAnalysisTool
from agent.tools.ml_text_model_tool import MLTextModelTool
from agent.tools.ml_image_model_tool import MLImageModelTool
from agent.storage import HistoryStorage


class VerifyNewsPipeline:
    def __init__(self):
        self.factcheck_tool = FactCheckTool()
        self.web_search_tool = WebSearchTool()
        self.ocr_tool = OCRTool()
        self.image_tool = ImageAnalysisTool()
        self.source_tool = SourceReputationTool()
        self.text_tool = TextAnalysisTool()
        self.ml_text_tool = MLTextModelTool()
        self.ml_image_tool = MLImageModelTool()
        self.report_tool = ReportTool()
        self.history_storage = HistoryStorage()

    def verify(self, text: str, image_path: str = "", source: str = "") -> dict:
        text = text.strip() if text else ""

        text_result = self.text_tool.analyze(text)
        ml_text_result = self.ml_text_tool.predict(text)

        factcheck_result = self.factcheck_tool.search_claim(text) if text else {
            "enabled": False,
            "matches": []
        }

        web_result = self.web_search_tool.search_gdelt(text) if text else {
            "enabled": False,
            "articles_found": 0,
            "articles": []
        }

        image_result = self.image_tool.analyze(image_path) if image_path else {
            "success": False,
            "message": "No image provided"
        }

        ml_image_result = self.ml_image_tool.predict(image_path) if image_path else {
            "enabled": False,
            "message": "No image provided"
        }

        ocr_result = self.ocr_tool.extract_text(image_path) if image_path else {
            "success": False,
            "text": ""
        }

        source_result = self.source_tool.analyze_source(source)

        risk_score, reasons = self._calculate_risk(
            text_result=text_result,
            ml_text_result=ml_text_result,
            factcheck_result=factcheck_result,
            web_result=web_result,
            image_result=image_result,
            ml_image_result=ml_image_result,
            ocr_result=ocr_result,
            source_result=source_result,
        )

        risk_level = self._risk_level(risk_score)

        report = {
            "input": {
                "text": text,
                "image_path": image_path,
                "source": source,
            },
            "decision": {
                "risk_score": risk_score,
                "risk_level": risk_level,
                "label": self._label_from_risk(risk_score),
                "reasons": reasons,
            },
            "signals": {
                "text_analysis": text_result,
                "ml_text_model": ml_text_result,
                "factcheck": factcheck_result,
                "web_search": web_result,
                "image_analysis": image_result,
                "ml_image_model": ml_image_result,
                "ocr": ocr_result,
                "source_reputation": source_result,
            },
        }

        report_path = self.report_tool.save_report(report)
        report["report_path"] = report_path

        self.history_storage.save(report)

        return report

    def _calculate_risk(
        self,
        text_result: dict,
        ml_text_result: dict,
        factcheck_result: dict,
        web_result: dict,
        image_result: dict,
        ml_image_result: dict,
        ocr_result: dict,
        source_result: dict,
    ) -> tuple[float, list[str]]:

        score = 0.10
        reasons = []

        # 1. Modèle TensorFlow texte
        if ml_text_result.get("enabled"):
            fake_probability = float(ml_text_result.get("fake_probability", 0.5))
            score += fake_probability * 0.35

            if fake_probability >= 0.70:
                reasons.append("Le modèle TensorFlow texte classe cette news comme probablement fake.")
            elif fake_probability >= 0.40:
                reasons.append("Le modèle TensorFlow texte est incertain.")
            else:
                reasons.append("Le modèle TensorFlow texte ne détecte pas de signal fake fort.")
        else:
            text_score = text_result.get("score", 0.0)
            score += text_score * 0.20
            reasons.append("Modèle TensorFlow texte non disponible : utilisation des règles textuelles.")

        # 2. Analyse textuelle par règles
        text_score = text_result.get("score", 0.0)
        score += text_score * 0.10

        if text_result.get("label") in ["suspicious_text", "highly_suspicious_text"]:
            reasons.append("Le texte contient des signaux linguistiques suspects.")

        # 3. Source
        source_score = source_result.get("score", 0.5)
        score += source_score * 0.12

        if source_result.get("label") == "high_risk_source":
            reasons.append("La source semble risquée ou peu claire.")

        if "missing_source" in source_result.get("signals", []):
            reasons.append("Aucune source claire n'a été fournie.")

        # 4. Fact-checking
        matches = factcheck_result.get("matches", [])

        if matches:
            reasons.append("Des résultats de fact-checking liés au claim ont été trouvés.")

            ratings = []
            for match in matches:
                for review in match.get("reviews", []):
                    ratings.append(review.get("rating", "").lower())

            negative_words = [
                "false", "faux", "misleading", "trompeur",
                "incorrect", "fake", "pants on fire",
            ]

            positive_words = [
                "true", "vrai", "correct", "accurate",
            ]

            if any(any(word in rating for word in negative_words) for rating in ratings):
                score += 0.25
                reasons.append("Un fact-check semble indiquer une information fausse ou trompeuse.")

            elif any(any(word in rating for word in positive_words) for rating in ratings):
                score -= 0.20
                reasons.append("Un fact-check semble confirmer l'information.")

            else:
                score += 0.05
                reasons.append("Des fact-checks existent, mais leur conclusion n'est pas claire.")

        # 5. Recherche web
        articles_found = web_result.get("articles_found", 0)

        if articles_found == 0:
            score += 0.07
            reasons.append("Aucun article similaire n'a été trouvé dans la recherche web.")
        elif articles_found >= 3:
            score -= 0.07
            reasons.append("Plusieurs articles similaires ont été trouvés sur le web.")

        # 6. Analyse image technique
        if image_result.get("success"):
            image_signals = image_result.get("suspicious_signals", [])
            if image_signals:
                score += 0.07
                reasons.append(f"Signaux suspects détectés dans l'image : {image_signals}.")
        else:
            if image_result.get("error"):
                score += 0.04
                reasons.append("L'image n'a pas pu être analysée.")

        # 7. Modèle TensorFlow image
        if ml_image_result.get("enabled"):
            image_fake_probability = float(ml_image_result.get("fake_probability", 0.5))
            score += image_fake_probability * 0.20

            if image_fake_probability >= 0.70:
                reasons.append("Le modèle TensorFlow image associe cette image à un risque élevé.")
            elif image_fake_probability >= 0.40:
                reasons.append("Le modèle TensorFlow image est incertain.")
            else:
                reasons.append("Le modèle TensorFlow image ne détecte pas de signal fake fort.")
        else:
            image_message = ml_image_result.get("message")
            if image_message and image_message != "No image provided":
                reasons.append(f"Modèle image non utilisé : {image_message}.")

        # 8. OCR
        ocr_text = ocr_result.get("text", "")
        if ocr_text:
            reasons.append("Du texte a été détecté dans l'image par OCR.")

        score = max(0.0, min(1.0, score))

        if not reasons:
            reasons.append("Aucun signal fort de désinformation n'a été détecté.")

        return round(score, 3), reasons

    def _risk_level(self, score: float) -> str:
        if score < 0.35:
            return "faible"
        if score < 0.7:
            return "moyen"
        return "élevé"

    def _label_from_risk(self, score: float) -> str:
        if score < 0.35:
            return "probably_reliable"
        if score < 0.7:
            return "uncertain"
        return "potential_disinformation"