import requests
from src.config import GOOGLE_FACT_CHECK_API_KEY


class FactCheckTool:
    """
    Recherche le claim dans Google Fact Check API.
    Si aucune clé API n'est fournie, l'outil est ignoré proprement.
    """

    BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

    def search_claim(self, query: str, language_code: str = "fr", page_size: int = 5) -> dict:
        if not GOOGLE_FACT_CHECK_API_KEY:
            return {
                "enabled": False,
                "message": "Google Fact Check API key missing",
                "matches": []
            }

        params = {
            "query": query,
            "languageCode": language_code,
            "pageSize": page_size,
            "key": GOOGLE_FACT_CHECK_API_KEY
        }

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            matches = []
            for claim in data.get("claims", []):
                claim_text = claim.get("text", "")
                claimant = claim.get("claimant", "")
                claim_date = claim.get("claimDate", "")

                reviews = []
                for review in claim.get("claimReview", []):
                    reviews.append({
                        "publisher": review.get("publisher", {}).get("name", ""),
                        "url": review.get("url", ""),
                        "title": review.get("title", ""),
                        "rating": review.get("textualRating", "")
                    })

                matches.append({
                    "claim": claim_text,
                    "claimant": claimant,
                    "claim_date": claim_date,
                    "reviews": reviews
                })

            return {
                "enabled": True,
                "matches": matches
            }

        except Exception as e:
            return {
                "enabled": True,
                "error": str(e),
                "matches": []
            }