import requests
from src.config import USER_AGENT


class WebSearchTool:
    """
    Recherche gratuite via GDELT DOC API.
    Version robuste : gère les réponses non JSON, les erreurs réseau et les requêtes trop longues.
    """

    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    def clean_query(self, query: str, max_words: int = 12) -> str:
        query = query.strip()
        words = query.split()
        return " ".join(words[:max_words])

    def search_gdelt(self, query: str, max_records: int = 10) -> dict:
        clean_query = self.clean_query(query)

        if not clean_query:
            return {
                "enabled": True,
                "query": "",
                "articles_found": 0,
                "articles": [],
                "error": "empty_query"
            }

        params = {
            "query": clean_query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": max_records,
            "sort": "HybridRel"
        }

        headers = {
            "User-Agent": USER_AGENT
        }

        try:
            response = requests.get(
                self.BASE_URL,
                params=params,
                headers=headers,
                timeout=25
            )

            content_type = response.headers.get("Content-Type", "")
            text_preview = response.text[:300]

            if response.status_code != 200:
                return {
                    "enabled": True,
                    "query": clean_query,
                    "articles_found": 0,
                    "articles": [],
                    "error": f"http_status_{response.status_code}",
                    "response_preview": text_preview
                }

            try:
                data = response.json()
            except Exception:
                return {
                    "enabled": True,
                    "query": clean_query,
                    "articles_found": 0,
                    "articles": [],
                    "error": "non_json_response",
                    "content_type": content_type,
                    "response_preview": text_preview
                }

            articles = []
            for item in data.get("articles", []):
                articles.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "domain": item.get("domain", ""),
                    "source_country": item.get("sourcecountry", ""),
                    "language": item.get("language", ""),
                    "seendate": item.get("seendate", "")
                })

            return {
                "enabled": True,
                "query": clean_query,
                "articles_found": len(articles),
                "articles": articles
            }

        except Exception as e:
            return {
                "enabled": True,
                "query": clean_query,
                "articles_found": 0,
                "articles": [],
                "error": str(e)
            }