from urllib.parse import urlparse
import tldextract


class SourceReputationTool:
    """
    Score simple de réputation.
    V1 = règles simples.
    Après, on pourra remplacer par une vraie base de réputation.
    """

    TRUSTED_DOMAINS = {
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "bbc.co.uk",
        "france24.com",
        "lemonde.fr",
        "afp.com",
        "aljazeera.com"
    }

    RISKY_KEYWORDS = [
        "truth",
        "real-news",
        "breaking-alert",
        "viral",
        "buzz",
        "secret",
        "exposed"
    ]

    def analyze_source(self, source: str) -> dict:
        if not source:
            return {
                "domain": "",
                "score": 0.5,
                "label": "unknown",
                "signals": ["missing_source"]
            }

        parsed = urlparse(source if source.startswith("http") else f"https://{source}")
        extracted = tldextract.extract(parsed.netloc)

        domain = f"{extracted.domain}.{extracted.suffix}" if extracted.suffix else source.lower()
        domain = domain.lower()

        signals = []
        score = 0.5

        if domain in self.TRUSTED_DOMAINS:
            score -= 0.3
            signals.append("known_trusted_domain")

        for kw in self.RISKY_KEYWORDS:
            if kw in domain:
                score += 0.2
                signals.append(f"risky_keyword:{kw}")

        if not extracted.suffix:
            score += 0.2
            signals.append("invalid_or_unclear_domain")

        score = max(0.0, min(1.0, score))

        if score < 0.35:
            label = "low_risk_source"
        elif score < 0.7:
            label = "medium_risk_source"
        else:
            label = "high_risk_source"

        return {
            "domain": domain,
            "score": score,
            "label": label,
            "signals": signals
        }