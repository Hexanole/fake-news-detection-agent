class TextAnalysisTool:
    """
    Analyse locale du texte.
    V1 : règles simples.
    Plus tard : remplacer/renforcer avec un vrai modèle entraîné.
    """

    SENSATIONAL_WORDS = [
        "urgent", "breaking", "choc", "scandale", "secret",
        "incroyable", "vérité cachée", "alerte", "exclusif",
        "les médias cachent", "partagez", "avant suppression",
        "catastrophe", "explosion énorme"
    ]

    UNCERTAIN_WORDS = [
        "on dit", "il paraît", "selon des sources", "rumeur",
        "peut-être", "non confirmé", "source anonyme"
    ]

    def analyze(self, text: str) -> dict:
        text = text or ""
        lowered = text.lower()

        signals = []
        score = 0.0

        if len(text.strip()) < 30:
            score += 0.15
            signals.append("very_short_text")

        for word in self.SENSATIONAL_WORDS:
            if word in lowered:
                score += 0.08
                signals.append(f"sensational_word:{word}")

        for word in self.UNCERTAIN_WORDS:
            if word in lowered:
                score += 0.08
                signals.append(f"uncertain_word:{word}")

        exclamation_count = text.count("!")
        if exclamation_count >= 3:
            score += 0.1
            signals.append("too_many_exclamation_marks")

        uppercase_words = [
            w for w in text.split()
            if len(w) > 3 and w.isupper()
        ]

        if len(uppercase_words) >= 3:
            score += 0.1
            signals.append("many_uppercase_words")

        score = max(0.0, min(1.0, score))

        if score < 0.25:
            label = "normal_text"
        elif score < 0.6:
            label = "suspicious_text"
        else:
            label = "highly_suspicious_text"

        return {
            "score": round(score, 3),
            "label": label,
            "signals": signals
        }