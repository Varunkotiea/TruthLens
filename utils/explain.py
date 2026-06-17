import numpy as np
import re
from utils.preprocess import preprocess


def get_top_keywords(text: str, vectorizer, model, top_n: int = 15) -> dict:
    cleaned = preprocess(text)
    vec = vectorizer.transform([cleaned])

    feature_names = np.array(vectorizer.get_feature_names_out())
    tfidf_scores = vec.toarray()[0]
    coefs = model.coef_[0]

    present_mask = tfidf_scores > 0
    present_indices = np.where(present_mask)[0]

    if len(present_indices) == 0:
        return {"real_keywords": [], "fake_keywords": [], "all_features": []}

    contributions = tfidf_scores[present_indices] * coefs[present_indices]
    words = feature_names[present_indices]

    real_mask = contributions > 0
    fake_mask = contributions < 0

    real_pairs = sorted(
        zip(words[real_mask], contributions[real_mask]),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    fake_pairs = sorted(
        zip(words[fake_mask], np.abs(contributions[fake_mask])),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    all_pairs = sorted(
        zip(words, tfidf_scores[present_indices]),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n * 2]

    return {
        "real_keywords": [(str(w), float(round(s, 4))) for w, s in real_pairs],
        "fake_keywords": [(str(w), float(round(s, 4))) for w, s in fake_pairs],
        "all_features":  [(str(w), float(round(s, 4))) for w, s in all_pairs],
    }


def normalize_scores(pairs: list[tuple]) -> list[tuple]:
    if not pairs:
        return []
    max_score = max(s for _, s in pairs)
    if max_score == 0:
        return [(w, 0.0) for w, _ in pairs]
    return [(w, round((s / max_score) * 100, 1)) for w, s in pairs]


def get_category_hint(text: str) -> str:
    text_lower = text.lower()
    categories = {
        "🏛️ Politics":      r"\b(president|congress|senate|election|democrat|republican|government|vote|ballot|policy|white house|minister|parliament)\b",
        "💰 Economy":       r"\b(economy|gdp|inflation|stock|market|trade|bank|dollar|tax|budget|recession|financial|invest)\b",
        "⚕️ Health":        r"\b(vaccine|covid|cancer|disease|hospital|health|doctor|medical|drug|virus|pandemic|medicine|surgery)\b",
        "🔬 Science":       r"\b(research|study|scientist|nasa|climate|space|discovery|experiment|technology|ai|robot|quantum)\b",
        "⚽ Sports":        r"\b(football|basketball|soccer|nba|nfl|championship|athlete|game|match|tournament|olympic|player)\b",
        "🌍 World Affairs": r"\b(war|military|nato|ukraine|russia|china|israel|iran|terrorism|foreign|diplomat|sanction|refugee)\b",
        "💻 Technology":    r"\b(apple|google|microsoft|elon|tesla|twitter|facebook|meta|openai|chatgpt|startup|silicon valley|crypto|bitcoin)\b",
        "🎬 Entertainment": r"\b(movie|celebrity|hollywood|music|oscar|grammy|singer|actor|film|album|concert|streaming|netflix)\b",
    }

    for label, pattern in categories.items():
        if re.search(pattern, text_lower):
            return label

    return "📰 General News"