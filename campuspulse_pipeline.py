import os, json, sqlite3, joblib, re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "campuspulse_models")
DB_PATH = os.path.join(BASE, "campuspulse.db")


def load(path):
    return joblib.load(path)

SENTIMENT = load(os.path.join(MODEL_DIR, "sentiment_model_v4.pkl"))
CATEGORY = load(os.path.join(MODEL_DIR, "category_model_v4.pkl"))
ISSUE = load(os.path.join(MODEL_DIR, "issue_tfidf_hybrid.pkl"))
ASPECT = load(os.path.join(MODEL_DIR, "edurabsa_aspect_model_v5.pkl"))
ASPECT_LABELS = load(os.path.join(MODEL_DIR, "edurabsa_aspect_labels_v5.pkl"))

# Explicit student-language cues are used as a small safety/quality layer on top
# of the statistical model. They target common conversational phrasing that a
# TF-IDF classifier can miss (e.g. "tired of", "fed up", "too many exams").
NEGATIVE_PHRASES = [
    r"\btired of\b", r"\bfed up\b", r"\bsick of\b", r"\bfrustrat(?:ed|ing)\b",
    r"\bexhaust(?:ed|ing)\b", r"\boverwhelm(?:ed|ing)\b", r"\bstress(?:ed|ful|ing)?\b",
    r"\bterrible\b", r"\bawful\b", r"\bhorrible\b", r"\bunbearable\b",
    r"\btoo many\b", r"\btoo frequent\b", r"\bnot satisfied\b", r"\bpoorly scheduled\b",
    r"\bdoesn'?t work\b", r"\bnot working\b", r"\bkeeps? (?:crashing|disconnecting|failing)\b",
    r"\bdisappointed\b", r"\bhate\b", r"\bworst\b", r"\bannoy(?:ed|ing)\b", r"\bno seats?\b", r"\bnot enough seats?\b", r"\bsuper slow\b", r"\bvery slow\b", r"\btoo slow\b"
]
POSITIVE_PHRASES = [
    r"\bexcellent\b", r"\bvery good\b", r"\breally good\b", r"\bgreat\b", r"\bawesome\b",
    r"\benjoy(?:ed|ing)?\b", r"\bhappy with\b", r"\bsatisfied with\b", r"\bworks? well\b",
    r"\bconvenient\b", r"\bhelpful\b", r"\bclear(?:ly)?\b"
]
NEGATION_PREFIX = re.compile(r"\b(?:not|no|never|don't|dont|didn't|didnt|isn't|isnt|wasn't|wasnt)\s+$")

CATEGORY_RULES = [
    ("Wi-Fi", [r"\bwifi\b", r"\bwi[ -]?fi\b", r"\binternet\b", r"\bnetwork\b"]),
    ("Food", [r"\bfood\b", r"\bmess\b", r"\blunch\b", r"\bdinner\b", r"\bbreakfast\b"]),
    ("Hostel", [r"\bhostel\b", r"\broom\b", r"\bdorm\b"]),
    ("Transport", [r"\bbus\b", r"\btransport\b", r"\bshuttle\b"]),
    ("Library", [r"\blibrary\b", r"\bbooks?\b", r"\bseats?\b.*library"]),
    ("Laboratory", [r"\blab\b", r"\blaboratory\b", r"\blabwork\b", r"\bequipment\b"]),
    ("Events", [r"\bevent\b", r"\bclub\b", r"\bextracurricular\b"]),
    ("Safety", [r"\bsafety\b", r"\bunsafe\b", r"\bemergency\b"]),
    ("Cleanliness", [r"\bclean(?:liness)?\b", r"\bdirty\b", r"\bgarbage\b", r"\bwaste\b"]),
    ("Examination", [r"\bexam(?:s|ination)?\b", r"\btest(?:s)?\b", r"\bassessment\b", r"\bquiz(?:zes)?\b"]),
]


def _predict(bundle, text):
    if hasattr(bundle, "predict"):
        label = bundle.predict([text])[0]
        if hasattr(bundle, "predict_proba"):
            probs = bundle.predict_proba([text])[0]
            confidence = float(max(probs))
        else:
            confidence = 0.0
    else:
        x = bundle["vectorizer"].transform([text])
        model = bundle["model"]
        label = model.predict(x)[0]
        confidence = float(max(model.predict_proba(x)[0]))
    return label, round(confidence, 3)


def _cue_score(text, patterns):
    text = text.lower()
    hits = []
    for p in patterns:
        m = re.search(p, text)
        if m:
            hits.append(m.group(0))
    return hits


def _hybrid_sentiment(text, model_label, model_confidence):
    lower = text.lower()
    neg_hits = _cue_score(lower, NEGATIVE_PHRASES)
    pos_hits = _cue_score(lower, POSITIVE_PHRASES)

    # Handle simple negation before positive phrases ("not good", "not happy").
    effective_pos = []
    for hit in pos_hits:
        idx = lower.find(hit.lower())
        prefix = lower[max(0, idx - 8):idx]
        if not NEGATION_PREFIX.search(prefix):
            effective_pos.append(hit)

    # Strong conversational dissatisfaction overrides a conflicting statistical
    # prediction when there is no equally strong positive signal.
    strong_neg = len(neg_hits) >= 1
    strong_pos = len(effective_pos) >= 1
    if strong_neg and not strong_pos:
        # Explicit phrases such as "tired of", "fed up", "too many" are high-value
        # signals; confidence is intentionally capped below 0.95.
        conf = min(0.90, max(0.76, model_confidence, 0.76 + 0.04 * (len(neg_hits) - 1)))
        return "negative", round(conf, 3), neg_hits, effective_pos, True

    if strong_pos and not strong_neg:
        conf = min(0.90, max(0.76, model_confidence, 0.76 + 0.04 * (len(effective_pos) - 1)))
        return "positive", round(conf, 3), neg_hits, effective_pos, True

    return str(model_label), model_confidence, neg_hits, effective_pos, False


def _hybrid_category(text, model_label, model_confidence):
    lower = text.lower()
    matches = []
    for label, patterns in CATEGORY_RULES:
        if any(re.search(p, lower) for p in patterns):
            matches.append(label)
    # Only override when the rule is clear and the model is uncertain, or when
    # the text contains an unambiguous domain term such as "exam" or "wifi".
    if matches:
        # Examination maps to the app's broad Academics bucket.
        chosen = matches[0]
        if chosen == "Examination":
            chosen = "Academics"
        if model_confidence < 0.72 or chosen in {"Wi-Fi", "Library", "Laboratory", "Transport", "Food", "Hostel", "Academics"}:
            return chosen, round(min(0.92, max(0.82, model_confidence)), 3), matches, True
    return str(model_label), model_confidence, matches, False


def predict_aspects(text, top_k=3, threshold=0.25):
    probabilities = ASPECT.predict_proba([text])[0]
    ranked = np.argsort(probabilities)[::-1]
    results = []
    for idx in ranked[:top_k]:
        score = float(probabilities[idx])
        if score < threshold and results:
            break
        results.append({"aspect": ASPECT_LABELS.classes_[idx], "confidence": round(score, 3)})
    return results


def _live_issue_records():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = __import__("pandas").read_sql_query(
            "SELECT feedback, category, location, created_at FROM feedback WHERE source='live' AND lower(sentiment)='negative'",
            conn,
        )
        conn.close()
        if df.empty:
            return df
        df["date"] = df["created_at"]
        df["source"] = "live"
        return df[["feedback", "category", "location", "date", "source"]]
    except Exception:
        return __import__("pandas").DataFrame()


def find_similar(text, predicted_category=None, top_k=5, threshold=0.30):
    """Find related negative complaints using word + character TF-IDF.

    Character features help with casual/spelling variants (wifi/Wi-Fi, slow/slower),
    while the broad-category filter prevents unrelated domains from dominating.
    Duplicate complaint text is collapsed so the UI shows distinct examples.
    """
    word_x = ISSUE["vectorizer"].transform([text])
    char_x = ISSUE["char_vectorizer"].transform([text])
    static = ISSUE["records"].copy()
    if "source" not in static.columns:
        static["source"] = "synthetic_demo"
    live = _live_issue_records()
    records = __import__("pandas").concat([static, live], ignore_index=True)
    if records.empty:
        return []

    texts = records["feedback"].fillna("").astype(str).tolist()
    word_mat = ISSUE["vectorizer"].transform(texts)
    char_mat = ISSUE["char_vectorizer"].transform(texts)
    word_scores = cosine_similarity(word_x, word_mat).ravel()
    char_scores = cosine_similarity(char_x, char_mat).ravel()
    scores = 0.60 * word_scores + 0.40 * char_scores

    if predicted_category is not None and "category" in records.columns:
        cats = records["category"].fillna("").astype(str).str.lower()
        target = str(predicted_category).lower()
        if target == "academics":
            mask = cats.isin(["academics", "examination"])
        else:
            mask = cats.eq(target)
        scores = np.where(mask.to_numpy(), scores, -1.0)

    results = []
    seen = set()
    query_norm = re.sub(r"\s+", " ", text.strip().lower())
    for idx in scores.argsort()[::-1]:
        score = float(scores[idx])
        if score < threshold:
            break
        row = records.iloc[idx]
        feedback = str(row["feedback"])
        norm = re.sub(r"\s+", " ", feedback.strip().lower())
        if norm == query_norm or norm in seen:
            continue
        seen.add(norm)
        results.append({
            "feedback": feedback,
            "category": row["category"],
            "location": row.get("location", "Not specified"),
            "date": str(row.get("date", "")),
            "similarity": round(score, 3),
            "source": row.get("source", "synthetic_demo")
        })
        if len(results) >= top_k:
            break
    return results

def save_feedback(text, result, location="Not specified"):
    conn = sqlite3.connect(DB_PATH)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(feedback)").fetchall()]
    if "aspects_json" not in cols:
        conn.execute("ALTER TABLE feedback ADD COLUMN aspects_json TEXT")
    if "review_status" not in cols:
        conn.execute("ALTER TABLE feedback ADD COLUMN review_status TEXT")
    conn.execute("""
        INSERT INTO feedback
        (feedback,sentiment,sentiment_confidence,category,category_confidence,location,source,aspects_json,review_status)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (
        text, result["sentiment"], result["sentiment_confidence"], result["category"],
        result["category_confidence"], location, "live",
        json.dumps(result.get("aspects", []), ensure_ascii=False), result.get("review_status", ""),
    ))
    conn.commit()
    conn.close()


def analyze_feedback(text, top_k=5, location="Not specified", save=False):
    raw_sentiment, raw_sc = _predict(SENTIMENT, text)
    sentiment, sc, neg_hits, pos_hits, sentiment_rule = _hybrid_sentiment(text, raw_sentiment, raw_sc)
    raw_category, raw_cc = _predict(CATEGORY, text)
    category, cc, category_hits, category_rule = _hybrid_category(text, raw_category, raw_cc)
    aspects = predict_aspects(text)
    similar = find_similar(text, predicted_category=category, top_k=top_k) if sentiment.lower() == "negative" else []

    review_reasons = []
    if sc < 0.60:
        review_reasons.append("low sentiment confidence")
    if cc < 0.60:
        review_reasons.append("low category confidence")
    review_status = "Needs review" if review_reasons else "Model result"

    result = {
        "feedback": text,
        "sentiment": sentiment,
        "sentiment_confidence": sc,
        "category": category,
        "category_confidence": cc,
        "aspects": aspects,
        "similar_issue_count": len(similar),
        "similar_issues": similar,
        "location": location,
        "sentiment_cues": neg_hits + pos_hits,
        "category_cues": category_hits,
        "hybrid_adjusted": bool(sentiment_rule or category_rule),
        "review_status": review_status,
        "review_reasons": review_reasons,
        "raw_sentiment": str(raw_sentiment),
        "raw_category": str(raw_category),
    }
    if save:
        save_feedback(text, result, location)
    return result
