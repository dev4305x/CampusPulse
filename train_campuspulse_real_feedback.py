"""Reproducible, timestamp-blind candidate training for curated CampusPulse feedback.

This script never overwrites live models. It evaluates OOF candidates, trains final
candidate artifacts, and writes a metrics report for human review.
"""
from __future__ import annotations
import hashlib, json, re, shutil, warnings, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "campuspulse_models"
DATA_DIR = ROOT / "data"
REAL_PATH = DATA_DIR / "campuspulse_real_feedback_curated.csv"
SEED = 20261030
N_SPLITS = 5
CATEGORY_MAP = {
    "teaching": "Academics", "course_content": "Academics", "examination": "Academics",
    "library_facilities": "Library", "labwork": "Laboratory", "extracurricular": "Events",
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().casefold())


def load_base() -> pd.DataFrame:
    pieces = []
    public = pd.read_csv(ROOT / "student_feedback_sentiment.csv")
    pieces.append(pd.DataFrame({"feedback": public.text, "sentiment": public.sentiment,
                                "category": public.category.map(CATEGORY_MAP)}))
    synth = pd.read_csv(ROOT / "campuspulse_realistic_synthetic_train.csv")
    pieces.append(synth[["feedback", "sentiment", "category"]])
    demo = pd.read_csv(ROOT / "campuspulse_mvp_feedback_1000.csv")
    pieces.append(demo[["feedback", "sentiment", "category"]])
    base = pd.concat(pieces, ignore_index=True)
    base = base.dropna(subset=["feedback", "sentiment", "category"])
    base["feedback"] = base.feedback.astype(str).str.strip()
    base["sentiment"] = base.sentiment.astype(str).str.strip().str.lower()
    base["category"] = base.category.astype(str).str.strip()
    base = base[base.feedback.ne("") & base.sentiment.isin(["negative", "neutral", "positive"])]
    base["_norm"] = base.feedback.map(normalize)
    base = base.drop_duplicates("_norm", keep="first").reset_index(drop=True)
    return base


def load_known_labeled_texts() -> set[str]:
    """Texts that the existing serialized baseline could have seen."""
    frames = [
        (pd.read_csv(ROOT / "student_feedback_sentiment.csv"), "text"),
        (pd.read_csv(ROOT / "campuspulse_realistic_synthetic_dataset.csv"), "feedback"),
        (pd.read_csv(ROOT / "campuspulse_mvp_feedback_1000.csv"), "feedback"),
    ]
    known = set()
    for frame, col in frames:
        known.update(frame[col].dropna().astype(str).map(normalize))
    return known


def make_model(task: str) -> Pipeline:
    if task == "sentiment":
        vec = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True)
    else:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=3000, random_state=42)
    return Pipeline([("tfidf", vec), ("clf", clf)])


def score(y, pred, labels):
    return {
        "n": int(len(y)), "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)), 4),
        "macro_precision": round(float(precision_score(y, pred, labels=labels, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y, pred, labels=labels, average="macro", zero_division=0)), 4),
        "labels": list(labels),
        "confusion_matrix_true_rows_pred_cols": confusion_matrix(y, pred, labels=labels).tolist(),
    }


def hybrid_predictions(task, model, texts, live_pipeline):
    out = []
    for text in texts:
        label = str(model.predict([text])[0])
        prob = model.predict_proba([text])[0]
        conf = float(max(prob))
        if task == "sentiment":
            out.append(live_pipeline._hybrid_sentiment(text, label, conf)[0])
        else:
            out.append(live_pipeline._hybrid_category(text, label, conf)[0])
    return out


def main():
    real_all = pd.read_csv(REAL_PATH).fillna("")
    included = real_all[real_all.review_status.eq("ready_for_training")].copy()
    pending = real_all[real_all.review_status.eq("pending_human_relabel")].copy()
    if len(real_all) != 27 or len(included) != 25 or len(pending) != 2:
        raise ValueError(f"Expected 27 total / 25 eligible / 2 pending; got {len(real_all)}/{len(included)}/{len(pending)}")
    if included.sentiment.eq("").any() or included.category.eq("").any():
        raise ValueError("Eligible feedback contains a missing sentiment or category label")
    # Explicitly restrict model inputs to narrative text. All other form fields remain metadata.
    real = included[["response_id", "feedback", "sentiment", "category"]].copy()
    real["_norm"] = real.feedback.map(normalize)
    if real._norm.duplicated().any():
        raise ValueError("Duplicate curated responses need review before cross-validation")

    base = load_base()
    # Never allow an exact normalized text from the form to appear in the base training corpus.
    real_norms = set(real._norm)
    overlap = int(base._norm.isin(real_norms).sum())
    base = base[~base._norm.isin(real_norms)].reset_index(drop=True)
    # Keep exact duplicates out of the test set as well: the serialized baseline
    # may have seen them through its original training data.
    known_labeled_texts = load_known_labeled_texts()
    evaluation_real = real[~real._norm.isin(known_labeled_texts)].reset_index(drop=True)

    import campuspulse_pipeline as live
    baseline_sent = joblib.load(MODEL_DIR / "sentiment_model_v4.pkl")
    baseline_cat = joblib.load(MODEL_DIR / "category_model_v4.pkl")

    folds = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(evaluation_real))
    results = {}
    oof = {"sentiment": {"baseline_raw": [None]*len(evaluation_real), "candidate_raw": [None]*len(evaluation_real),
                          "baseline_hybrid": [None]*len(evaluation_real), "candidate_hybrid": [None]*len(evaluation_real)},
           "category": {"baseline_raw": [None]*len(evaluation_real), "candidate_raw": [None]*len(evaluation_real),
                        "baseline_hybrid": [None]*len(evaluation_real), "candidate_hybrid": [None]*len(evaluation_real)}}
    labels_by_task = {"sentiment": ["negative", "neutral", "positive"],
                      "category": sorted(set(evaluation_real.category) | set(base.category))}

    for task in ("sentiment", "category"):
        target = evaluation_real[task].astype(str).tolist()
        baseline = baseline_sent if task == "sentiment" else baseline_cat
        for train_idx, test_idx in folds:
            train_frame = pd.concat([base[["feedback", "sentiment", "category"]],
                                     evaluation_real.iloc[train_idx][["feedback", "sentiment", "category"]]], ignore_index=True)
            train_frame["_norm"] = train_frame.feedback.map(normalize)
            train_frame = train_frame.drop_duplicates("_norm", keep="first")
            candidate = make_model(task)
            candidate.fit(train_frame.feedback.tolist(), train_frame[task].astype(str).tolist())
            texts = evaluation_real.iloc[test_idx].feedback.tolist()
            bp = [str(x) for x in baseline.predict(texts)]
            cp = [str(x) for x in candidate.predict(texts)]
            bh = hybrid_predictions(task, baseline, texts, live)
            ch = hybrid_predictions(task, candidate, texts, live)
            for idx, b, c, bhv, chv in zip(test_idx, bp, cp, bh, ch):
                oof[task]["baseline_raw"][idx] = b
                oof[task]["candidate_raw"][idx] = c
                oof[task]["baseline_hybrid"][idx] = bhv
                oof[task]["candidate_hybrid"][idx] = chv
        results[task] = {
            "baseline_raw_oof": score(target, oof[task]["baseline_raw"], labels_by_task[task]),
            "candidate_raw_oof": score(target, oof[task]["candidate_raw"], labels_by_task[task]),
            "baseline_pipeline_hybrid_oof": score(target, oof[task]["baseline_hybrid"], labels_by_task[task]),
            "candidate_pipeline_hybrid_oof": score(target, oof[task]["candidate_hybrid"], labels_by_task[task]),
        }

    # Preserve baseline artifacts before writing any new model files.
    backup_dir = MODEL_DIR / "backups_pre_real_feedback_20261031"
    backup_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ("sentiment_model_v4.pkl", "category_model_v4.pkl"):
        src = MODEL_DIR / name
        dst = backup_dir / name
        if not dst.exists():
            shutil.copy2(src, dst)
        hashes[name] = hashlib.sha256(dst.read_bytes()).hexdigest()
    (backup_dir / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    # Train final candidates on the full existing labelled corpus plus all eligible campus examples.
    final = pd.concat([base[["feedback", "sentiment", "category"]], real[["feedback", "sentiment", "category"]]], ignore_index=True)
    final["_norm"] = final.feedback.map(normalize)
    final = final.drop_duplicates("_norm", keep="first")
    candidates = {}
    for task, filename in (("sentiment", "sentiment_model_v5_candidate.pkl"),
                           ("category", "category_model_v5_candidate.pkl")):
        model = make_model(task)
        model.fit(final.feedback.tolist(), final[task].astype(str).tolist())
        joblib.dump(model, MODEL_DIR / filename, compress=3)
        candidates[task] = {"path": f"campuspulse_models/{filename}", "training_rows": int(len(final)),
                            "classes": [str(x) for x in model.classes_]}

    # Summarize exact training provenance; metadata columns are never converted into features.
    real_count = {str(k): int(v) for k, v in included.sentiment.value_counts().items()}
    category_count = {str(k): int(v) for k, v in included.category.value_counts().items()}
    metrics = {
        "evaluation_type": "5-fold out-of-fold evaluation on eligible new campus responses after excluding exact-text overlap with existing labeled data; each evaluated response is held out exactly once.",
        "split": {"method": "KFold", "n_splits": N_SPLITS, "shuffle": True, "random_state": SEED,
                  "exact_text_overlap_removed_from_base": overlap,
                  "exact_text_overlap_excluded_from_evaluation": int(len(real) - len(evaluation_real)),
                  "existing_text_sources_checked_for_exact_duplicates": ["student_feedback_sentiment.csv", "campuspulse_realistic_synthetic_dataset.csv", "campuspulse_mvp_feedback_1000.csv"],
                  "evaluation_response_ids": evaluation_real.response_id.tolist()},
        "dataset": {"total_rows": int(len(real_all)), "eligible_rows": int(len(included)),
                    "pending_human_relabel_rows": int(len(pending)),
                    "evaluation_rows_after_duplicate_exclusion": int(len(evaluation_real)),
                    "eligible_sentiment_counts": real_count, "eligible_category_counts": category_count,
                    "feature_fields": ["feedback"],
                    "metadata_fields_excluded_from_features": ["timestamp_metadata", "year_of_study_metadata", "area_as_reported", "location_metadata", "frequency_metadata", "response_id", "review_status", "review_note"],
                    "timestamp_used_as_feature": False},
        "existing_training_sources": {"student_feedback_sentiment.csv": 724,
                                      "campuspulse_realistic_synthetic_train.csv": 780,
                                      "campuspulse_mvp_feedback_1000.csv": 1000,
                                      "deduplicated_base_rows": int(len(base)),
                                      "final_candidate_unique_training_rows": int(len(final))},
        "candidate_config": {"sentiment": "word TF-IDF (1,2), unicode accent stripping, sublinear TF; LogisticRegression C=2, class_weight=balanced, max_iter=3000",
                             "category": "char_wb TF-IDF (3,5), sublinear TF; LogisticRegression C=2, class_weight=balanced, max_iter=3000"},
        "metrics": results,
        "candidate_artifacts": candidates,
        "backup_dir": "campuspulse_models/backups_pre_real_feedback_20261031",
        "backup_sha256": hashes,
        "promotion": {"decision": "not_promoted_pending_larger_representative_reviewed_sample",
                      "reason": "Only 25 eligible examples are available, with no verified positive examples; category coverage is sparse and rare categories have one example. Cross-validation is descriptive and cannot establish reliable generalization."},
        "runtime": {"python": "3.12.8 isolated runtime", "scikit_learn": "1.9.1 from archived venv",
                    "baseline_pickle_version": "1.8.0 (load emitted sklearn InconsistentVersionWarning)"},
    }
    report_path = DATA_DIR / "campuspulse_real_feedback_evaluation.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
