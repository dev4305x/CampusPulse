# CampusPulse real feedback training report

Date: 2026-10-31

## Data curation

- Retrieved 27 Google Form responses from the spreadsheet shared in the earlier conversation.
- 25 rows are eligible for supervised training. Sentiment labels were normalized as `Very Negative` → `negative`, `Negative` → `negative`, `Neutral` → `neutral`, and `Positive` → `positive`.
- CPR008 (exam scheduling) and CPR015 (food quality with food-poisoning/vomiting concerns) were labeled Positive in the form despite clearly critical narratives. Both are flagged `pending_human_relabel`, have no training target, and are excluded from all model training until reviewed.
- The 25 eligible rows contain 16 negative and 9 neutral labels, with no verified positive examples. Category counts: Wi-Fi 10, Food 7, Transport 3, Academics 2, Events 1, Laboratory 1, Infrastructure 1.
- CPR007 (`Bad`) exactly matches a text in the existing public dataset. It remains in the curated set and final candidate training corpus once, but is excluded from the evaluation so the current baseline cannot receive credit for a possible training duplicate. The evaluation therefore covers 24 unique new narratives.
- Timestamps, year, location, frequency, and other form fields are stored as metadata. The training code explicitly uses only `feedback` as model input.

## Evaluation design

Five-fold shuffled KFold cross-validation (random seed 20261030) was run on the 24 eligible, non-duplicate new narratives. Each was held out once. Candidate fold models were trained on existing labeled data plus the other eligible new narratives. Exact text overlaps with the existing training sources were removed before fitting.

Metrics are descriptive for this small sample. The sentiment test set contains no positive examples, and several category labels have only one example. They should not be read as reliable campus-wide accuracy estimates.

## Exact out-of-fold metrics

Accuracy / macro F1 / weighted F1:

All four rows below use the same 24-response out-of-fold evaluation set.

| Task and inference path | Current V4 baseline | Candidate | Change |
|---|---:|---:|---:|
| Sentiment, raw model | 58.33% / 0.3082 / 0.5325 | 62.50% / 0.3199 / 0.5543 | +4.17 pp accuracy |
| Sentiment, current hybrid rules | 62.50% / 0.3199 / 0.5543 | 62.50% / 0.3199 / 0.5543 | no change |
| Category, raw model | 75.00% / 0.3881 / 0.7806 | 83.33% / 0.4296 / 0.8565 | +8.33 pp accuracy |
| Category, current hybrid rules | 87.50% / 0.4630 / 0.8981 | 87.50% / 0.4630 / 0.8981 | no change |

The candidate models improve the raw-model accuracy on these held-out responses, but the current hybrid inference layer already brings category accuracy to 87.50% and the candidates do not improve either hybrid path's accuracy. Sentiment hybrid accuracy is unchanged. Given the sample size, zero positive examples, and sparse category coverage, these results do not justify replacing the live models.

## Changes and artifacts

- Added `data/campuspulse_real_feedback_curated.csv` with all 27 responses and review metadata.
- Added `train_campuspulse_real_feedback.py` to reproduce curation-aware, timestamp-blind training and evaluation.
- Trained candidate artifacts `campuspulse_models/sentiment_model_v5_candidate.pkl` and `campuspulse_models/category_model_v5_candidate.pkl` on 2,019 unique rows from the existing labeled corpora plus the 25 eligible responses.
- Backed up the live sentiment and category models under `campuspulse_models/backups_pre_real_feedback_20261031/`; SHA-256 values are in `SHA256SUMS.json`.
- Preserved the V4 models as the live models. No candidate model was promoted.
- Exact metrics and confusion matrices are in `data/campuspulse_real_feedback_evaluation.json`.

## Runtime note

The archived virtual environment was built for Python 3.12.10 but its configured Windows Store interpreter was unavailable. Training ran under an isolated Python 3.12.8 runtime using the archive's Python 3.12 dependencies (scikit-learn 1.9.1). The existing V4 model files were serialized with scikit-learn 1.8.0; loading them emitted compatibility warnings, although prediction and evaluation completed. Re-run in the project's matching scikit-learn 1.8.0 environment before using the reported baseline numbers as a formal benchmark.
