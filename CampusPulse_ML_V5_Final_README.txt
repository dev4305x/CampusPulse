# CampusPulse ML V5 — Final Integration

## Data
- EduRABSA ACD: 4,000 public training examples; 3,200 train / 800 held-out test for the auxiliary aspect model.
- Student Feedback Dataset: 724 public examples; 579 train / 145 held-out test for CampusPulse category benchmarking.
- CampusPulse synthetic data: 976 realistic synthetic examples plus the original 1,000-record demo dataset.

## Models
- Sentiment: V4 word TF-IDF + Logistic Regression (selected on the public held-out benchmark).
- Main CampusPulse category: V4 character TF-IDF + Logistic Regression (selected on the public held-out benchmark).
- Fine-grained aspect detector: V5 EduRABSA multi-label word+character TF-IDF + One-vs-Rest Logistic Regression.
- Similarity: lightweight TF-IDF cosine similarity baseline, now restricted to the predicted broad category to reduce cross-domain false matches.

## Evaluation
- Main category V4 on held-out public Student Feedback test: accuracy 88.28%, weighted F1 88.33%.
- EduRABSA aspect model on held-out EduRABSA test: micro F1 65.96%, macro F1 54.18%, top-1 gold-aspect hit rate 82.88%.
- EduRABSA-augmented single-label category candidate scored 65.52% accuracy / 67.11% weighted F1 on the same public category test, so it was not promoted to the live category model.

## Product behavior
The live pipeline returns sentiment, broad CampusPulse category, fine-grained EduRABSA aspects, and related negative complaints. The broad category remains V4 because the benchmark supports it. EduRABSA adds detail without forcing a regression.

## Important limitation
The similarity engine is a TF-IDF baseline, not a transformer embedding model. EduRABSA and the Student Feedback Dataset are public datasets; CampusPulse demo records are synthetic. None of these metrics should be presented as real-campus accuracy.

## Final smoke tests
See `campuspulse_v5_integration_tests.json`. Representative examples were checked for broad category, sentiment, aspect output, and related-complaint behavior.
