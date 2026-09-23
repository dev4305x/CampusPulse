from campuspulse_pipeline import analyze_feedback

CASES = [
    ("Im really tired of all this exams and test", "negative", "Academics"),
    ("The exam schedule is terrible.", "negative", "Academics"),
    ("Exams were okay.", "neutral", "Academics"),
    ("I enjoyed today's exam.", "positive", "Academics"),
    ("The professor is good but the exams are too frequent.", "negative", "Academics"),
    ("C block wifi dies every night after 7", "negative", "Wi-Fi"),
    ("library is fine but there are no seats", "negative", "Library"),
    ("lab PC is super slow", "negative", "Laboratory"),
]

failed=[]
for text, expected_sentiment, expected_category in CASES:
    r=analyze_feedback(text, save=False)
    ok=(r["sentiment"].lower()==expected_sentiment and r["category"]==expected_category)
    print(f"{'PASS' if ok else 'FAIL'} | {text} | {r['sentiment']} | {r['category']} | {r['sentiment_confidence']:.0%} | {r['category_confidence']:.0%}")
    if not ok: failed.append((text,r))
print(f"\n{len(CASES)-len(failed)}/{len(CASES)} representative checks passed")
raise SystemExit(1 if failed else 0)
