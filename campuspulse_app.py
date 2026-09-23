import os
import sqlite3
import json
import html
import pandas as pd
import streamlit as st
from campuspulse_pipeline import analyze_feedback

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "campuspulse.db")

st.set_page_config(
    page_title="CampusPulse",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1420px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stDecoration"] {display:none;}

/* Header */
.hero {
    padding: 24px 28px;
    border: 1px solid rgba(255,255,255,.09);
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(255,75,81,.16), rgba(20,22,31,.75) 55%, rgba(75,80,110,.18));
    margin-bottom: 18px;
}
.hero-title {
    font-size: 2.25rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0;
}
.hero-title span { color: #ff4b51; }
.hero-sub {
    color: #a9adbb;
    margin-top: 7px;
    font-size: .98rem;
}
.pill {
    display:inline-block;
    padding: 5px 10px;
    border-radius: 999px;
    font-size: .75rem;
    font-weight: 700;
    margin-top: 14px;
    background: rgba(55, 210, 135, .12);
    border: 1px solid rgba(55, 210, 135, .25);
    color: #55e39b;
}

/* Cards */
.card {
    background: #171922;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 16px;
    padding: 18px 19px;
    min-height: 100px;
}
.card-label {
    color: #8f94a5;
    font-size: .78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .06em;
}
.card-value {
    font-size: 1.55rem;
    font-weight: 800;
    margin-top: 6px;
}
.card-sub {
    color: #8f94a5;
    font-size: .78rem;
    margin-top: 5px;
}

/* Section headings */
.section {
    margin-top: 26px;
    margin-bottom: 10px;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 750;
}
.section-sub {
    color: #858a9a;
    font-size: .82rem;
    margin-top: 3px;
}

/* Result badges */
.badge {
    display:inline-block;
    padding: 7px 11px;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 700;
    border: 1px solid rgba(255,255,255,.1);
}
.badge-red { background: rgba(255,75,81,.12); color:#ff7277; }
.badge-green { background: rgba(55,210,135,.12); color:#55e39b; }
.badge-yellow { background: rgba(255,193,7,.12); color:#ffd45c; }
.badge-blue { background: rgba(90,140,255,.12); color:#8bb0ff; }

/* Empty / insight blocks */
.insight {
    padding: 15px 17px;
    border-radius: 14px;
    background: #171922;
    border: 1px solid rgba(255,255,255,.08);
}
.insight strong { color:#f1f2f6; }
.muted { color:#858a9a; }

/* Inputs */
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
    border-radius: 12px;
}
div[data-testid="stTextArea"] textarea {
    min-height: 145px;
}

/* Buttons */
.stButton > button {
    border-radius: 11px;
    font-weight: 700;
    min-height: 44px;
}

/* Tables */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 650;
}


/* Product polish */
.kpi-accent { border-top: 3px solid #ff4b51; }
.issue-high { border-left: 4px solid #ff4b51; }
.issue-medium { border-left: 4px solid #ffd45c; }
.issue-watch { border-left: 4px solid #55e39b; }
.demo-box {
    padding: 14px 16px;
    border-radius: 14px;
    background: rgba(90,140,255,.07);
    border: 1px solid rgba(90,140,255,.18);
}
.small-note { color:#747989; font-size:.75rem; }

/* Mobile */
@media (max-width: 800px) {
    .hero-title {font-size: 1.7rem;}
}
</style>
""", unsafe_allow_html=True)



# ---------- Product UX helpers ----------
def priority_level(row):
    s = str(row.get("sentiment", "")).lower()
    conf = float(row.get("sentiment_confidence", 0) or 0)
    count = int(row.get("feedback_count", 1) or 1)
    if s == "negative" and count >= 10:
        return "High"
    if s == "negative" and (count >= 5 or conf >= .75):
        return "Medium"
    return "Watch"

def trend_label(current, previous):
    if previous <= 0:
        return ("↑ New", "badge-red") if current > 0 else ("→ Stable", "badge-yellow")
    change = (current - previous) / previous
    if change >= .25:
        return (f"↑ {change:.0%}", "badge-red")
    if change <= -.25:
        return (f"↓ {abs(change):.0%}", "badge-green")
    return (f"→ {change:+.0%}", "badge-yellow")

def demo_feedback():
    return [
        "The Wi-Fi in C block keeps disconnecting after 7 PM.",
        "The library is too crowded and there are not enough seats.",
        "Lab PCs are extremely slow during practical sessions.",
        "The exam schedule was announced too late.",
        "The mess food quality has improved this week.",
        "The campus buses are frequently late in the morning.",
    ]


# ---------- Data ----------
def db():
    return sqlite3.connect(DB_PATH)

def get_feedback(source=None):
    con = db()
    if source:
        q = "SELECT * FROM feedback WHERE source=? ORDER BY created_at DESC"
        df = pd.read_sql_query(q, con, params=(source,))
    else:
        q = "SELECT * FROM feedback ORDER BY created_at DESC"
        df = pd.read_sql_query(q, con)
    con.close()
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df

def trend_data(df):
    if df.empty:
        return pd.DataFrame()
    x = df[df["sentiment"].str.lower() == "negative"].copy()
    if x.empty:
        return pd.DataFrame()
    x["week"] = pd.to_datetime(x["created_at"]).dt.to_period("W").apply(lambda p: p.start_time)
    return x.groupby("week").size().rename("Negative feedback").to_frame()

def issue_clusters(df):
    x = df[df["sentiment"].str.lower() == "negative"].copy()
    if x.empty:
        return pd.DataFrame()
    x["location"] = x["location"].fillna("Campus").replace("", "Campus")
    g = (
        x.groupby(["category", "location"], dropna=False)
        .size()
        .reset_index(name="feedback_count")
        .sort_values("feedback_count", ascending=False)
    )
    g["issue_name"] = g["category"] + " · " + g["location"]
    g["priority"] = g.apply(
        lambda r: "High" if r["feedback_count"] >= 10 else ("Medium" if r["feedback_count"] >= 5 else "Watch"),
        axis=1,
    )
    g["trend"] = "→ Stable"
    return g[["issue_name", "category", "location", "feedback_count", "priority", "trend"]].head(10)

def confidence_class(conf):
    if conf >= .75:
        return "badge-green", "High confidence"
    if conf >= .60:
        return "badge-yellow", "Moderate confidence"
    return "badge-red", "Low confidence · review recommended"

def sentiment_badge(sentiment):
    s = str(sentiment).lower()
    if s == "negative":
        return "badge-red"
    if s == "positive":
        return "badge-green"
    return "badge-yellow"

def metric_card(label, value, sub=""):
    return f"""
    <div class="card">
        <div class="card-label">{html.escape(str(label))}</div>
        <div class="card-value">{html.escape(str(value))}</div>
        <div class="card-sub">{html.escape(str(sub))}</div>
    </div>
    """

def section(title, subtitle=""):
    st.markdown(
        f'<div class="section"><div class="section-title">{html.escape(title)}</div>'
        f'<div class="section-sub">{html.escape(subtitle)}</div></div>',
        unsafe_allow_html=True
    )


# ---------- Hero ----------
st.markdown("""
<div class="hero">
    <div class="hero-title">🎓 <span>CampusPulse</span></div>
    <div class="hero-sub">AI-powered student feedback intelligence · From student voices to actionable campus insights</div>
    <div class="pill">● AI ANALYSIS ACTIVE · V6</div>
</div>
""", unsafe_allow_html=True)

tab_student, tab_admin = st.tabs(["🎓  Student Feedback", "📊  Admin Intelligence"])


# ---------- Student ----------
with tab_student:
    left, right = st.columns([1.55, 1], gap="large")

    with left:
        section("Tell us what happened", "Describe the issue naturally. No special format is required.")
        feedback = st.text_area(
            "Feedback",
            placeholder="Example: The Wi-Fi in C block keeps disconnecting after 7 PM.",
            height=150,
            label_visibility="collapsed",
        )
        location = st.text_input(
            "Location",
            placeholder="Optional · C Block / Main Gate / Library",
            label_visibility="visible",
        )

        d1, d2 = st.columns([1, 1])
        with d1:
            submitted = st.button("✨  Analyze & Submit", type="primary", use_container_width=True)
        with d2:
            demo = st.button("🧪  Demo example", use_container_width=True)

        if demo:
            import random
            st.session_state["demo_feedback"] = random.choice(demo_feedback())
            st.rerun()

        if st.session_state.get("demo_feedback"):
            st.markdown(
                f'<div class="demo-box"><strong>Demo example loaded</strong>'
                f'<div style="margin-top:6px;">{html.escape(st.session_state["demo_feedback"])}</div></div>',
                unsafe_allow_html=True
            )
            if not feedback:
                feedback = st.session_state["demo_feedback"]

        if submitted:
            if not feedback.strip():
                st.warning("Please enter some feedback before submitting.")
            else:
                result = analyze_feedback(
                    feedback.strip(),
                    location=location.strip() or "Not specified",
                    save=True,
                )
                st.session_state["last_result"] = result
                st.success("Feedback analyzed and stored.")

    with right:
        section("What happens next", "CampusPulse converts one sentence into structured insight.")
        st.markdown("""
        <div class="insight">
            <div><span class="badge badge-blue">01</span> <strong>Understand</strong></div>
            <div class="muted" style="margin:7px 0 13px 39px;">Detect sentiment and confidence.</div>
            <div><span class="badge badge-blue">02</span> <strong>Classify</strong></div>
            <div class="muted" style="margin:7px 0 13px 39px;">Identify the campus category.</div>
            <div><span class="badge badge-blue">03</span> <strong>Connect</strong></div>
            <div class="muted" style="margin:7px 0 13px 39px;">Find related complaints and fine-grained aspects.</div>
            <div><span class="badge badge-blue">04</span> <strong>Flag</strong></div>
            <div class="muted" style="margin:7px 0 0 39px;">Surface uncertain predictions for human review.</div>
        </div>
        """, unsafe_allow_html=True)

    result = st.session_state.get("last_result")

    if result:
        section("Analysis result", "AI-generated classification for the submitted feedback.")

        sent_conf = result.get("sentiment_confidence", 0)
        cat_conf = result.get("category_confidence", 0)
        sent_cls = sentiment_badge(result.get("sentiment", ""))
        sent_text = str(result.get("sentiment", "Unknown")).title()
        sent_conf_cls, sent_conf_text = confidence_class(sent_conf)
        cat_conf_cls, cat_conf_text = confidence_class(cat_conf)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(metric_card("Sentiment", sent_text, f"{sent_conf:.0%} confidence"), unsafe_allow_html=True)
            st.markdown(f'<span class="badge {sent_cls}">{sent_text}</span>', unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Category", result.get("category", "Unknown"), f"{cat_conf:.0%} confidence"), unsafe_allow_html=True)
            st.markdown(f'<span class="badge {cat_conf_cls}">{cat_conf_text}</span>', unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("Related issues", result.get("similar_issue_count", 0), "matched complaints"), unsafe_allow_html=True)

        if result.get("review_status") == "Needs review":
            st.warning("⚠️ Low-confidence prediction · human review recommended.")
            reasons = result.get("review_reasons", [])
            if reasons:
                st.caption(" · ".join(reasons))
        else:
            st.success("✓ Prediction confidence is above the review threshold.")

        if result.get("hybrid_adjusted"):
            st.info("🧠 Context-aware language checks were applied to conversational wording.")

        if result.get("sentiment_cues"):
            st.caption("Detected language cues: " + ", ".join(dict.fromkeys(result["sentiment_cues"])))

        if result.get("aspects"):
            section("Fine-grained aspects", "Additional detail from the EduRABSA-trained aspect model.")
            cols = st.columns(min(4, max(1, len(result["aspects"]))))
            for col, item in zip(cols, result["aspects"]):
                with col:
                    st.markdown(metric_card(item["aspect"], f'{item["confidence"]:.0%}', "aspect confidence"), unsafe_allow_html=True)

        if result.get("similar_issues"):
            section("Related complaints", "Potentially connected feedback already seen by CampusPulse.")
            for item in result["similar_issues"]:
                src = "Live" if item.get("source") == "live" else "Demo"
                st.markdown(
                    f'<div class="insight"><span class="badge badge-blue">{html.escape(src)}</span> '
                    f'<strong>{html.escape(str(item.get("category","")))} · {html.escape(str(item.get("location","")))}</strong>'
                    f'<span class="muted"> · similarity {float(item.get("similarity",0)):.2f}</span>'
                    f'<div style="margin-top:8px;">{html.escape(str(item.get("feedback","")))}</div></div>',
                    unsafe_allow_html=True
                )
                st.write("")
        elif str(result.get("sentiment", "")).lower() == "negative":
            st.info("No sufficiently similar complaint was found yet. Matching becomes more useful as live feedback grows.")


# ---------- Admin ----------
with tab_admin:
    all_df = get_feedback()
    live_df = get_feedback("live")
    synthetic_df = get_feedback("synthetic_demo")

    section("Campus intelligence", "A live view of what students are saying across the campus.")

    total = len(all_df)
    live = len(live_df)
    negative = int((all_df["sentiment"].str.lower() == "negative").sum()) if not all_df.empty else 0
    categories = all_df["category"].nunique() if not all_df.empty else 0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(metric_card("All feedback", total, "live + demo records"), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card("Live submissions", live, "student-entered feedback"), unsafe_allow_html=True)
    with m3:
        st.markdown(metric_card("Negative", negative, "feedback needing attention"), unsafe_allow_html=True)
    with m4:
        st.markdown(metric_card("Categories", categories, "campus issue areas"), unsafe_allow_html=True)

    section("Feedback landscape", "Distribution of sentiment and campus categories.")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Sentiment**")
        if not all_df.empty:
            st.bar_chart(all_df["sentiment"].str.title().value_counts(), height=280)
        else:
            st.info("No feedback yet.")
    with right:
        st.markdown("**Categories**")
        if not all_df.empty:
            st.bar_chart(all_df["category"].value_counts(), height=280)
        else:
            st.info("No feedback yet.")

    section("Negative feedback trend", "Weekly volume of negative feedback.")
    tr = trend_data(all_df)
    if not tr.empty:
        st.line_chart(tr, height=300)
    else:
        st.info("No negative feedback yet.")

    section("Top recurring issues", "The areas receiving the most negative feedback right now.")
    clusters = issue_clusters(all_df)
    if not clusters.empty:
        top = clusters.head(3)
        cards = st.columns(3)
        for col, (_, r) in zip(cards, top.iterrows()):
            p = str(r["priority"])
            cls = "issue-high" if p == "High" else ("issue-medium" if p == "Medium" else "issue-watch")
            col.markdown(
                f'<div class="insight {cls}">'
                f'<div class="card-label">{html.escape(str(r["category"]))}</div>'
                f'<div style="font-size:1.08rem;font-weight:750;margin-top:6px;">{html.escape(str(r["location"]))}</div>'
                f'<div style="margin-top:10px;"><strong>{int(r["feedback_count"])}</strong> negative reports</div>'
                f'<div class="small-note" style="margin-top:6px;">Priority: {html.escape(p)} · {html.escape(str(r["trend"]))}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.dataframe(clusters, use_container_width=True, hide_index=True)
    else:
        st.info("No recurring issues yet.")

    section("Fine-grained aspects", "Additional themes detected by the EduRABSA-trained model.")
    if not all_df.empty and "aspects_json" in all_df.columns:
        aspect_rows = []
        for raw in all_df["aspects_json"].dropna():
            try:
                for item in json.loads(raw):
                    if item.get("aspect"):
                        aspect_rows.append(item["aspect"])
            except Exception:
                pass
        if aspect_rows:
            st.bar_chart(pd.Series(aspect_rows).value_counts().head(10), height=300)
        else:
            st.info("Aspect data will appear after V5/V6 live submissions.")
    else:
        st.info("Aspect data is not available yet.")

    section("Live submissions", "Newest student-entered feedback.")
    if not live_df.empty:
        cols = [
            c for c in [
                "created_at", "feedback", "sentiment", "sentiment_confidence",
                "category", "category_confidence", "location", "review_status",
                "aspects_json"
            ] if c in live_df.columns
        ]
        st.dataframe(live_df[cols], use_container_width=True, hide_index=True)
        csv = live_df[cols].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Export live submissions", csv, "campuspulse_live_feedback.csv", "text/csv")
    else:
        st.info("No live submissions yet.")

    section("Issue explorer", "Filter the feedback behind a category or location before taking action.")
    if not all_df.empty:
        f1, f2 = st.columns(2)
        categories_list = ["All"] + sorted(all_df["category"].dropna().astype(str).unique().tolist())
        locations_list = ["All"] + sorted(all_df["location"].dropna().astype(str).unique().tolist())
        with f1:
            selected_category = st.selectbox("Category", categories_list, key="issue_category")
        with f2:
            selected_location = st.selectbox("Location", locations_list, key="issue_location")

        filtered = all_df.copy()
        if selected_category != "All":
            filtered = filtered[filtered["category"].astype(str) == selected_category]
        if selected_location != "All":
            filtered = filtered[filtered["location"].astype(str) == selected_location]

        if filtered.empty:
            st.info("No feedback matches these filters.")
        else:
            show_cols = [c for c in ["created_at","feedback","sentiment","category","location","review_status"] if c in filtered.columns]
            st.dataframe(filtered[show_cols].head(25), use_container_width=True, hide_index=True)
    else:
        st.info("Submit feedback to populate the issue explorer.")

    section("Demo data", "Synthetic records used to demonstrate the dashboard. They are not real student opinions.")
    st.metric("Synthetic records", len(synthetic_df))

st.markdown("""
<div style="margin-top:35px;padding-top:14px;border-top:1px solid rgba(255,255,255,.07);
color:#666b79;font-size:.72rem;text-align:center;">
CampusPulse · AI-powered student feedback intelligence · V6 · AI outputs are decision-support, not automatic disciplinary decisions.
</div>
""", unsafe_allow_html=True)
