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
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Styling ----------
st.markdown("""
<style>
.block-container {
    max-width: 1160px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 26px 30px;
    border: 1px solid #293a45;
    border-radius: 22px;
    background: linear-gradient(115deg, #142a2d 0%, #111a24 58%, #191b2a 100%);
    margin-bottom: 22px;
}
.hero-title {
    font-size: 2.35rem;
    font-weight: 800;
    letter-spacing: -1.2px;
    margin: 0;
    color: #edf3f6;
}
.hero-title span { color: #77d6c5; }
.hero-sub {
    color: #b0bec7;
    margin-top: 7px;
    font-size: .98rem;
}
.pill {
    display:inline-block;
    padding: 6px 11px;
    border-radius: 999px;
    font-size: .75rem;
    font-weight: 700;
    margin-top: 14px;
    background: #172a29;
    border: 1px solid #315b55;
    color: #a0e4d7;
}
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
    border-radius: 11px;
}
div[data-testid="stTextArea"] textarea {
    min-height: 145px;
}
.stButton > button {
    border-radius: 10px;
    font-weight: 700;
    min-height: 45px;
}
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}
button[data-baseweb="tab"] { font-weight: 650; }
.demo-box {
    padding: 14px 16px;
    border-radius: 14px;
    background: #15212b;
    border: 1px solid #2a4354;
}
@media (max-width: 800px) {
    .hero-title {font-size: 1.7rem;}
    .hero {padding: 20px;}
}
</style>
""", unsafe_allow_html=True)



# ---------- Product UX helpers ----------
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
    g["group"] = g["category"] + " · " + g["location"]
    return g[["group", "category", "location", "feedback_count"]].head(10)

def score_band(score):
    if score >= .75:
        return "upper score band"
    if score >= .60:
        return "mid score band"
    return "lower score band · review recommended"

def section(title, subtitle=""):
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


# ---------- Hero ----------
st.markdown("""
<div class="hero">
<div class="hero-title"><span>CampusPulse</span></div>
    <div class="hero-sub">Student feedback, organized for thoughtful campus action.</div>
    <div class="pill">PROTOTYPE · MODEL OUTPUTS NEED HUMAN REVIEW</div>
</div>
""", unsafe_allow_html=True)

tab_student, tab_admin = st.tabs([":material/edit_note: Student feedback", ":material/insights: Admin intelligence"])


# ---------- Student ----------
with tab_student:
    left, right = st.columns([1.55, 1], gap="large")

    with left:
        section("Tell us what happened", "Describe the issue in your own words. No special format is required.")
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
        st.caption("Please leave out names and other details that could identify you or someone else.")

        d1, d2 = st.columns([1, 1])
        with d1:
            submitted = st.button("Analyze & submit", type="primary", width="stretch")
        with d2:
            demo = st.button("Load demo example", width="stretch")

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
        with st.container(border=True):
            st.markdown("**01 · Understand**")
            st.caption("Suggest sentiment and a model score.")
            st.markdown("**02 · Classify**")
            st.caption("Identify the campus category.")
            st.markdown("**03 · Connect**")
            st.caption("Find similar wording and possible aspects.")
            st.markdown("**04 · Review**")
            st.caption("Flag uncertain predictions for a person to check.")

    result = st.session_state.get("last_result")

    if result:
        section("Analysis result", "Automated suggestions to help organize this report; scores are not calibrated confidence estimates.")

        sent_conf = result.get("sentiment_confidence", 0)
        cat_conf = result.get("category_confidence", 0)
        sent_text = str(result.get("sentiment", "Unknown")).title()
        cat_conf_text = score_band(cat_conf)

        c1, c2, c3 = st.columns(3)
        with c1:
            with st.container(border=True):
                st.metric("Suggested sentiment", sent_text)
                st.caption(f"Model score: {sent_conf:.0%} · uncalibrated")
        with c2:
            with st.container(border=True):
                st.metric("Suggested category", result.get("category", "Unknown"))
                st.caption(f"Model score: {cat_conf:.0%} · {cat_conf_text}")
        with c3:
            with st.container(border=True):
                st.metric("Possible text matches", result.get("similar_issue_count", 0))
                st.caption("Lexical matches; check whether they describe the same issue.")

        if result.get("review_status") == "Needs review":
            st.warning("Review recommended · the model flagged this result for attention.")
            reasons = result.get("review_reasons", [])
            if reasons:
                st.caption(" · ".join(reasons))
        else:
            st.info("No review flag was raised by the current prototype. This does not confirm the prediction is correct.")

        if result.get("hybrid_adjusted"):
            st.info("🧠 Context-aware language checks were applied to conversational wording.")

        if result.get("sentiment_cues"):
            st.caption("Detected language cues: " + ", ".join(dict.fromkeys(result["sentiment_cues"])))

        if result.get("aspects"):
            section("Fine-grained aspects", "Additional detail from the EduRABSA-trained aspect model.")
            cols = st.columns(min(4, max(1, len(result["aspects"]))))
            for col, item in zip(cols, result["aspects"]):
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{item['aspect']}**")
                        st.metric("Aspect model score", f'{item["confidence"]:.0%}')

        if result.get("similar_issues"):
            section("Possible text matches", "Similarity finds overlapping wording. Review the examples before grouping them as one issue.")
            for item in result["similar_issues"][:3]:
                src = "Student submission" if item.get("source") == "live" else "Synthetic demo"
                with st.container(border=True):
                    st.caption(f"{src} · similarity {float(item.get('similarity', 0)):.2f}")
                    st.markdown(f"**{item.get('category', 'Uncategorized')} · {item.get('location', 'Location not specified')}**")
                    st.write(item.get("feedback", ""))
        elif str(result.get("sentiment", "")).lower() == "negative":
            st.info("No sufficiently similar complaint was found yet. Matching becomes more useful as live feedback grows.")


# ---------- Admin ----------
with tab_admin:
    live_df = get_feedback("live")
    synthetic_df = get_feedback("synthetic_demo")

    data_mode = st.segmented_control(
        "Dashboard data",
        options=["Synthetic demo", "Live submissions"],
        default="Synthetic demo",
        key="admin_data_mode",
    )
    using_synthetic = data_mode == "Synthetic demo"
    dataset_df = synthetic_df.copy() if using_synthetic else live_df.copy()

    if using_synthetic:
        st.info(
            f"Synthetic demo data · {len(synthetic_df):,} fictional records. These are examples for the walkthrough, not student submissions.",
            icon=":material/science:",
        )
    else:
        st.caption("Live student submissions only. Synthetic demo records are not included.")

    if dataset_df.empty:
        st.info("No records are available in this view yet.")
    else:
        section("Explore feedback", "Filter the current dataset, compare its labels, and inspect report examples.")

        filter_cols = st.columns([1.1, 1.1, 1, 1.5])
        categories_list = ["All categories"] + sorted(dataset_df["category"].dropna().astype(str).unique().tolist())
        locations_list = ["All locations"] + sorted(dataset_df["location"].dropna().astype(str).unique().tolist())
        sentiments_list = ["All sentiments", "Negative", "Neutral", "Positive"]
        with filter_cols[0]:
            selected_category = st.selectbox("Category", categories_list, key="dashboard_category")
        with filter_cols[1]:
            selected_location = st.selectbox("Location", locations_list, key="dashboard_location")
        with filter_cols[2]:
            selected_sentiment = st.selectbox("Sentiment", sentiments_list, key="dashboard_sentiment")
        with filter_cols[3]:
            search_text = st.text_input("Search feedback", placeholder="Try: Wi-Fi, exams, food…", key="dashboard_search")

        filtered = dataset_df.copy()
        if selected_category != "All categories":
            filtered = filtered[filtered["category"].astype(str) == selected_category]
        if selected_location != "All locations":
            filtered = filtered[filtered["location"].astype(str) == selected_location]
        if selected_sentiment != "All sentiments":
            filtered = filtered[filtered["sentiment"].astype(str).str.lower() == selected_sentiment.lower()]
        if search_text.strip():
            filtered = filtered[filtered["feedback"].astype(str).str.contains(search_text.strip(), case=False, regex=False, na=False)]

        negative_count = int((filtered["sentiment"].str.lower() == "negative").sum()) if not filtered.empty else 0
        neutral_count = int((filtered["sentiment"].str.lower() == "neutral").sum()) if not filtered.empty else 0
        positive_count = int((filtered["sentiment"].str.lower() == "positive").sum()) if not filtered.empty else 0
        sentiment_word = "labels" if using_synthetic else "suggested labels"

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Reports in view", f"{len(filtered):,}", border=True)
        k2.metric("Negative", f"{negative_count:,}", border=True, help=f"Model-assigned {sentiment_word}; review before taking action.")
        k3.metric("Neutral", f"{neutral_count:,}", border=True)
        k4.metric("Positive", f"{positive_count:,}", border=True)
        st.caption(f"Showing {len(filtered):,} of {len(dataset_df):,} records in the selected dataset.")

        if filtered.empty:
            st.info("No reports match these filters. Change a filter or clear the search text.")
        else:
            section("Overview", "Counts describe the selected dataset and filters; they are not campus-wide rates.")
            chart_left, chart_right = st.columns(2)
            with chart_left:
                st.markdown("**Sentiment labels**")
                st.bar_chart(filtered["sentiment"].str.title().value_counts())
            with chart_right:
                st.markdown("**Feedback categories**")
                st.bar_chart(filtered["category"].value_counts().head(10))

            section("Weekly negative report counts", "Descriptive counts only. Changes in response volume can change these totals.")
            tr = trend_data(filtered)
            if not tr.empty and len(tr) >= 4 and int(tr["Negative feedback"].sum()) >= 5:
                st.line_chart(tr, height=260)
            elif not tr.empty:
                st.info("There is not enough history in this view to draw a trend yet.")
            else:
                st.caption("No negative reports in the selected view.")

            section("Most common negative report groups", "Grouped by model-assigned category and location; these are not confirmed issue clusters.")
            groups = issue_clusters(filtered)
            if not groups.empty:
                st.dataframe(groups, width="stretch", hide_index=True)
            else:
                st.caption("No negative reports in the selected view.")

        section("Aspect themes", "Themes returned by the aspect model; treat them as suggestions.")
        if not filtered.empty and "aspects_json" in filtered.columns:
            aspect_rows = []
            for raw in filtered["aspects_json"].dropna():
                try:
                    for item in json.loads(raw):
                        if item.get("aspect"):
                            aspect_rows.append(item["aspect"])
                except Exception:
                    continue
            if aspect_rows:
                st.bar_chart(pd.Series(aspect_rows).value_counts().head(10))
            else:
                st.caption("No aspect labels are available for these records.")
        else:
            st.caption("Aspect data is not available in the current view.")

        section("Feedback explorer", "Search and filters above apply here too.")
        display_cols = [c for c in ["created_at", "feedback", "sentiment", "category", "location", "review_status"] if c in filtered.columns]
        st.dataframe(filtered[display_cols].head(100), width="stretch", hide_index=True)
        export_name = "campuspulse_synthetic_demo.csv" if using_synthetic else "campuspulse_live_feedback.csv"
        st.download_button(
            "Download filtered reports",
            filtered[display_cols].to_csv(index=False).encode("utf-8"),
            export_name,
            "text/csv",
        )

    if not using_synthetic and not live_df.empty and "review_status" in live_df.columns:
        # Keep raw review flags visible only for actual live submissions.
        review_count = int((filtered["review_status"].astype(str).str.lower() == "needs review").sum()) if "review_status" in filtered.columns else 0
        st.caption(f"Live submissions flagged for review: {review_count:,}. This is a model flag, not a completed review queue.")

st.markdown("""
        <div style="margin-top:35px;padding-top:14px;border-top:1px solid #293a45;
color:#9aaab3;font-size:.75rem;text-align:center;">
CampusPulse · Prototype decision support · automated outputs require human review.
</div>
""", unsafe_allow_html=True)
