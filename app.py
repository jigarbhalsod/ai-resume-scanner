"""
app.py — Main Streamlit Application
=====================================
This is the entry point for the Resume Scanner web app.

How Streamlit works:
- The script runs TOP TO BOTTOM every time ANYTHING changes
  (file upload, button click, tab switch, dropdown selection)
- session_state is a dict that persists across these reruns
- @st.cache_resource prevents expensive objects (spaCy models) from
  reloading on every rerun — they load once and stay in memory
- @st.cache_data caches function return values — same input = same output
  returned instantly from cache

Run this app with:
    streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — required for Streamlit

from wordcloud import WordCloud
import io

# Import all five analysis modules
from resume_scanner import ResumeParser, NLPEngine, ATSScorer, AIDetector, JobMatcher

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the FIRST Streamlit call in the script
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Scanner",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CACHED RESOURCE INITIALISATION
# @st.cache_resource: runs ONCE, result stays in memory across all reruns
# Without this, spaCy would reload its neural model on every interaction
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_nlp_engine():
    return NLPEngine(use_spacy=True)

@st.cache_resource
def load_ats_scorer():
    return ATSScorer()

@st.cache_resource
def load_ai_detector():
    return AIDetector()

@st.cache_resource
def load_job_matcher():
    return JobMatcher()

# Initialise all engines once
nlp_engine  = load_nlp_engine()
ats_scorer  = load_ats_scorer()
ai_detector = load_ai_detector()
job_matcher = load_job_matcher()
parser      = ResumeParser()  # Lightweight — no model to load

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — minimal styling for a clean, professional look
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Score display boxes */
    .score-box {
        text-align: center;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
    }
    .score-green  { background: #d4edda; color: #155724; }
    .score-yellow { background: #fff3cd; color: #856404; }
    .score-red    { background: #f8d7da; color: #721c24; }

    /* Skill tag pills */
    .skill-tag {
        display: inline-block;
        background: #e8f4fd;
        color: #1a6e9e;
        padding: 3px 10px;
        border-radius: 20px;
        margin: 3px;
        font-size: 0.85em;
        font-weight: 500;
    }

    /* Section header underline style */
    .section-header {
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Resume Scanner")
    st.markdown("*AI-powered resume analysis*")
    st.divider()

    # File uploader — accepts PDF and DOCX
    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT",
    )

    # Target role selector
    role_options = {
        "Auto-detect":        None,
        "Data Scientist":     "data_scientist",
        "ML Engineer":        "ml_engineer",
        "Data Analyst":       "data_analyst",
        "Software Engineer":  "software_engineer",
        "Data Engineer":      "data_engineer",
        "Frontend Developer": "frontend_developer",
        "Backend Developer":  "backend_developer",
    }
    selected_role_label = st.selectbox(
        "Target Role",
        options=list(role_options.keys()),
        help="Select your target role for more accurate ATS keyword scoring",
    )
    target_role = role_options[selected_role_label]

    st.divider()

    # Analyse button
    analyse_btn = st.button("🔍 Analyse Resume", use_container_width=True, type="primary")

    st.divider()
    st.caption("Built with spaCy · scikit-learn · Streamlit")
    st.caption("v1.0.0 · Jigar Bhalsod")

# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS — triggered when button clicked AND file is uploaded
# Results stored in session_state so they survive tab switches (reruns)
# ──────────────────────────────────────────────────────────────────────────────
if analyse_btn and uploaded_file is not None:
    with st.spinner("Analysing your resume..."):
        try:
            # Read file bytes and detect extension
            file_bytes = uploaded_file.read()
            file_ext   = "." + uploaded_file.name.split(".")[-1].lower()

            # Step 1: Parse file → plain text
            resume_text = parser.parse(file_content=file_bytes, file_type=file_ext)

            if not resume_text or len(resume_text.strip()) < 50:
                st.error("Could not extract text from this file. Try a different format.")
                st.stop()

            # Step 2: Run all four analysis modules
            skills_data  = nlp_engine.get_skill_summary(resume_text)
            exp_years    = nlp_engine.calculate_experience_years(resume_text)
            text_quality = nlp_engine.analyze_text_quality(resume_text)
            ats_results  = ats_scorer.calculate_score(resume_text, target_role)
            ai_results   = ai_detector.analyze(resume_text)
            job_results  = job_matcher.match(resume_text)
            all_skills   = nlp_engine.get_all_skills_flat(resume_text)

            # Step 3: Store everything in session_state
            # session_state survives reruns — so tab switches don't re-analyse
            st.session_state["results"] = {
                "resume_text":  resume_text,
                "skills_data":  skills_data,
                "exp_years":    exp_years,
                "text_quality": text_quality,
                "ats_results":  ats_results,
                "ai_results":   ai_results,
                "job_results":  job_results,
                "all_skills":   all_skills,
                "filename":     uploaded_file.name,
            }
            st.success(f"✅ Analysis complete for **{uploaded_file.name}**")

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.stop()

elif analyse_btn and uploaded_file is None:
    st.sidebar.warning("Please upload a resume file first.")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT AREA
# ──────────────────────────────────────────────────────────────────────────────

if "results" not in st.session_state:
    # ── Welcome screen — shown when no analysis has been run yet ──────────────
    st.title("📄 Resume Scanner")
    st.markdown("### AI-powered resume analysis in seconds")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("🧠 **Skill Extraction**\n\nIdentifies 200+ skills across 7 categories using NLP")
    with col2:
        st.info("📊 **ATS Scoring**\n\nChecks compatibility with Applicant Tracking Systems")
    with col3:
        st.info("🤖 **AI Detection**\n\nDetects if your resume was AI-generated")
    with col4:
        st.info("🎯 **Job Matching**\n\nMatches your resume to 6 job roles using TF-IDF")

    st.divider()
    st.markdown("""
    **How to use:**
    1. Upload your resume (PDF, DOCX, or TXT) in the sidebar
    2. Select your target role (or leave on Auto-detect)
    3. Click **Analyse Resume**
    4. Explore results across the four tabs
    """)

else:
    # ── Results screen — shown after analysis ─────────────────────────────────
    r = st.session_state["results"]  # shorthand

    # Top-level summary metrics
    st.title(f"📄 {r['filename']}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Skills",     r["skills_data"]["total_skills"])
    m2.metric("ATS Score",        f"{r['ats_results']['overall_score']}/100")
    m3.metric("AI Probability",   f"{r['ai_results']['ai_probability']}%")
    m4.metric("Best Job Match",   f"{r['job_results']['best_match']} ({r['job_results']['best_score']}%)")

    st.divider()

    # ── Four analysis tabs ────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧠 Skills Analysis",
        "📊 ATS Score",
        "🤖 AI Detection",
        "🎯 Job Match",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — SKILLS ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        skills_data = r["skills_data"]
        all_skills  = r["all_skills"]

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.metric("Total Skills Found", skills_data["total_skills"])
            st.metric("Years of Experience", f"{r['exp_years']} years")
            st.metric("Word Count", r["text_quality"].get("word_count", "N/A"))

            st.markdown("#### Skills by Category")

            # Friendly display names for categories
            category_labels = {
                "programming_languages": "💻 Programming Languages",
                "frameworks_libraries":  "📦 Frameworks & Libraries",
                "data_science_tools":    "🔬 Data Science Tools",
                "databases":             "🗄️ Databases",
                "cloud_devops":          "☁️ Cloud & DevOps",
                "soft_skills":           "🤝 Soft Skills",
                "ml_ai_concepts":        "🤖 ML/AI Concepts",
            }

            for cat_key, cat_label in category_labels.items():
                skills_in_cat = skills_data["skills_by_category"].get(cat_key, [])
                if skills_in_cat:
                    with st.expander(f"{cat_label} ({len(skills_in_cat)})"):
                        # Render each skill as a styled pill/tag
                        tags_html = "".join(
                            f'<span class="skill-tag">{skill}</span>'
                            for skill in skills_in_cat
                        )
                        st.markdown(tags_html, unsafe_allow_html=True)

        with col_right:
            # Bar chart — skills per category
            cat_counts = skills_data["category_counts"]
            if cat_counts:
                labels = [category_labels.get(k, k).split(" ", 1)[1] for k in cat_counts.keys()]
                values = list(cat_counts.values())

                fig_bar = go.Figure(go.Bar(
                    x=values,
                    y=labels,
                    orientation="h",
                    marker_color="#4C9BE8",
                    text=values,
                    textposition="outside",
                ))
                fig_bar.update_layout(
                    title="Skills Per Category",
                    xaxis_title="Count",
                    height=380,
                    margin=dict(l=20, r=40, t=40, b=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # Word cloud
            if all_skills:
                st.markdown("#### Skills Word Cloud")
                wc_text = " ".join(all_skills)
                try:
                    wc = WordCloud(
                        width=600,
                        height=300,
                        background_color="white",
                        colormap="Blues",
                        max_words=80,
                        prefer_horizontal=0.8,
                    ).generate(wc_text)

                    fig_wc, ax = plt.subplots(figsize=(8, 4))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    plt.tight_layout(pad=0)
                    st.pyplot(fig_wc)
                    plt.close(fig_wc)
                except Exception:
                    st.info("Word cloud could not be generated.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — ATS SCORE
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        ats = r["ats_results"]
        overall = ats["overall_score"]

        col_score, col_radar = st.columns([1, 1.5])

        with col_score:
            # Colour-coded score display
            if overall >= 75:
                score_class = "score-green"
                emoji = "✅"
            elif overall >= 50:
                score_class = "score-yellow"
                emoji = "⚠️"
            else:
                score_class = "score-red"
                emoji = "❌"

            st.markdown(f"""
            <div class="score-box {score_class}">
                <h1 style="margin:0; font-size: 3em;">{overall}</h1>
                <h2 style="margin:0;">/ 100</h2>
                <h3 style="margin:8px 0 0 0;">Grade: {ats['grade']} {emoji}</h3>
            </div>
            """, unsafe_allow_html=True)

            # Pass/Fail indicator
            if ats["pass_ats"]:
                st.success("✅ Likely to PASS ATS screening")
            else:
                st.error("❌ May FAIL ATS screening — see suggestions below")

            st.markdown(f"**Detected Role:** `{ats['detected_role'].replace('_', ' ').title()}`")

            st.divider()

            # Sub-scores breakdown table
            st.markdown("#### Sub-Score Breakdown")
            sub = ats["sub_scores"]
            for name, score in sub.items():
                colour = "green" if score >= 75 else "orange" if score >= 50 else "red"
                st.markdown(
                    f"**{name.title()}**: "
                    f"<span style='color:{colour}; font-weight:bold;'>{score}/100</span>",
                    unsafe_allow_html=True,
                )
                st.progress(score / 100)

        with col_radar:
            # Radar chart showing all 6 sub-scores
            sub_scores = ats["sub_scores"]
            categories = [k.title() for k in sub_scores.keys()]
            values     = list(sub_scores.values())
            # Close the radar shape by repeating first value
            values_closed     = values + [values[0]]
            categories_closed = categories + [categories[0]]

            fig_radar = go.Figure(go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                fillcolor="rgba(76, 155, 232, 0.2)",
                line=dict(color="#4C9BE8", width=2),
                name="ATS Scores",
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100]),
                ),
                title="ATS Score Radar",
                height=400,
                margin=dict(l=40, r=40, t=60, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Feedback list
        st.divider()
        st.markdown("#### Detailed Feedback")
        feedback = ats.get("feedback", [])
        for item in feedback:
            st.markdown(f"- {item}")

        # Improvement suggestions
        st.divider()
        st.markdown("#### 🔧 Improvement Suggestions")
        suggestions = ats_scorer.get_improvement_suggestions()
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — AI DETECTION
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        ai = r["ai_results"]
        prob = ai["ai_probability"]

        col_gauge, col_details = st.columns([1, 1])

        with col_gauge:
            # Gauge chart — Plotly indicator
            # Colour: red=AI, yellow=mixed, green=human
            if prob >= 70:
                gauge_color = "#dc3545"
                verdict_color = "red"
            elif prob >= 45:
                gauge_color = "#ffc107"
                verdict_color = "orange"
            else:
                gauge_color = "#28a745"
                verdict_color = "green"

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                title={"text": "AI Probability", "font": {"size": 18}},
                number={"suffix": "%", "font": {"size": 36}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": gauge_color},
                    "steps": [
                        {"range": [0, 25],  "color": "#d4edda"},  # green zone
                        {"range": [25, 45], "color": "#fff3cd"},  # yellow-green
                        {"range": [45, 70], "color": "#ffeeba"},  # yellow
                        {"range": [70, 100],"color": "#f8d7da"},  # red zone
                    ],
                    "threshold": {
                        "line":  {"color": "black", "width": 3},
                        "thickness": 0.8,
                        "value": prob,
                    },
                },
            ))
            fig_gauge.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Verdict and confidence
            st.markdown(
                f"<h3 style='color:{verdict_color}; text-align:center;'>"
                f"{ai['verdict']}</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='text-align:center;'>Confidence: "
                f"<strong>{ai['confidence']}</strong></p>",
                unsafe_allow_html=True,
            )

        with col_details:
            st.markdown("#### Detection Scores")

            # Score table
            score_data = {
                "Check": [
                    "AI Phrases Detected",
                    "Overused Verbs",
                    "Vocabulary Diversity (TTR)",
                    "Sentence Repetition",
                ],
                "Score": [
                    f"{ai['phrase_score']}%",
                    f"{ai['verb_score']}%",
                    f"{ai['ttr_score']}% (human)",
                    f"{ai['repetition_score']}%",
                ],
                "Signal": [
                    "🔴 High" if ai['phrase_score'] >= 70 else "🟡 Medium" if ai['phrase_score'] >= 40 else "🟢 Low",
                    "🔴 High" if ai['verb_score'] >= 70 else "🟡 Medium" if ai['verb_score'] >= 40 else "🟢 Low",
                    "🟢 Human" if ai['ttr_score'] >= 60 else "🟡 Mixed" if ai['ttr_score'] >= 40 else "🔴 AI",
                    "🔴 High" if ai['repetition_score'] >= 70 else "🟡 Medium" if ai['repetition_score'] >= 40 else "🟢 Low",
                ],
            }
            st.dataframe(
                pd.DataFrame(score_data),
                hide_index=True,
                use_container_width=True,
            )

            # Flags
            st.divider()
            st.markdown("#### 🚩 Detected Flags")
            flags = ai.get("flags", [])
            if flags:
                for flag in flags:
                    st.warning(flag)
            else:
                st.success("No AI writing patterns detected.")

            # What this means
            st.divider()
            st.markdown("#### ℹ️ What This Means")
            if prob >= 70:
                st.error(
                    "Your resume shows strong AI-generated writing patterns. "
                    "Recruiters increasingly screen for this. Consider rewriting "
                    "key sections in your own authentic voice."
                )
            elif prob >= 45:
                st.warning(
                    "Your resume shows some AI-assisted patterns. "
                    "This is common when using AI to polish writing. "
                    "Review flagged phrases and personalise where possible."
                )
            else:
                st.success(
                    "Your resume reads as human-written. "
                    "Good use of natural language and varied vocabulary."
                )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — JOB MATCH
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        job = r["job_results"]

        col_best, col_chart = st.columns([1, 1.5])

        with col_best:
            st.metric(
                "Best Match",
                job["best_match"],
                delta=f"{job['best_score']}% similarity",
            )

            st.divider()
            st.markdown("#### 💡 Recommendations")
            for rec in job["recommendations"]:
                st.markdown(f"- {rec}")

            st.divider()

            # Keyword gap analysis for best match
            kw_analysis = job.get("keyword_analysis", {})
            present = kw_analysis.get("present", [])
            missing = kw_analysis.get("missing", [])

            if present:
                st.markdown(f"#### ✅ Keywords Present ({len(present)})")
                tags_present = "".join(
                    f'<span class="skill-tag" style="background:#d4edda; color:#155724;">{kw}</span>'
                    for kw in present
                )
                st.markdown(tags_present, unsafe_allow_html=True)

            if missing:
                st.divider()
                st.markdown(f"#### ❌ Missing Keywords ({len(missing)})")
                tags_missing = "".join(
                    f'<span class="skill-tag" style="background:#f8d7da; color:#721c24;">{kw}</span>'
                    for kw in missing[:15]
                )
                st.markdown(tags_missing, unsafe_allow_html=True)
                st.caption("Consider adding these to your resume if they match your experience")

        with col_chart:
            # Horizontal bar chart — all roles ranked by match %
            all_matches = job["all_matches"]
            roles       = [m["role"] for m in reversed(all_matches)]
            scores      = [m["percentage"] for m in reversed(all_matches)]

            # Colour bars by match quality
            bar_colors = []
            for s in scores:
                if s >= 70:   bar_colors.append("#28a745")   # green
                elif s >= 50: bar_colors.append("#ffc107")   # yellow
                else:         bar_colors.append("#dc3545")   # red

            fig_match = go.Figure(go.Bar(
                x=scores,
                y=roles,
                orientation="h",
                marker_color=bar_colors,
                text=[f"{s}%" for s in scores],
                textposition="outside",
            ))
            fig_match.update_layout(
                title="Job Role Match Scores",
                xaxis=dict(title="Match %", range=[0, 105]),
                height=380,
                margin=dict(l=20, r=60, t=50, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            # Add a vertical line at 70% (strong match threshold)
            fig_match.add_vline(
                x=70,
                line_dash="dash",
                line_color="gray",
                annotation_text="Strong match",
                annotation_position="top",
            )
            st.plotly_chart(fig_match, use_container_width=True)
