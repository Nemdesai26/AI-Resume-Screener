"""
ATS Resume Screener — app.py
Run: streamlit run app.py
"""

import os, re, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

import fitz
from docx import Document as DocxDoc
from sentence_transformers import SentenceTransformer
import nltk
from nltk.corpus import stopwords
import google.generativeai as genai

st.set_page_config(page_title="ATS Resume Screener", page_icon="🎯",
                   layout="wide", initial_sidebar_state="collapsed")
nltk.download("stopwords", quiet=True)

GEMINI_API_KEY  = "AIzaSyCGr4ucdAeI-y5HMVu44kD8AHtngsJkxuY"
SCORE_THRESHOLD = 60
PENALTY_MAX     = 0.40

LABEL_WEIGHTS = {
    "sbert_similarity"    : 0.35,
    "jaccard_skills"      : 0.40,
    "keyword_match_ratio" : 0.25,
}

SKILL_KEYWORDS = {
    "programming":  ["python", "java"],
    "ml_ai":        ["machine learning", "tensorflow"],
}
ALL_SKILLS = [s for grp in SKILL_KEYWORDS.values() for s in grp]

@st.cache_resource(show_spinner="Loading models...")
def load_models():
    stop_words = set(stopwords.words("english"))
    sbert      = SentenceTransformer("all-MiniLM-L6-v2")
    xgb = None
    for model_name in ["ats_xgb_v3.joblib", "ats_xgboost.joblib", "xgb_scorer.joblib"]:
        model_path = os.path.join("models", model_name)
        if os.path.exists(model_path):
            try:
                import joblib
                xgb = joblib.load(model_path)
                break
            except Exception:
                pass
    return sbert, stop_words, xgb

sbert_model, STOP_WORDS, xgb_model = load_models()
using_xgb = xgb_model is not None

def clean_text(text, remove_stopwords=False):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+",        " ", text)
    text = re.sub(r"\S+@\S+",                  " ", text)
    text = re.sub(r"\+?\d[\d\s\-().]{7,}\d"," ", text)
    text = re.sub(r"[^a-z0-9\s+#./]",           " ", text)
    text = re.sub(r"\s+",                        " ", text).strip()
    if remove_stopwords:
        text = " ".join(t for t in text.split() if t not in STOP_WORDS)
    return text

def extract_skills(text):
    tl    = text.lower()
    found = set()
    for skill in ALL_SKILLS:
        if " " in skill:
            if skill in tl:
                found.add(skill)
        else:
            if re.search(r"\b" + re.escape(skill) + r"\b", tl):
                found.add(skill)
    return found

def compute_features(resume_text, jd_text):
    embs  = sbert_model.encode([clean_text(resume_text), clean_text(jd_text)],
                                normalize_embeddings=True, show_progress_bar=False)
    sbert    = float(np.clip(np.dot(embs[0], embs[1]), 0.0, 1.0))
    r_skills = extract_skills(resume_text)
    j_skills = extract_skills(jd_text)
    jacc = 0.5 if not j_skills else (0.0 if not (r_skills | j_skills) else
           len(r_skills & j_skills) / len(r_skills | j_skills))
    j_tokens = set(clean_text(jd_text,    remove_stopwords=True).split())
    r_tokens = set(clean_text(resume_text, remove_stopwords=True).split())
    keywords = {t for t in j_tokens if len(t) >= 4}
    kwmr     = len(keywords & r_tokens) / len(keywords) if keywords else 0.0
    return {"sbert_similarity": round(sbert,4), "jaccard_skills": round(jacc,4), "keyword_match_ratio": round(kwmr,4)}

def score_from_features(features):
    if using_xgb:
        X = np.array([[features[k] for k in LABEL_WEIGHTS]], dtype=np.float32)
        return float(np.clip(xgb_model.predict(X)[0], 0.0, 1.0))
    return sum(LABEL_WEIGHTS[k] * features[k] for k in LABEL_WEIGHTS)

def adaptive_penalty(resume_text, jd_text):
    r_skills  = extract_skills(resume_text)
    j_skills  = extract_skills(jd_text)
    if not j_skills: return 1.0, set(), set(), 0.0
    missing   = j_skills - r_skills
    matched   = j_skills & r_skills
    gap_ratio = len(missing) / len(j_skills)
    return 1.0 - PENALTY_MAX * (gap_ratio ** 2), missing, matched, gap_ratio

def extract_from_upload(f):
    ext = Path(f.name).suffix.lower()
    try:
        if ext == ".pdf":
            doc = fitz.open(stream=f.read(), filetype="pdf")
            return " ".join(p.get_text("text") for p in doc).strip()
        elif ext == ".docx":
            return " ".join(p.text for p in DocxDoc(f).paragraphs).strip()
        return f.read().decode("utf-8", errors="ignore").strip()
    except Exception as e:
        st.error(f"File read error: {e}"); return ""

def get_llm_suggestions(resume_text, jd_text, score, missing_skills):
    missing_str = ", ".join(sorted(missing_skills)) if missing_skills else "none"
    prompt = (f"ATS Score: {score}/100\nMissing skills: {missing_str}\n"
              f"JD: {jd_text[:2000]}\nResume: {resume_text[:3000]}\n"
              "Return ONLY valid JSON: overall_analysis, missing_keywords, priority_actions, suggestions list.")
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp  = model.generate_content(prompt,
            generation_config=genai.GenerationConfig(temperature=0.25, max_output_tokens=1500))
        raw = re.sub(r"^```(?:json)?\s*", "", resp.text.strip())
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"overall_analysis": resp.text, "suggestions": [], "priority_actions": []}
    except Exception as e:
        return {"overall_analysis": f"Gemini error: {e}", "suggestions": [], "priority_actions": []}

st.markdown("""<style>
    .skill-tag-green { display:inline-block; background:#1e4d2b; color:#2ecc71;
                       border-radius:6px; padding:3px 10px; margin:3px; }
    .skill-tag-red   { display:inline-block; background:#4d1e1e; color:#e74c3c;
                       border-radius:6px; padding:3px 10px; margin:3px; }
    .metric-box      { background:#1a1a2e; border-radius:10px; padding:12px 18px; margin-bottom:8px; }
</style>""", unsafe_allow_html=True)
st.title("🎯 ATS Resume Screener")
mode_label = "XGBoost + Adaptive Penalty" if using_xgb else "Weighted Formula + Adaptive Penalty"
st.caption(f"SBERT · Jaccard Skills · Keyword Match · {mode_label} · Gemini 1.5 Flash")
if not using_xgb:
    st.warning("⚠️ XGBoost model not found in models/ — run Cell G in the notebook to train it. "
               "Using weighted formula as fallback.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("📄 Resume")
    tab_up, tab_paste = st.tabs(["Upload File", "Paste Text"])
    resume_text = ""
    with tab_up:
        uploaded = st.file_uploader("PDF / DOCX / TXT", type=["pdf","docx","txt"], label_visibility="collapsed")
        if uploaded:
            resume_text = extract_from_upload(uploaded)
            if resume_text: st.success(f"✅ Extracted {len(resume_text):,} characters")
    with tab_paste:
        pasted = st.text_area("Paste resume here", height=280, label_visibility="collapsed")
        if pasted.strip(): resume_text = pasted.strip()
with col2:
    st.subheader("💼 Job Description")
    jd_text = st.text_area("Paste JD here", height=330, label_visibility="collapsed")

if st.button("🚀 Analyze Resume", type="primary", use_container_width=True):
    if not resume_text:
        st.error("❌ Please upload or paste a resume first."); st.stop()
    if not jd_text or len(jd_text.strip()) < 50:
        st.error("❌ Please paste a job description (min 50 chars)."); st.stop()

    with st.spinner("Analysing resume..."):
        features    = compute_features(resume_text, jd_text)
        raw_score   = round(score_from_features(features) * 100, 1)
        pf, missing_skills, matched_skills, gap_ratio = adaptive_penalty(resume_text, jd_text)
        final_score = round(raw_score * pf, 1)

    if final_score >= 75:   verdict, v_icon, score_color = "Strong Match",   "✅", "#2ecc71"
    elif final_score >= SCORE_THRESHOLD: verdict, v_icon, score_color = "Moderate Match", "⚠️", "#f39c12"
    else:                   verdict, v_icon, score_color = "Weak Match",     "❌", "#e74c3c"

    st.divider()
    r1, r2, r3 = st.columns([1.1, 1.1, 0.8])

    with r1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=final_score,
            title={"text": "ATS Score", "font": {"size": 17}},
            number={"font": {"size": 48, "color": score_color}},
            gauge={"axis": {"range": [0,100]}, "bar": {"color": score_color, "thickness": 0.25},
                   "bgcolor": "rgba(0,0,0,0)",
                   "steps": [{"range": [0, SCORE_THRESHOLD], "color": "rgba(231,76,60,0.12)"},
                             {"range": [SCORE_THRESHOLD,75], "color": "rgba(243,156,18,0.12)"},
                             {"range": [75,100],             "color": "rgba(46,204,113,0.12)"}],
                   "threshold": {"line": {"color":"red","width":3}, "thickness":0.75, "value":SCORE_THRESHOLD}}
        ))
        gauge.update_layout(height=270, paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=30,b=0,l=20,r=20))
        st.plotly_chart(gauge, use_container_width=True)
        st.markdown(f"<h3 style='text-align:center;color:{score_color};margin-top:-10px'>"
                    f"{verdict} {v_icon}</h3>", unsafe_allow_html=True)

    with r2:
        st.markdown("**Feature Breakdown**")
        feat_labels = {"sbert_similarity":"Semantic Similarity","jaccard_skills":"Skill Overlap","keyword_match_ratio":"Keyword Match"}
        feat_colors = ["#2ecc71" if v>=0.5 else "#f39c12" if v>=0.3 else "#e74c3c" for v in features.values()]
        fig_f = go.Figure(go.Bar(x=list(features.values()), y=[feat_labels[k] for k in features],
            orientation="h", marker_color=feat_colors,
            text=[f"{v:.2f}" for v in features.values()], textposition="outside"))
        fig_f.update_layout(height=220, xaxis_range=[0,1.15], paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10,b=10,l=10,r=50),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(showgrid=False), font=dict(size=12))
        st.plotly_chart(fig_f, use_container_width=True)

    with r3:
        st.markdown("**Scoring Summary**")
        penalty_pts = round(raw_score - final_score, 1)
        n_jd = len(matched_skills) + len(missing_skills)
        st.markdown(
            f'<div class="metric-box"><small>{len(missing_skills)}/{n_jd} JD skills missing ({gap_ratio*100:.0f}% gap)</small></div>'
            f'<div class="metric-box"><b>Final Score</b><br>'
            f'<span style="font-size:1.6rem;color:{score_color}"><b>{final_score}</b></span> / 100</div>',
            unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🔍 Skill Gap Analysis")
    sk1, sk2 = st.columns(2)
    with sk1:
        st.markdown(f"**✅ Matched Skills ({len(matched_skills)})**")
        if matched_skills:
            st.markdown(" ".join(f'<span class="skill-tag-green">{s}</span>' for s in sorted(matched_skills)),
                        unsafe_allow_html=True)
        else:
            st.caption("None of the JD skills found in resume.")
    with sk2:
        st.markdown(f"**❌ Missing Skills ({len(missing_skills)})**")
        if missing_skills:
            st.markdown(" ".join(f'<span class="skill-tag-red">{s}</span>' for s in sorted(missing_skills)),
                        unsafe_allow_html=True)
        else:
            st.success("All detected JD skills are present! 🎉")

    st.divider()
    if final_score < SCORE_THRESHOLD:
        st.markdown("### 💡 AI Improvement Suggestions")
        with st.spinner("Generating with Gemini 2.0 Flash..."):
            sg = get_llm_suggestions(resume_text, jd_text, final_score, list(missing_skills))
        if sg.get("overall_analysis"): st.info(sg["overall_analysis"])
        sg1, sg2 = st.columns(2)
        with sg1:
            if sg.get("priority_actions"):
                st.markdown("**🎯 Top Priority Actions**")
                for i, a in enumerate(sg["priority_actions"][:3], 1): st.markdown(f"{i}. {a}")
            if sg.get("missing_keywords"):
                st.markdown("**🔑 Keywords to Add**")
                st.markdown(" ".join(f'<span class="skill-tag-red">{k}</span>' for k in sg["missing_keywords"]),
                            unsafe_allow_html=True)
        with sg2:
            if sg.get("suggestions"):
                st.markdown("**📝 Section-wise Fixes**")
                for s in sg["suggestions"]:
                    lbl = f"[{s.get('section','General')}] {s.get('issue','')[:60]}"
                    with st.expander(lbl):
                        st.markdown(f"**Fix:** {s.get('action','')}")
                        if s.get("example"): st.code(s["example"], language=None)
    else:
        st.success(f"✅ Score {final_score}/100 — your resume is a good match! No improvements needed.")
