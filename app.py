import os
import sys
import time
import json
import datetime
import warnings
import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from utils.preprocess import preprocess
from utils.scraper import get_article_text
from utils.explain import get_top_keywords, normalize_scores, get_category_hint

st.set_page_config(
    page_title="TruthLens · Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH      = "model/fake_news_model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"
METADATA_PATH   = "model/model_metadata.pkl"
HISTORY_KEY     = "prediction_history"


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --ink:       #0d0d0d;
        --paper:     #fafaf7;
        --accent:    #c0392b;
        --accent2:   #2980b9;
        --real:      #27ae60;
        --fake:      #e74c3c;
        --muted:     #7f8c8d;
        --border:    #e8e3da;
        --card-bg:   #ffffff;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--paper);
        color: var(--ink);
    }

    /* 1. White Box wrapper specifically targeting your requested labels */
    div[data-testid="stWidgetLabel"] p {
        background-color: #ffffff !important;
        color: #0d0d0d !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border: 1px solid #dcd7cc !important;
        border-radius: 6px !important;
        padding: 6px 14px !important;
        display: inline-block !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
        margin-bottom: 4px !important;
    }

    /* 2. Forcing a custom blinking keyboard cursor inside the actual text boxes */
    .stTextArea textarea, .stTextInput input {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        border-radius: 10px !important;
        border-color: var(--border) !important;
        background: #ffffff !important;
        color: #0d0d0d !important;
        
        /* Forces standard browser text box carets to flash brightly matching your brand color */
        caret-color: #c0392b !important; 
    }

    /* 3. Fallback blinking layout animation if you want a simulated blinking cursor inside empty fields */
    .stTextArea textarea::placeholder, .stTextInput input::placeholder {
        animation: placeholderBlink 1.5s step-end infinite;
    }

    @keyframes placeholderBlink {
        from, to { border-left: 2px solid transparent; }
        50% { border-left: 2px solid #c0392b; }
    }

    /* --- Core Application Theme Layouts --- */
    .hero-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 48px 56px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.1;
        margin: 0 0 12px 0;
    }
    .hero-sub {
        font-size: 1rem;
        color: rgba(255,255,255,0.65);
        font-weight: 300;
        margin: 0;
        max-width: 520px;
        line-height: 1.6;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(192,57,43,0.85);
        color: white;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 16px;
    }

    .card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .card-title {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 10px;
    }

    .result-real { border-left: 5px solid var(--real); background: linear-gradient(to right, #f0fff4, #ffffff); }
    .result-fake { border-left: 5px solid var(--fake); background: linear-gradient(to right, #fff5f5, #ffffff); }
    .verdict-label { font-family: 'Playfair Display', serif; font-size: 2.4rem; font-weight: 700; margin: 0; }
    .verdict-real  { color: var(--real); }
    .verdict-fake  { color: var(--fake); }
    .confidence-number { font-family: 'DM Mono', monospace; font-size: 1rem; color: var(--muted); }

    .pill-real {
        display: inline-block;
        background: #e8f8ee; color: #1e8449;
        border: 1px solid #a9dfbf; border-radius: 20px;
        padding: 4px 14px; margin: 3px;
        font-size: 0.82rem; font-weight: 500;
        font-family: 'DM Mono', monospace;
    }
    .pill-fake {
        display: inline-block;
        background: #fef0f0; color: #c0392b;
        border: 1px solid #f1948a; border-radius: 20px;
        padding: 4px 14px; margin: 3px;
        font-size: 0.82rem; font-weight: 500;
        font-family: 'DM Mono', monospace;
    }

    .section-rule { border: none; border-top: 1px solid var(--border); margin: 28px 0; }

    [data-testid="stTabContent"] {
        padding-top: 16px !important;
        padding-bottom: 4px !important;
    }

    .stButton > button {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        border-radius: 10px;
        padding: 10px 28px;
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white;
        border: none;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #a93226, #c0392b);
        box-shadow: 0 4px 16px rgba(192,57,43,0.35);
        transform: translateY(-1px);
    }

    [data-testid="stSidebar"] {
        background-color: #0d0d0d;
        border-right: 1px solid #1e1e1e;
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    
    .sidebar-footer {
        margin-top: 40px;
        font-size: 0.72rem;
        color: #444;
        line-height: 1.7;
        padding-top: 12px;
        border-top: 1px solid #222;
    }

    .hist-row {
        display: flex; align-items: center; gap: 12px;
        padding: 10px 0; border-bottom: 1px solid var(--border);
        font-size: 0.88rem;
    }
    .badge-real { color: var(--real); font-weight: 700; font-size: 0.8rem; }
    .badge-fake { color: var(--fake); font-weight: 700; font-size: 0.8rem; }

    .footer {
        text-align: center;
        padding: 36px 20px;
        margin-top: 48px;
        color: var(--muted);
        font-size: 0.82rem;
        border-top: 1px solid var(--border);
    }

    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    """Loads pre-trained artifacts or handles missing dependencies gracefully."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        try:
            from train_model import create_demo_model
            create_demo_model()
        except ModuleNotFoundError:
            st.error("🚨 Missing trained ML model files. Please run `python train_model.py` first.")
            st.stop()

    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    metadata   = joblib.load(METADATA_PATH) if os.path.exists(METADATA_PATH) else {}
    return model, vectorizer, metadata


def predict(text: str, model, vectorizer) -> dict:
    cleaned = preprocess(text)
    vec     = vectorizer.transform([cleaned])
    pred    = model.predict(vec)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        conf_real = float(proba[1])
        conf_fake = float(proba[0])
    else:
        score = model.decision_function(vec)[0]
        conf_real = 1 / (1 + np.exp(-score))
        conf_fake = 1 - conf_real

    verdict    = "REAL" if pred == 1 else "FAKE"
    confidence = conf_real if verdict == "REAL" else conf_fake

    return {
        "verdict":     verdict,
        "confidence":  round(confidence * 100, 1),
        "conf_real":   round(conf_real * 100, 1),
        "conf_fake":   round(conf_fake * 100, 1),
    }


def make_gauge(confidence: float, verdict: str) -> go.Figure:
    color = "#27ae60" if verdict == "REAL" else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"<b>{verdict}</b>", "font": {"size": 22, "family": "Playfair Display"}},
        number={"suffix": "%", "font": {"size": 36, "family": "DM Mono, monospace"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"size": 11}},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#e8e3da",
            "steps": [
                {"range": [0,   40], "color": "#fef9f9"},
                {"range": [40,  65], "color": "#fffbf0"},
                {"range": [65, 100], "color": "#f0fff4"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": confidence,
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans"},
    )
    return fig


def make_keyword_chart(kw_real: list, kw_fake: list) -> go.Figure:
    real_norm = normalize_scores(kw_real[:8])
    fake_norm = normalize_scores(kw_fake[:8])

    words  = [w for w, _ in real_norm] + [w for w, _ in fake_norm]
    scores = [s for _, s in real_norm] + [-s for _, s in fake_norm]
    colors = ["#27ae60"] * len(real_norm) + ["#e74c3c"] * len(fake_norm)
    labels = ["REAL signal"] * len(real_norm) + ["FAKE signal"] * len(fake_norm)

    fig = go.Figure(go.Bar(
        x=scores,
        y=words,
        orientation="h",
        marker_color=colors,
        text=[f"{abs(s):.0f}" for s in scores],
        textposition="outside",
        hovertemplate="%{y}: %{customdata}<extra></extra>",
        customdata=labels,
    ))
    fig.update_layout(
        xaxis=dict(
            title="← Fake signal  |  Real signal →",
            showgrid=True, gridcolor="#f0ece6",
            zeroline=True, zerolinecolor="#bbb", zerolinewidth=1.5,
            range=[-110, 110],
        ),
        yaxis=dict(autorange="reversed"),
        height=max(300, (len(words)) * 34 + 60),
        margin=dict(l=10, r=40, t=16, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Mono, monospace", "size": 12},
        showlegend=False,
    )
    return fig


def _pdf_safe(text: str) -> str:
    import unicodedata
    result_chars = []
    for ch in str(text):
        try:
            ch.encode("latin-1")
            result_chars.append(ch)
        except (UnicodeEncodeError, UnicodeDecodeError):
            name = unicodedata.name(ch, "")
            if "CATEGORY" in name.upper() or name == "":
                result_chars.append(" ")
            else:
                result_chars.append("?")
    return "".join(result_chars).strip()


def generate_pdf_report(result: dict, text_snippet: str, source: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(20, 20, 20)

    safe_category = _pdf_safe(result.get("category", "General News"))
    safe_source   = _pdf_safe(source)

    pdf.set_fill_color(26, 26, 46)
    pdf.rect(0, 0, 210, 40, "F")
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 12, "", ln=True)
    pdf.cell(0, 12, "  TruthLens - Fake News Detection Report", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(180, 180, 180)
    pdf.cell(0, 8, f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    pdf.ln(12)
    verdict = result["verdict"]
    pdf.set_font("Helvetica", "B", 28)
    if verdict == "REAL":
        pdf.set_text_color(39, 174, 96)
    else:
        pdf.set_text_color(231, 76, 60)
    pdf.cell(0, 14, f"  Verdict: {verdict}", ln=True)

    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, f"  Confidence: {result['confidence']}%", ln=True)
    pdf.cell(0, 8, f"  Category: {safe_category}", ln=True)
    pdf.cell(0, 8, _pdf_safe(f"  Source: {safe_source}"), ln=True)

    pdf.ln(5)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 8, "Article Text (first 600 chars):", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)
    safe_snippet = _pdf_safe(text_snippet[:600])
    pdf.multi_cell(0, 5, safe_snippet or "(no text)")

    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 8, "Real-signal keywords:", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(39, 174, 96)
    real_kw = _pdf_safe(", ".join(w for w, _ in result.get("kw_real", [])[:10]))
    pdf.multi_cell(0, 5, real_kw or "None detected")

    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(0, 8, "Fake-signal keywords:", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(231, 76, 60)
    fake_kw = _pdf_safe(", ".join(w for w, _ in result.get("kw_fake", [])[:10]))
    pdf.multi_cell(0, 5, fake_kw or "None detected")

    pdf.set_y(-25)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, "TruthLens AI - Fake News Detector - For informational use only", align="C")

    return bytes(pdf.output())


def render_sidebar(metadata: dict):
    with st.sidebar:
        st.markdown("""
        <div style='padding: 20px 0 8px 0;'>
            <span style='font-size:1.7rem;'>🔍</span>
            <span style='font-family: Playfair Display, serif; font-size:1.25rem;
                         font-weight:700; margin-left:8px; color:#fff;'>TruthLens</span>
        </div>
        <p style='font-size:0.75rem; color:#666; margin-top:2px; padding-bottom:16px;
                  border-bottom:1px solid #222;'>AI-Powered News Verifier</p>
        """, unsafe_allow_html=True)

        page = st.radio(
            "NAVIGATE",
            ["🔎 Detector", "📜 History", "📊 Model Info", "ℹ️ About"],
            label_visibility="visible",
            key="navigation_menu"
        )

        st.markdown("<hr style='border-color:#222; margin:20px 0;'>", unsafe_allow_html=True)

        if metadata:
            st.markdown("<p style='font-size:0.7rem; letter-spacing:2px; color:#555; text-transform:uppercase;'>Model</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:0.82rem; color:#aaa;'>{metadata.get('model_name','—')}</p>", unsafe_allow_html=True)

            acc = metadata.get("accuracy", "—")
            if isinstance(acc, float):
                st.markdown(f"<p style='font-size:0.82rem; color:#aaa;'>Accuracy: <b style='color:#27ae60;'>{acc*100:.1f}%</b></p>", unsafe_allow_html=True)

            vocab = metadata.get("vocab_size", 0)
            if vocab:
                st.markdown(f"<p style='font-size:0.82rem; color:#aaa;'>Vocabulary: {vocab:,} terms</p>", unsafe_allow_html=True)

            if metadata.get("is_demo"):
                st.warning("⚡ Demo model active. Train with real data for production accuracy.", icon="⚡")

        st.markdown("""
        <div class="sidebar-footer">
            Built with Streamlit &amp; scikit-learn<br>
            NLTK · Plotly · BeautifulSoup
        </div>
        """, unsafe_allow_html=True)

    return page


def page_detector(model, vectorizer):
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge">AI-Powered · NLP · Real-Time</div>
        <h1 class="hero-title">Is This News Real?</h1>
        <p class="hero-sub">
            Paste any news article or URL below. TruthLens analyses
            the text using machine-learning to detect patterns common
            in misinformation and disinformation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if "scraped_text"  not in st.session_state:
        st.session_state.scraped_text  = ""
    if "scraped_title" not in st.session_state:
        st.session_state.scraped_title = ""
    if "scraped_url"   not in st.session_state:
        st.session_state.scraped_url   = ""

    tab_text, tab_url = st.tabs(["📝 Paste Article Text", "🔗 Enter News URL"])
    input_text   = ""
    source_label = "Manual input"

    with tab_text:
        input_text = st.text_area(
            "Paste the full news article or headline here:",
            height=220,
            placeholder="Paste article headline and body text here...",
            help="Minimum 30 characters required.",
            key="article_text_input"
        )

    with tab_url:
        url_input = st.text_input(
            "News article URL:",
            placeholder="https://www.bbc.com/news/article-example",
            key="article_url_input"
        )
        fetch_col, _ = st.columns([1, 3])
        with fetch_col:
            fetch_btn = st.button("🌐 Fetch Article", key="fetch_url", use_container_width=True)

        if fetch_btn:
            if not url_input:
                st.warning("Please paste a URL first.")
            else:
                with st.spinner("Scraping article — please wait…"):
                    scraped = get_article_text(url_input)
                if scraped["error"]:
                    st.error(f"❌ {scraped['error']}")
                    st.session_state.scraped_text  = ""
                    st.session_state.scraped_title = ""
                    st.session_state.scraped_url   = ""
                else:
                    st.session_state.scraped_text  = scraped["text"]
                    st.session_state.scraped_title = scraped["title"]
                    st.session_state.scraped_url   = url_input
                    word_count = len(scraped["text"].split())
                    st.success(f"✅ Article fetched! ({word_count:,} words extracted)")

        if st.session_state.scraped_text and not fetch_btn:
            wc = len(st.session_state.scraped_text.split())
            st.info(f"Ready to analyse: **{st.session_state.scraped_title or 'Fetched article'}** ({wc:,} words).")

    if not input_text and st.session_state.scraped_text:
        input_text   = st.session_state.scraped_text
        source_label = st.session_state.scraped_url or "URL"
    elif input_text:
        source_label = "Manual input"
        st.session_state.scraped_text  = ""
        st.session_state.scraped_title = ""
        st.session_state.scraped_url   = ""

    lang_col, btn_col = st.columns([2, 1])
    with lang_col:
        language = st.selectbox(
            "Input language:",
            ["English", "Hindi", "Spanish", "French", "German", "Arabic", "Portuguese", "Russian", "Japanese", "Chinese"],
            key="ui_language"
        )
    with btn_col:
        analyse_btn = st.button("🔍 Analyse Article", use_container_width=True, key="trigger_analysis")

    if analyse_btn and input_text and language != "English":
        try:
            from deep_translator import GoogleTranslator
            lang_map = {
                "Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de",
                "Arabic": "ar", "Portuguese": "pt", "Russian": "ru",
                "Japanese": "ja", "Chinese": "zh-CN",
            }
            code = lang_map.get(language, "auto")
            with st.spinner(f"Translating from {language}…"):
                chunks = [input_text[i:i+4500] for i in range(0, len(input_text), 4500)]
                translated = " ".join(GoogleTranslator(source=code, target="en").translate(c) for c in chunks)
            input_text = translated
        except Exception as e:
            st.warning(f"Translation failed ({e}). Analysing as-is.")

    if analyse_btn:
        if not input_text or len(input_text.strip()) < 30:
            st.error("⚠️ No article text found or content is too short.")
            return

        with st.spinner("Analysing with AI…"):
            time.sleep(0.2)
            result = predict(input_text, model, vectorizer)
            kw     = get_top_keywords(input_text, vectorizer, model, top_n=12)
            cat    = get_category_hint(input_text)

        result["kw_real"]  = kw["real_keywords"]
        result["kw_fake"]  = kw["fake_keywords"]
        result["category"] = cat

        if HISTORY_KEY not in st.session_state:
            st.session_state[HISTORY_KEY] = []
        st.session_state[HISTORY_KEY].insert(0, {
            "time":       datetime.datetime.now().strftime("%H:%M:%S"),
            "verdict":    result["verdict"],
            "confidence": result["confidence"],
            "category":   cat,
            "snippet":    input_text[:80] + "…",
        })

        st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
        st.markdown("### 📋 Analysis Results")

        verdict_class = "result-real" if result["verdict"] == "REAL" else "result-fake"
        verdict_label = "verdict-real" if result["verdict"] == "REAL" else "verdict-fake"
        icon = "✅" if result["verdict"] == "REAL" else "❌"

        col_v, col_g = st.columns([1, 1])

        with col_v:
            st.markdown(f"""
            <div class="card animate-in {verdict_class}">
                <div class="card-title">Verdict</div>
                <p class="verdict-label {verdict_label}">{icon} {result['verdict']}</p>
                <p class="confidence-number">Confidence: {result['confidence']}%</p>
                <p style="color:#999; font-size:0.85rem; margin-top:8px;">Category: {cat}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="card animate-in" style="margin-top:0;">
                <div class="card-title">Probability Breakdown</div>
                <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                    <span style="color:#27ae60; font-weight:600;">REAL</span>
                    <span style="font-family:'DM Mono',monospace;">{result['conf_real']}%</span>
                </div>
            """, unsafe_allow_html=True)
            st.progress(result["conf_real"] / 100)
            st.markdown(f"""
                <div style="display:flex; justify-content:space-between; margin:12px 0 8px;">
                    <span style="color:#e74c3c; font-weight:600;">FAKE</span>
                    <span style="font-family:'DM Mono',monospace;">{result['conf_fake']}%</span>
                </div>
            """, unsafe_allow_html=True)
            st.progress(result["conf_fake"] / 100)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_g:
            st.plotly_chart(
                make_gauge(result["confidence"], result["verdict"]),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
        st.markdown("### 🔑 Keyword Influence Analysis")

        kw_col1, kw_col2 = st.columns(2)

        with kw_col1:
            st.markdown("<div class='card'><div class='card-title'>🟢 Real-News Signal Words</div>", unsafe_allow_html=True)
            if kw["real_keywords"]:
                pills = "".join(f'<span class="pill-real">{w}</span>' for w, _ in kw["real_keywords"][:10])
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.markdown("<em style='color:#aaa;'>None detected</em>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with kw_col2:
            st.markdown("<div class='card'><div class='card-title'>🔴 Fake-News Indicator Words</div>", unsafe_allow_html=True)
            if kw["fake_keywords"]:
                pills = "".join(f'<span class="pill-fake">{w}</span>' for w, _ in kw["fake_keywords"][:10])
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.markdown("<em style='color:#aaa;'>None detected</em>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        if kw["real_keywords"] or kw["fake_keywords"]:
            st.plotly_chart(
                make_keyword_chart(kw["real_keywords"], kw["fake_keywords"]),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
        pdf_bytes = generate_pdf_report(result, input_text, source_label)
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name=f"truthlens_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )


def page_history():
    st.markdown("## 📜 Recent Predictions")
    history = st.session_state.get(HISTORY_KEY, [])
    if not history:
        st.info("No predictions yet.")
        return

    cols = st.columns(3)
    real_count = sum(1 for h in history if h["verdict"] == "REAL")
    fake_count = len(history) - real_count

    cols[0].metric("Total Analysed", len(history))
    cols[1].metric("✅ Real", real_count)
    cols[2].metric("❌ Fake", fake_count)

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

    for entry in history:
        badge = f'<span class="badge-real">✅ REAL</span>' if entry["verdict"] == "REAL" else f'<span class="badge-fake">❌ FAKE</span>'
        st.markdown(f"""
        <div class="hist-row">
            {badge}
            <span style="color:#999; font-family:'DM Mono',monospace; font-size:0.8rem;">{entry['time']}</span>
            <span style="color:#999; font-size:0.8rem;">{entry.get('category','')}</span>
            <span style="flex:1; color:#555;">{entry['snippet']}</span>
            <span style="font-family:'DM Mono',monospace; font-size:0.85rem; color:#888;">{entry['confidence']}%</span>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🗑️ Clear History"):
        st.session_state[HISTORY_KEY] = []
        st.rerun()


def page_model_info(metadata: dict):
    st.markdown("## Model Information")
    if not metadata:
        st.warning("No metadata found.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Architecture")
        st.markdown(f"**Algorithm:** {metadata.get('model_name', '—')}")
        st.markdown(f"**Vectorizer:** TF-IDF {metadata.get('ngram_range', '(1,2)')} n-grams")
        st.markdown(f"**Max Features:** {metadata.get('max_features', 0):,}")
        st.markdown(f"**Vocabulary:** {metadata.get('vocab_size', 0):,} terms")

    with col2:
        st.markdown("#### Training Stats")
        st.markdown(f"**Train Samples:** {metadata.get('train_samples', 0):,}")
        st.markdown(f"**Test Samples:** {metadata.get('test_samples', 0):,}")
        acc = metadata.get("accuracy", "—")
        if isinstance(acc, float):
            st.markdown(f"**Test Accuracy:** :green[**{acc*100:.1f}%**]")

    st.divider()
    st.markdown("""
    ### Improving Accuracy with BERT
    | Approach | Accuracy | Notes |
    |---|---|---|
    | TF-IDF + LR (current) | ~92–94% | Fast, interpretable, deployable anywhere |
    | DistilBERT fine-tuned | ~96–98% | Best accuracy, needs GPU for training |
    """)


def page_about():
    st.markdown("## ℹ️ About TruthLens")
    st.markdown("""
    **TruthLens** is an open-source AI-powered fake news detection system.
    
    ### ⚠️ Disclaimer
    This tool is for educational purposes only. No AI model is 100% accurate.
    """)


def main():
    inject_css()
    with st.spinner("Loading TruthLens AI…"):
        model, vectorizer, metadata = load_model()

    page = render_sidebar(metadata)

    if page == "🔎 Detector":
        page_detector(model, vectorizer)
    elif page == "📜 History":
        page_history()
    elif page == "📊 Model Info":
        page_model_info(metadata)
    elif page == "ℹ️ About":
        page_about()

    st.markdown("""
    <div class="footer">
        🔍 <strong>TruthLens</strong> · AI Fake News Detector
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()