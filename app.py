# =============================================================================
# InsureIQ · Insurance Charges Prediction Platform
# Author   : Om  |  v1.0.0  |  2025
# Run with : streamlit run app.py
# =============================================================================

from __future__ import annotations

import io
import pickle
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
APP_NAME       = "InsureIQ"
APP_SUBTITLE   = "Insurance Charges Intelligence Platform"
APP_VERSION    = "v1.0.0"
APP_UPDATED    = "June 2025"
DATA_PATH      = Path(__file__).parent / "insurance-checkpoint ml-01.csv"
BUNDLE_PATH    = Path("model_bundle.pkl")
ACCENT         = "#7C3AED"
ACCENT_LIGHT   = "#A78BFA"
BG             = "#0F1117"
CARD_BG        = "#1E2130"
CARD2_BG       = "#161929"
SUCCESS_COLOR  = "#10B981"
DANGER_COLOR   = "#EF4444"
WARN_COLOR     = "#F59E0B"
INFO_COLOR     = "#3B82F6"
PLOTLY_THEME   = "plotly_dark"

# Shared Plotly layout defaults for consistency across all charts
PLOTLY_LAYOUT_DEFAULTS = dict(
    template=PLOTLY_THEME,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12, color="#E2E8F0"),
    hoverlabel=dict(
        bgcolor=CARD_BG,
        font_size=13,
        font_family="Inter, sans-serif",
        font_color="#E2E8F0",
        bordercolor=ACCENT,
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color="#94A3B8"),
    ),
)

# Color palette for multi-series charts
PALETTE = [ACCENT_LIGHT, SUCCESS_COLOR, WARN_COLOR, DANGER_COLOR, INFO_COLOR, "#F472B6", "#34D399", "#FBBF24"]

FEATURE_HELP = {
    "age":      "Your age in years (18–64). Older individuals tend to have higher medical costs.",
    "sex":      "Biological sex. Minor influence on predicted charges.",
    "bmi":      "Body Mass Index — weight(kg) / height(m)². Values above 30 indicate obesity.",
    "children": "Number of dependents covered by the insurance plan (0–5).",
    "smoker":   "Whether you are a current smoker. One of the strongest predictors of charges.",
    "region":   "US geographic region of the policyholder.",
}

SAMPLE_ROWS = [
    {"age": 19, "sex": "female", "bmi": 27.9,  "children": 0, "smoker": "yes", "region": "southwest"},
    {"age": 45, "sex": "male",   "bmi": 33.44, "children": 2, "smoker": "no",  "region": "southeast"},
    {"age": 31, "sex": "female", "bmi": 25.74, "children": 0, "smoker": "no",  "region": "southeast"},
    {"age": 55, "sex": "male",   "bmi": 36.0,  "children": 1, "smoker": "yes", "region": "northeast"},
]

# Sidebar navigation config — icon, label
NAV_ITEMS = [
    ("🏠", "Predict"),
    ("📊", "Data Explorer"),
    ("📈", "Model Performance"),
    ("ℹ️",  "About"),
]

# ---------------------------------------------------------------------------
# PAGE CONFIG (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_NAME} · Insurance Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# PLOTLY HELPER — shared config for all charts
# ---------------------------------------------------------------------------
PLOTLY_CONFIG = {
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "insureiq_chart",
        "height": 600,
        "width": 1000,
        "scale": 2,
    },
}


def _apply_defaults(fig: go.Figure, **overrides) -> go.Figure:
    """Apply consistent layout defaults to any Plotly figure."""
    layout = {**PLOTLY_LAYOUT_DEFAULTS, **overrides}
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------------------------
def inject_css() -> None:
    """Inject the full custom stylesheet into the Streamlit app."""
    st.markdown(f"""
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        background-color: {BG};
        color: #E2E8F0;
    }}
    .main .block-container {{ padding: 1.5rem 2rem 4rem; max-width: 1200px; }}

    /* ── Sidebar overhaul ── */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {CARD2_BG} 0%, #0D0F1A 100%) !important;
        border-right: 1px solid #2D374855;
        padding: 0 !important;
    }}
    section[data-testid="stSidebar"] > div:first-child {{
        padding: 0 !important;
    }}
    /* Hide the default radio labels */
    section[data-testid="stSidebar"] .stRadio {{ display: none !important; }}

    /* ── Sidebar brand logo ── */
    .sidebar-brand {{
        text-align: center;
        padding: 2rem 1.5rem 1.2rem;
        position: relative;
        overflow: hidden;
    }}
    .sidebar-brand::before {{
        content: '';
        position: absolute;
        top: -40px; left: -40px; right: -40px;
        height: 160px;
        background: radial-gradient(ellipse at 50% 0%, {ACCENT}25 0%, transparent 70%);
        pointer-events: none;
    }}
    .sidebar-brand-icon {{
        width: 56px; height: 56px;
        margin: 0 auto 10px;
        border-radius: 16px;
        background: linear-gradient(135deg, {ACCENT}, #4F46E5);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.6rem;
        box-shadow: 0 8px 30px {ACCENT}33;
        position: relative;
        animation: pulse-glow 3s ease-in-out infinite;
    }}
    @keyframes pulse-glow {{
        0%, 100% {{ box-shadow: 0 8px 30px {ACCENT}33; }}
        50% {{ box-shadow: 0 8px 45px {ACCENT}55; }}
    }}
    .sidebar-brand-name {{
        font-size: 1.3rem; font-weight: 700;
        background: linear-gradient(135deg, {ACCENT_LIGHT}, #818CF8);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }}
    .sidebar-brand-tagline {{
        font-size: 0.68rem; color: #64748B; margin-top: 3px;
        letter-spacing: 0.04em;
    }}

    /* ── Sidebar nav pills ── */
    .sidebar-nav {{
        padding: 0 0.9rem;
    }}
    .sidebar-nav-label {{
        font-size: 0.62rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.1em;
        color: #475569; padding: 0 0.6rem; margin-bottom: 6px;
    }}
    .sidebar-nav-item {{
        display: flex; align-items: center; gap: 10px;
        padding: 0.65rem 0.9rem;
        border-radius: 10px; cursor: pointer;
        font-size: 0.85rem; font-weight: 500;
        color: #94A3B8;
        transition: all 0.2s ease;
        margin-bottom: 3px;
        text-decoration: none;
        border: 1px solid transparent;
    }}
    .sidebar-nav-item:hover {{
        background: #ffffff08;
        color: #E2E8F0;
        border-color: #ffffff08;
        transform: translateX(3px);
    }}
    .sidebar-nav-item.active {{
        background: linear-gradient(135deg, {ACCENT}22, {ACCENT}11);
        color: {ACCENT_LIGHT};
        border-color: {ACCENT}33;
        font-weight: 600;
        box-shadow: 0 2px 12px {ACCENT}15;
    }}
    .sidebar-nav-item.active .nav-indicator {{
        opacity: 1;
    }}
    .nav-icon {{
        font-size: 1.05rem;
        width: 24px; text-align: center;
    }}
    .nav-indicator {{
        width: 6px; height: 6px; border-radius: 50%;
        background: {ACCENT_LIGHT};
        margin-left: auto; opacity: 0;
        transition: opacity 0.2s ease;
    }}

    /* ── Sidebar divider ── */
    .sidebar-divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, #2D374888, transparent);
        margin: 1rem 1.2rem;
    }}

    /* ── Sidebar model card ── */
    .sidebar-stats {{
        padding: 0 0.9rem;
    }}
    .sidebar-stats-label {{
        font-size: 0.62rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.1em;
        color: #475569; padding: 0 0.6rem; margin-bottom: 8px;
    }}
    .sidebar-stat-card {{
        background: linear-gradient(135deg, #1a1d2e, #161929);
        border: 1px solid #2D374866;
        border-radius: 10px; padding: 0.85rem 1rem;
    }}
    .stat-row {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 5px 0;
        font-size: 0.78rem;
    }}
    .stat-row .stat-label {{
        color: #64748B;
        display: flex; align-items: center; gap: 6px;
    }}
    .stat-row .stat-value {{
        color: #E2E8F0; font-weight: 600;
        font-variant-numeric: tabular-nums;
    }}
    .stat-row .stat-value.accent {{ color: {ACCENT_LIGHT}; }}
    .stat-row .stat-value.green  {{ color: {SUCCESS_COLOR}; }}
    .stat-row .stat-value.warn   {{ color: {WARN_COLOR}; }}
    .stat-divider {{
        height: 1px; background: #2D374855; margin: 4px 0;
    }}
    /* R² progress bar */
    .r2-bar-bg {{
        height: 4px; border-radius: 99px;
        background: #2D3748; margin-top: 6px;
        overflow: hidden;
    }}
    .r2-bar-fill {{
        height: 100%; border-radius: 99px;
        background: linear-gradient(90deg, {ACCENT}, {SUCCESS_COLOR});
        transition: width 0.8s ease;
    }}

    /* ── Sidebar prediction counter ── */
    .sidebar-counter {{
        padding: 0 0.9rem; margin-top: 0.2rem;
    }}
    .counter-card {{
        background: linear-gradient(135deg, {ACCENT}12, #4F46E508);
        border: 1px solid {ACCENT}22;
        border-radius: 10px; padding: 0.7rem 1rem;
        display: flex; align-items: center; gap: 10px;
    }}
    .counter-num {{
        font-size: 1.5rem; font-weight: 700;
        color: {ACCENT_LIGHT};
        font-variant-numeric: tabular-nums;
        line-height: 1;
    }}
    .counter-label {{
        font-size: 0.68rem; color: #64748B;
        line-height: 1.3;
    }}
    .counter-label strong {{ color: #94A3B8; }}

    /* ── Sidebar status dot ── */
    .status-dot {{
        display: inline-block;
        width: 7px; height: 7px; border-radius: 50%;
        margin-right: 6px;
        animation: blink 2s ease-in-out infinite;
    }}
    .status-dot.online {{ background: {SUCCESS_COLOR}; box-shadow: 0 0 8px {SUCCESS_COLOR}66; }}
    @keyframes blink {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.4; }}
    }}

    /* ── Sidebar footer ── */
    .sidebar-footer {{
        padding: 1rem 1.2rem;
        text-align: center;
    }}
    .sidebar-footer p {{
        font-size: 0.65rem; color: #475569; margin: 0;
        line-height: 1.6;
    }}
    .sidebar-footer a {{
        color: {ACCENT_LIGHT}; text-decoration: none;
    }}
    .sidebar-footer a:hover {{ text-decoration: underline; }}

    /* ── Sticky header ── */
    .app-header {{
        position: sticky; top: 0; z-index: 999;
        background: linear-gradient(90deg, {BG} 0%, {CARD2_BG} 100%);
        border-bottom: 1px solid #2D3748;
        padding: 0.75rem 1.5rem;
        display: flex; align-items: center; justify-content: space-between;
        margin: -1.5rem -2rem 1.5rem;
    }}
    .app-header-left {{ display: flex; align-items: center; gap: 12px; }}
    .app-header-logo {{ font-size: 1.5rem; font-weight: 700; color: {ACCENT_LIGHT}; }}
    .app-header-sub  {{ font-size: 0.75rem; color: #94A3B8; }}
    .badge {{
        background: {ACCENT}22; color: {ACCENT_LIGHT};
        border: 1px solid {ACCENT}55; border-radius: 99px;
        padding: 2px 10px; font-size: 0.7rem; font-weight: 600;
    }}
    .badge-green {{ background: {SUCCESS_COLOR}22; color: {SUCCESS_COLOR}; border-color: {SUCCESS_COLOR}55; }}
    .badge-red   {{ background: {DANGER_COLOR}22;  color: {DANGER_COLOR};  border-color: {DANGER_COLOR}55;  }}

    /* ── Cards ── */
    .card {{
        background: {CARD_BG}; border: 1px solid #2D3748;
        border-radius: 12px; padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }}
    .card-accent {{ border-left: 3px solid {ACCENT}; }}
    .result-card-green {{
        background: {SUCCESS_COLOR}11; border: 1px solid {SUCCESS_COLOR}44;
        border-left: 4px solid {SUCCESS_COLOR}; border-radius: 12px;
        padding: 1.5rem; text-align: center;
    }}
    .result-card-red {{
        background: {DANGER_COLOR}11; border: 1px solid {DANGER_COLOR}44;
        border-left: 4px solid {DANGER_COLOR}; border-radius: 12px;
        padding: 1.5rem; text-align: center;
    }}
    .result-amount {{ font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0; }}

    /* ── Metric override ── */
    [data-testid="stMetric"] {{
        background: {CARD_BG}; border: 1px solid #2D3748;
        border-radius: 10px; padding: 0.85rem 1rem;
    }}
    [data-testid="stMetricValue"] {{ color: {ACCENT_LIGHT} !important; font-size: 1.4rem !important; font-weight: 600 !important; }}
    [data-testid="stMetricLabel"] {{ color: #94A3B8 !important; font-size: 0.75rem !important; }}

    /* ── Buttons ── */
    .stButton > button {{
        background: {ACCENT}; color: white; border: none;
        border-radius: 8px; padding: 0.5rem 1.25rem;
        font-weight: 600; transition: all .2s;
    }}
    .stButton > button:hover {{ background: #6D28D9; transform: translateY(-1px); box-shadow: 0 4px 15px {ACCENT}44; }}

    /* ── Tables ── */
    .stDataFrame {{ border-radius: 10px; overflow: hidden; }}

    /* ── History table ── */
    .history-row {{
        background: {CARD_BG}; border: 1px solid #2D3748;
        border-radius: 8px; padding: 0.6rem 0.9rem;
        margin-bottom: 6px; font-size: 0.82rem;
        display: flex; justify-content: space-between; align-items: center;
    }}

    /* ── Feature tags ── */
    .feat-tag {{
        display: inline-block;
        background: {ACCENT}22; color: {ACCENT_LIGHT};
        border: 1px solid {ACCENT}44; border-radius: 6px;
        padding: 2px 8px; font-size: 0.72rem; margin: 2px;
    }}

    /* ── About tech stack ── */
    .tech-badge {{
        display: inline-block;
        padding: 4px 14px; border-radius: 99px; font-size: 0.75rem; font-weight: 600;
        margin: 3px; border: 1px solid;
    }}

    /* ── Divider ── */
    hr {{ border-color: #2D3748; margin: 1.5rem 0; }}

    /* ── Selectbox / input ── */
    .stSelectbox > div, .stNumberInput > div {{ background: {CARD2_BG} !important; }}

    /* ── Tabs styling ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: {CARD2_BG};
        border-radius: 10px;
        padding: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        background: {ACCENT}22 !important;
        color: {ACCENT_LIGHT} !important;
    }}
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MODEL & DATA LOADING
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_bundle() -> dict:
    """Train and cache the model + preprocessing artefacts."""
    df = _load_data()
    df_enc = df.copy()
    le_sex    = LabelEncoder(); df_enc["sex"]    = le_sex.fit_transform(df["sex"])
    le_smoker = LabelEncoder(); df_enc["smoker"] = le_smoker.fit_transform(df["smoker"])
    le_region = LabelEncoder(); df_enc["region"] = le_region.fit_transform(df["region"])

    X = df_enc.drop("charges", axis=1)
    y = df_enc["charges"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    model = LinearRegression()
    model.fit(X_tr_sc, y_tr)
    y_pred = model.predict(X_te_sc)

    r2   = r2_score(y_te, y_pred)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))

    return {
        "model":        model,
        "scaler":       scaler,
        "le_sex":       le_sex,
        "le_smoker":    le_smoker,
        "le_region":    le_region,
        "feature_names": X.columns.tolist(),
        "coef":         dict(zip(X.columns, model.coef_)),
        "intercept":    model.intercept_,
        "metrics":      {"r2": r2, "mae": mae, "rmse": rmse},
        "train_size":   len(X_tr),
        "test_size":    len(X_te),
        "y_test":       y_te.values,
        "y_pred":       y_pred,
        "X_test":       X_te,
    }


@st.cache_data(show_spinner=False)
def _load_data() -> pd.DataFrame:
    """Load the insurance dataset (CSV must be co-located with app.py)."""
    path = DATA_PATH
    if not path.exists():
        # fallback – generate synthetic data so the app doesn't crash
        rng = np.random.default_rng(42)
        n = 1338
        return pd.DataFrame({
            "age":      rng.integers(18, 65, n),
            "sex":      rng.choice(["male","female"], n),
            "bmi":      rng.uniform(16, 54, n).round(2),
            "children": rng.integers(0, 6, n),
            "smoker":   rng.choice(["yes","no"], n, p=[0.2, 0.8]),
            "region":   rng.choice(["northeast","northwest","southeast","southwest"], n),
            "charges":  rng.uniform(1122, 63770, n).round(2),
        })
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# PREDICTION HELPER
# ---------------------------------------------------------------------------
def predict_charges(bundle: dict, inputs: dict) -> tuple[float, float]:
    """Return (predicted_charge, latency_ms)."""
    t0 = time.perf_counter()
    row = pd.DataFrame([{
        "age":      inputs["age"],
        "sex":      bundle["le_sex"].transform([inputs["sex"]])[0],
        "bmi":      inputs["bmi"],
        "children": inputs["children"],
        "smoker":   bundle["le_smoker"].transform([inputs["smoker"]])[0],
        "region":   bundle["le_region"].transform([inputs["region"]])[0],
    }])
    X_sc  = bundle["scaler"].transform(row[bundle["feature_names"]])
    charge = float(bundle["model"].predict(X_sc)[0])
    latency = (time.perf_counter() - t0) * 1000
    return max(charge, 0.0), latency


def feature_contributions(bundle: dict, inputs: dict) -> dict[str, float]:
    """Approximate per-feature contribution using coef * scaled_value."""
    row = pd.DataFrame([{
        "age":      inputs["age"],
        "sex":      bundle["le_sex"].transform([inputs["sex"]])[0],
        "bmi":      inputs["bmi"],
        "children": inputs["children"],
        "smoker":   bundle["le_smoker"].transform([inputs["smoker"]])[0],
        "region":   bundle["le_region"].transform([inputs["region"]])[0],
    }])
    X_sc = bundle["scaler"].transform(row[bundle["feature_names"]])[0]
    contribs = {f: float(bundle["coef"][f] * x) for f, x in zip(bundle["feature_names"], X_sc)}
    return dict(sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True))


# ---------------------------------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------------------------------
def init_session() -> None:
    """Initialise default session-state keys if they don't exist yet."""
    defaults = {
        "page":            "Predict",
        "history":         [],
        "sample_idx":      0,
        "current_inputs":  {},
        "last_prediction": None,
        "last_inputs":     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
def render_header() -> None:
    """Render the sticky top header bar."""
    st.markdown(f"""
    <div class="app-header">
        <div class="app-header-left">
            <span class="app-header-logo">🏥 {APP_NAME}</span>
            <span class="app-header-sub">{APP_SUBTITLE}</span>
        </div>
        <div style="display:flex; gap:8px; align-items:center;">
            <span class="badge">{APP_VERSION}</span>
            <span style="font-size:0.72rem; color:#64748B;">Updated {APP_UPDATED}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SIDEBAR  (redesigned)
# ---------------------------------------------------------------------------
def render_sidebar(bundle: dict) -> str:
    """Render the premium sidebar with pill navigation, live stats, and prediction counter."""
    m = bundle["metrics"]
    r2_pct = max(0, min(100, m["r2"] * 100))
    num_preds = len(st.session_state.get("history", []))

    with st.sidebar:
        # ── Brand / logo area ──
        st.markdown(f"""
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">🏥</div>
            <div class="sidebar-brand-name">{APP_NAME}</div>
            <div class="sidebar-brand-tagline">{APP_SUBTITLE}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # ── Navigation pills ── (using st.button for interactivity)
        st.markdown('<div class="sidebar-nav"><div class="sidebar-nav-label">Navigation</div></div>',
                    unsafe_allow_html=True)

        for icon, label in NAV_ITEMS:
            col_btn, = st.columns(1)
            with col_btn:
                if st.button(
                    f"{icon}  {label}",
                    key=f"nav_{label}",
                    use_container_width=True,
                ):
                    st.session_state["page"] = label
                    st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # ── Prediction counter ──
        st.markdown(f"""
        <div class="sidebar-counter">
            <div class="counter-card">
                <div class="counter-num">{num_preds}</div>
                <div class="counter-label">
                    <strong>Predictions</strong><br/>made this session
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # ── Model stats card ──
        st.markdown(f"""
        <div class="sidebar-stats">
            <div class="sidebar-stats-label">Model Overview</div>
            <div class="sidebar-stat-card">
                <div class="stat-row">
                    <span class="stat-label">
                        <span class="status-dot online"></span>
                        Status
                    </span>
                    <span class="stat-value green">Online</span>
                </div>
                <div class="stat-divider"></div>
                <div class="stat-row">
                    <span class="stat-label">Algorithm</span>
                    <span class="stat-value">Linear Reg.</span>
                </div>
                <div class="stat-divider"></div>
                <div class="stat-row">
                    <span class="stat-label">Features</span>
                    <span class="stat-value accent">{len(bundle["feature_names"])}</span>
                </div>
                <div class="stat-divider"></div>
                <div class="stat-row">
                    <span class="stat-label">Train Rows</span>
                    <span class="stat-value">{bundle["train_size"]:,}</span>
                </div>
                <div class="stat-divider"></div>
                <div class="stat-row">
                    <span class="stat-label">R² Score</span>
                    <span class="stat-value accent">{m["r2"]:.4f}</span>
                </div>
                <div class="r2-bar-bg">
                    <div class="r2-bar-fill" style="width:{r2_pct:.1f}%;"></div>
                </div>
                <div class="stat-divider"></div>
                <div class="stat-row">
                    <span class="stat-label">MAE</span>
                    <span class="stat-value warn">${m["mae"]:,.0f}</span>
                </div>
                <div class="stat-divider"></div>
                <div class="stat-row">
                    <span class="stat-label">RMSE</span>
                    <span class="stat-value warn">${m["rmse"]:,.0f}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # ── Footer ──
        st.markdown(f"""
        <div class="sidebar-footer">
            <p>Built with ❤️ using <a href="https://streamlit.io" target="_blank">Streamlit</a></p>
            <p>© 2025 {APP_NAME} · {APP_VERSION}</p>
        </div>
        """, unsafe_allow_html=True)

    return st.session_state.get("page", "Predict")


# ---------------------------------------------------------------------------
# PAGE 1 — PREDICT  (enhanced with interactive charts)
# ---------------------------------------------------------------------------
def page_predict(bundle: dict) -> None:
    """Main prediction page with gauge chart, waterfall, radar, and distribution overlay."""
    df = _load_data()

    st.markdown("## 🏠 Insurance Charges Predictor")
    st.markdown("<p style='color:#94A3B8; margin-top:-0.5rem;'>Enter policyholder details below to estimate annual insurance charges.</p>", unsafe_allow_html=True)

    # ── INPUT FORM ────────────────────────────────────────────────────────
    col_l, col_r = st.columns([1.1, 0.9], gap="large")

    with col_l:
        st.markdown('<div class="card card-accent">', unsafe_allow_html=True)
        st.markdown("### 📝 Policyholder Details")

        # Action buttons row
        b1, b2, b3 = st.columns(3)
        run_predict  = b1.button("🔮 Predict",      use_container_width=True)
        load_sample  = b2.button("📋 Load Sample",  use_container_width=True)
        reset_inputs = b3.button("🔄 Reset",         use_container_width=True)

        st.markdown("---")

        # Load sample logic
        if load_sample:
            idx = st.session_state["sample_idx"] % len(SAMPLE_ROWS)
            st.session_state["current_inputs"] = SAMPLE_ROWS[idx].copy()
            st.session_state["sample_idx"] += 1
        if reset_inputs:
            st.session_state["current_inputs"] = {}

        ci = st.session_state.get("current_inputs", {})

        r1c1, r1c2 = st.columns(2)
        age      = r1c1.number_input("🎂 Age",      min_value=18, max_value=64, step=1,
                                      value=int(ci.get("age", 30)),
                                      help=FEATURE_HELP["age"])
        bmi      = r1c2.number_input("⚖️ BMI",       min_value=10.0, max_value=60.0, step=0.1, format="%.2f",
                                      value=float(ci.get("bmi", 25.0)),
                                      help=FEATURE_HELP["bmi"])

        r2c1, r2c2 = st.columns(2)
        children = r2c1.selectbox("👶 Children", options=[0,1,2,3,4,5],
                                   index=int(ci.get("children", 0)),
                                   help=FEATURE_HELP["children"])
        sex      = r2c2.selectbox("🧬 Sex", options=["female","male"],
                                   index=["female","male"].index(ci.get("sex","female")),
                                   help=FEATURE_HELP["sex"])

        r3c1, r3c2 = st.columns(2)
        smoker   = r3c1.selectbox("🚬 Smoker", options=["no","yes"],
                                   index=["no","yes"].index(ci.get("smoker","no")),
                                   help=FEATURE_HELP["smoker"])
        region   = r3c2.selectbox("🌎 Region",
                                   options=["northeast","northwest","southeast","southwest"],
                                   index=["northeast","northwest","southeast","southwest"].index(ci.get("region","northeast")),
                                   help=FEATURE_HELP["region"])

        # ── Validation warnings
        if bmi > 40:
            st.warning("⚠️ BMI > 40 is in the extreme range — prediction may be less reliable.")
        if age < 20:
            st.info("ℹ️ Very young policyholders are less common in this dataset.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ── RESULTS PANEL ──────────────────────────────────────────────────────
    with col_r:
        inputs = {"age": age, "sex": sex, "bmi": bmi,
                  "children": children, "smoker": smoker, "region": region}

        if run_predict:
            with st.spinner("Running model…"):
                charge, latency = predict_charges(bundle, inputs)

            st.session_state["last_prediction"] = charge
            st.session_state["last_inputs"]     = inputs.copy()

            # Persist in history
            st.session_state["history"].append({
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "charge":    charge,
                "smoker":    smoker,
                "age":       age,
                "bmi":       bmi,
                "region":    region,
                "latency":   latency,
            })

        # Display result
        pred   = st.session_state.get("last_prediction")
        inputs_saved = st.session_state.get("last_inputs", inputs)

        if pred is not None:
            high_risk = pred > 15000
            color     = DANGER_COLOR if high_risk else SUCCESS_COLOR

            # ── GAUGE CHART — replaces the static result card ──
            max_charge = float(df["charges"].max())
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pred,
                number={"prefix": "$", "font": {"size": 36, "color": color},
                         "valueformat": ",.0f"},
                delta={"reference": float(df["charges"].mean()), "prefix": "$",
                       "valueformat": ",.0f",
                       "increasing": {"color": DANGER_COLOR},
                       "decreasing": {"color": SUCCESS_COLOR}},
                title={"text": "🔴 High Cost Risk" if high_risk else "🟢 Low Cost Profile",
                       "font": {"size": 14, "color": "#94A3B8"}},
                gauge={
                    "axis": {"range": [0, max_charge], "tickprefix": "$",
                             "tickformat": ",.0s", "tickcolor": "#475569",
                             "dtick": max_charge / 5},
                    "bar": {"color": color, "thickness": 0.7},
                    "bgcolor": CARD2_BG,
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 10000], "color": "rgba(16,185,129,0.08)"},
                        {"range": [10000, 25000], "color": "rgba(245,158,11,0.08)"},
                        {"range": [25000, max_charge], "color": "rgba(239,68,68,0.08)"},
                    ],
                    "threshold": {
                        "line": {"color": WARN_COLOR, "width": 3},
                        "thickness": 0.8,
                        "value": float(df["charges"].mean()),
                    },
                },
            ))
            _apply_defaults(gauge_fig, height=260,
                            margin=dict(l=30, r=30, t=60, b=10))
            st.plotly_chart(gauge_fig, use_container_width=True, config=PLOTLY_CONFIG)

            # KPI metrics
            m    = bundle["metrics"]
            lat  = st.session_state["history"][-1]["latency"] if st.session_state["history"] else 0
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Model R²",       f"{m['r2']:.3f}")
            mc2.metric("Avg MAE",        f"${m['mae']:,.0f}")
            mc3.metric("Latency",        f"{lat:.1f} ms")

            # ── WATERFALL — shows how each feature builds up to the prediction ──
            contribs = feature_contributions(bundle, inputs_saved)
            wf_labels = list(contribs.keys()) + ["Total"]
            wf_values = list(contribs.values())
            wf_measures = ["relative"] * len(contribs) + ["total"]
            wf_text = [f"${v:+,.0f}" for v in wf_values] + [f"${pred:,.0f}"]

            wf_fig = go.Figure(go.Waterfall(
                orientation="v",
                measure=wf_measures,
                x=wf_labels,
                y=wf_values + [0],  # total auto-calculated
                text=wf_text,
                textposition="outside",
                connector={"line": {"color": "#2D3748", "width": 1.5}},
                increasing={"marker": {"color": DANGER_COLOR}},
                decreasing={"marker": {"color": SUCCESS_COLOR}},
                totals={"marker": {"color": ACCENT_LIGHT}},
                textfont=dict(size=10, color="#94A3B8"),
            ))
            _apply_defaults(wf_fig,
                title="Feature Contribution Waterfall",
                height=300,
                margin=dict(l=10, r=10, t=50, b=10),
                xaxis=dict(tickfont=dict(size=10)),
                yaxis=dict(showgrid=True, gridcolor="rgba(45,55,72,0.2)",
                           tickprefix="$", tickformat=",.0s"),
            )
            st.plotly_chart(wf_fig, use_container_width=True, config=PLOTLY_CONFIG)

            # Plain-English explanation
            with st.expander("💡 Explanation", expanded=False):
                top3 = list(contribs.items())[:3]
                st.markdown("**Top 3 drivers of this prediction:**")
                icons = {True: "📈 Increases", False: "📉 Decreases"}
                for feat, val in top3:
                    direction = icons[val >= 0]
                    st.markdown(f"- **{feat.title()}** ({inputs_saved.get(feat,'?')}) — {direction} estimate by **${abs(val):,.0f}**")

            # Download report
            report_df = pd.DataFrame([{
                "Feature": k, "Value": v
            } for k, v in inputs_saved.items()] + [{"Feature": "Predicted Charge ($)", "Value": f"${pred:,.2f}"}])
            csv_bytes = report_df.to_csv(index=False).encode()
            st.download_button("⬇️ Download Prediction Report", data=csv_bytes,
                               file_name="insureiq_prediction.csv", mime="text/csv",
                               use_container_width=True)
        else:
            st.markdown(f"""
            <div class="card" style="text-align:center; padding:3rem 1rem;">
                <div style="font-size:3rem;">🔮</div>
                <div style="color:#94A3B8; margin-top:0.5rem;">Fill in the form and click <strong>Predict</strong></div>
            </div>
            """, unsafe_allow_html=True)

    # ── INTERACTIVE COMPARISON SECTION ─────────────────────────────────────
    st.markdown("---")

    if pred is not None:
        comp_tabs = st.tabs(["📊 Where You Stand", "🎯 Risk Threshold", "🕐 History"])

        # Tab 1: Distribution overlay showing where prediction falls
        with comp_tabs[0]:
            st.markdown("#### Your Prediction vs. Dataset Distribution")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                hist_fig = go.Figure()
                hist_fig.add_trace(go.Histogram(
                    x=df["charges"], nbinsx=50,
                    marker_color=ACCENT,
                    opacity=0.6, name="All Records",
                    hovertemplate="Range: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
                ))
                # Mark the prediction
                hist_fig.add_vline(x=pred, line_dash="dash", line_color=color, line_width=2.5,
                                   annotation_text=f"  Your: ${pred:,.0f}",
                                   annotation_font_color=color,
                                   annotation_font_size=12)
                # Mark the mean
                hist_fig.add_vline(x=float(df["charges"].mean()), line_dash="dot",
                                   line_color="#94A3B8", line_width=1.5,
                                   annotation_text=f"  Avg: ${df['charges'].mean():,.0f}",
                                   annotation_font_color="#94A3B8",
                                   annotation_font_size=11,
                                   annotation_position="top left")
                _apply_defaults(hist_fig,
                    title="Charges Distribution — Where You Fall",
                    xaxis_title="Annual Charges ($)",
                    yaxis_title="Number of Policyholders",
                    height=340, margin=dict(t=50, b=40),
                    xaxis=dict(tickprefix="$", tickformat=",.0s"),
                )
                st.plotly_chart(hist_fig, use_container_width=True, config=PLOTLY_CONFIG)

            with col_d2:
                # Percentile info
                percentile = float((df["charges"] < pred).mean() * 100)

                # Donut chart for percentile
                donut_fig = go.Figure(go.Pie(
                    values=[percentile, 100 - percentile],
                    labels=["Below Your Cost", "Above Your Cost"],
                    hole=0.7,
                    marker=dict(colors=[color, "#2D3748"]),
                    textinfo="none",
                    hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
                ))
                donut_fig.add_annotation(
                    text=f"<b>{percentile:.0f}%</b><br><span style='font-size:11px;color:#94A3B8'>Percentile</span>",
                    showarrow=False, font=dict(size=28, color=color),
                )
                _apply_defaults(donut_fig,
                    title="Your Cost Percentile",
                    height=340, margin=dict(t=50, b=10, l=10, r=10),
                    showlegend=True,
                    legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
                )
                st.plotly_chart(donut_fig, use_container_width=True, config=PLOTLY_CONFIG)

        # Tab 2: Risk threshold
        with comp_tabs[1]:
            st.markdown("#### Customise Your Risk Threshold")
            threshold = st.slider("💰 High-Risk Threshold ($)", 5000, 40000, 15000, 500,
                                   help="Charges above this value are flagged as 'High Cost'")
            if pred > threshold:
                st.warning(f"⚠️ Predicted charge ${pred:,.2f} exceeds your threshold of ${threshold:,}")

            # Bullet chart with threshold
            bullet_fig = go.Figure(go.Indicator(
                mode="number+gauge",
                value=pred,
                number={"prefix": "$", "font": {"size": 24, "color": "#E2E8F0"},
                         "valueformat": ",.0f"},
                gauge={
                    "shape": "bullet",
                    "axis": {"range": [0, 60000], "tickprefix": "$", "tickformat": ",.0s"},
                    "bar": {"color": color, "thickness": 0.6},
                    "bgcolor": CARD2_BG,
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, threshold], "color": "rgba(16,185,129,0.09)"},
                        {"range": [threshold, 60000], "color": "rgba(239,68,68,0.09)"},
                    ],
                    "threshold": {
                        "line": {"color": WARN_COLOR, "width": 3},
                        "thickness": 0.9,
                        "value": threshold,
                    },
                },
            ))
            _apply_defaults(bullet_fig,
                height=130,
                margin=dict(l=100, r=30, t=20, b=20),
            )
            st.plotly_chart(bullet_fig, use_container_width=True, config=PLOTLY_CONFIG)

        # Tab 3: Prediction history
        with comp_tabs[2]:
            if st.session_state["history"]:
                st.markdown("#### Session Predictions Timeline")
                hist_data = st.session_state["history"]
                hist_df = pd.DataFrame(hist_data)

                # Timeline scatter
                timeline_fig = go.Figure()
                timeline_fig.add_trace(go.Scatter(
                    x=list(range(1, len(hist_data) + 1)),
                    y=[h["charge"] for h in hist_data],
                    mode="lines+markers+text",
                    text=[f"${h['charge']:,.0f}" for h in hist_data],
                    textposition="top center",
                    textfont=dict(size=10, color="#94A3B8"),
                    marker=dict(
                        size=12,
                        color=[DANGER_COLOR if h["charge"] > 15000 else SUCCESS_COLOR for h in hist_data],
                        line=dict(width=2, color="#0F1117"),
                    ),
                    line=dict(color=ACCENT_LIGHT, width=2),
                    hovertemplate=(
                        "<b>Prediction #%{x}</b><br>"
                        "Charge: $%{y:,.2f}<br>"
                        "<extra></extra>"
                    ),
                ))
                _apply_defaults(timeline_fig,
                    title="Prediction History",
                    xaxis_title="Prediction #",
                    yaxis_title="Predicted Charge ($)",
                    height=300,
                    margin=dict(t=50, b=40),
                    yaxis=dict(tickprefix="$", tickformat=",.0s",
                               showgrid=True, gridcolor="rgba(45,55,72,0.2)"),
                    xaxis=dict(dtick=1),
                )
                st.plotly_chart(timeline_fig, use_container_width=True, config=PLOTLY_CONFIG)

                # Data table
                disp_df = hist_df.copy()
                disp_df["charge"] = disp_df["charge"].apply(lambda x: f"${x:,.2f}")
                disp_df["latency"] = disp_df["latency"].apply(lambda x: f"{x:.1f} ms")
                st.dataframe(disp_df, use_container_width=True, hide_index=True)
                st.download_button("⬇️ Export History CSV",
                                   data=pd.DataFrame(hist_data).to_csv(index=False).encode(),
                                   file_name="insureiq_history.csv", mime="text/csv")
            else:
                st.info("No predictions yet — run a prediction to see your history here.")
    else:
        # No prediction yet — show threshold slider standalone
        col_t, _ = st.columns([1, 1])
        with col_t:
            threshold = st.slider("💰 High-Risk Threshold ($)", 5000, 40000, 15000, 500,
                                   help="Charges above this value are flagged as 'High Cost'")


# ---------------------------------------------------------------------------
# PAGE 2 — DATA EXPLORER  (massively enhanced)
# ---------------------------------------------------------------------------
def page_data_explorer() -> None:
    """Interactive data explorer with scatter matrix, violin, sunburst, 3D scatter, and more."""
    df = _load_data()
    st.markdown("## 📊 Data Explorer")
    st.markdown(f"<p style='color:#94A3B8;'>Dataset: **{len(df):,} records × {len(df.columns)} features** — Insurance Charges Benchmark</p>",
                unsafe_allow_html=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # ── KPI Row ──
    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Total Records", f"{len(df):,}")
    kc2.metric("Avg Charges",   f"${df['charges'].mean():,.0f}")
    kc3.metric("Median BMI",    f"{df['bmi'].median():.1f}")
    kc4.metric("Smoker %",      f"{df['smoker'].value_counts(normalize=True).get('yes', 0)*100:.1f}%")

    st.markdown("---")

    # ── TABBED SECTIONS ──
    tabs = st.tabs(["🔍 Filter & Explore", "📈 Interactive Charts", "🌐 Advanced Visuals", "📐 Statistics"])

    # ─── TAB 1: Filter & Explore ─────────────────────────────────────────
    with tabs[0]:
        st.markdown("### 🔍 Filter & Explore")
        all_cols = df.columns.tolist()
        sel_cols = st.multiselect("Select columns to display", all_cols, default=all_cols)
        filtered = df[sel_cols] if sel_cols else df

        # Categorical filters
        exp = st.expander("⚙️ Row Filters", expanded=False)
        with exp:
            fc1, fc2, fc3 = st.columns(3)
            sel_sex    = fc1.multiselect("Sex",    ["female","male"],               default=["female","male"])
            sel_smoker = fc2.multiselect("Smoker", ["yes","no"],                    default=["yes","no"])
            sel_region = fc3.multiselect("Region", ["northeast","northwest","southeast","southwest"],
                                         default=["northeast","northwest","southeast","southwest"])
            min_age, max_age = fc1.slider("Age range", 18, 64, (18, 64))
            min_bmi, max_bmi = fc2.slider("BMI range", float(df.bmi.min()), float(df.bmi.max()),
                                           (float(df.bmi.min()), float(df.bmi.max())), step=0.1)
            min_c, max_c = fc3.slider("Charges ($)", float(df.charges.min()), float(df.charges.max()),
                                       (float(df.charges.min()), float(df.charges.max())), step=100.0)

        mask = (
            df["sex"].isin(sel_sex) & df["smoker"].isin(sel_smoker) & df["region"].isin(sel_region) &
            df["age"].between(min_age, max_age) & df["bmi"].between(min_bmi, max_bmi) &
            df["charges"].between(min_c, max_c)
        )
        view_df = filtered[mask] if sel_cols else df[mask]

        st.dataframe(view_df, use_container_width=True, height=340)
        st.caption(f"Showing {len(view_df):,} of {len(df):,} rows")

        st.download_button("⬇️ Download Filtered Data", data=view_df.to_csv(index=False).encode(),
                           file_name="insureiq_filtered.csv", mime="text/csv")

    # ─── TAB 2: Interactive Charts ───────────────────────────────────────
    with tabs[1]:
        st.markdown("### 📈 Interactive Charts")
        chart_type = st.selectbox("Choose chart type", [
            "Scatter Plot", "Violin Plot", "Box Plot", "Histogram", "Strip Plot",
        ], key="chart_type_sel")

        ct1, ct2, ct3 = st.columns(3)
        x_axis = ct1.selectbox("X Axis", df.columns.tolist(), index=0, key="x_sel")
        y_axis = ct2.selectbox("Y Axis", df.columns.tolist(),
                                index=df.columns.tolist().index("charges"), key="y_sel")
        color_by = ct3.selectbox("Color By", ["None"] + cat_cols, index=0, key="color_sel")
        color_col = None if color_by == "None" else color_by

        if chart_type == "Scatter Plot":
            fig = px.scatter(
                df, x=x_axis, y=y_axis, color=color_col,
                color_discrete_sequence=PALETTE,
                opacity=0.65, size_max=10,
                hover_data=df.columns.tolist(),
                trendline="ols" if (df[x_axis].dtype in [np.int64, np.float64] and
                                     df[y_axis].dtype in [np.int64, np.float64]) else None,
            )
        elif chart_type == "Violin Plot":
            group_col = ct1.selectbox("Group by", cat_cols, key="violin_grp") if cat_cols else None
            fig = px.violin(
                df, x=group_col, y=y_axis, color=color_col,
                color_discrete_sequence=PALETTE,
                box=True, points="outliers",
                hover_data=["age", "bmi", "charges"],
            )
        elif chart_type == "Box Plot":
            group_col = x_axis if x_axis in cat_cols else (cat_cols[0] if cat_cols else x_axis)
            fig = px.box(
                df, x=group_col, y=y_axis, color=color_col,
                color_discrete_sequence=PALETTE,
                notched=True,
                hover_data=["age", "bmi", "charges"],
            )
        elif chart_type == "Histogram":
            nbins = st.slider("Number of bins", 10, 80, 30, key="hist_bins")
            fig = px.histogram(
                df, x=x_axis, nbins=nbins, color=color_col,
                color_discrete_sequence=PALETTE,
                marginal="rug",
                barmode="overlay", opacity=0.7,
            )
        elif chart_type == "Strip Plot":
            group_col = x_axis if x_axis in cat_cols else (cat_cols[0] if cat_cols else x_axis)
            fig = px.strip(
                df, x=group_col, y=y_axis, color=color_col,
                color_discrete_sequence=PALETTE,
                hover_data=["age", "bmi", "charges"],
            )
        else:
            fig = go.Figure()

        _apply_defaults(fig,
            title=f"{chart_type}: {x_axis} vs {y_axis}",
            height=450, margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # ─── TAB 3: Advanced Visuals ─────────────────────────────────────────
    with tabs[2]:
        st.markdown("### 🌐 Advanced Visualisations")
        adv_col1, adv_col2 = st.columns(2)

        # Sunburst
        with adv_col1:
            st.markdown("#### 🌞 Sunburst — Charges Breakdown")
            # Create binned charges column for sunburst
            df_sb = df.copy()
            df_sb["cost_tier"] = pd.cut(df_sb["charges"],
                                         bins=[0, 10000, 25000, float("inf")],
                                         labels=["Low (<$10k)", "Medium ($10k–25k)", "High (>$25k)"]).astype(str)
            sun_fig = px.sunburst(
                df_sb, path=["smoker", "sex", "cost_tier"],
                values="charges",
                color="cost_tier",
                color_discrete_map={
                    "Low (<$10k)": SUCCESS_COLOR,
                    "Medium ($10k–25k)": WARN_COLOR,
                    "High (>$25k)": DANGER_COLOR,
                },
                hover_data={"charges": ":$,.0f"},
            )
            _apply_defaults(sun_fig, height=420, margin=dict(t=30, b=10))
            st.plotly_chart(sun_fig, use_container_width=True, config=PLOTLY_CONFIG)

        # Treemap
        with adv_col2:
            st.markdown("#### 🗺️ Treemap — Regional Cost Breakdown")
            tree_fig = px.treemap(
                df, path=["region", "smoker", "sex"],
                values="charges",
                color="charges",
                color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
                hover_data={"charges": ":$,.0f"},
            )
            _apply_defaults(tree_fig, height=420, margin=dict(t=30, b=10))
            st.plotly_chart(tree_fig, use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("---")

        # 3D Scatter
        st.markdown("#### 🔮 3D Interactive Scatter")
        sc1, sc2, sc3 = st.columns(3)
        x3d = sc1.selectbox("X Axis (3D)", num_cols, index=0, key="x3d")
        y3d = sc2.selectbox("Y Axis (3D)", num_cols,
                             index=num_cols.index("bmi") if "bmi" in num_cols else 1, key="y3d")
        z3d = sc3.selectbox("Z Axis (3D)", num_cols,
                             index=num_cols.index("charges") if "charges" in num_cols else 2, key="z3d")
        color_3d = st.selectbox("Color by (3D)", cat_cols, index=0, key="c3d") if cat_cols else None

        fig3d = px.scatter_3d(
            df, x=x3d, y=y3d, z=z3d, color=color_3d,
            color_discrete_sequence=PALETTE,
            opacity=0.7, size_max=8,
            hover_data=df.columns.tolist(),
        )
        _apply_defaults(fig3d,
            height=550, margin=dict(t=30, b=10, l=10, r=10),
            scene=dict(
                xaxis=dict(backgroundcolor=CARD2_BG, gridcolor="#2D3748",
                           title=x3d),
                yaxis=dict(backgroundcolor=CARD2_BG, gridcolor="#2D3748",
                           title=y3d),
                zaxis=dict(backgroundcolor=CARD2_BG, gridcolor="#2D3748",
                           title=z3d),
            ),
        )
        st.plotly_chart(fig3d, use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("---")

        # Parallel Coordinates
        st.markdown("#### 🧵 Parallel Coordinates — Multi-Feature Relationships")
        df_par = df.copy()
        df_par["smoker_num"] = (df_par["smoker"] == "yes").astype(int)
        df_par["sex_num"]    = (df_par["sex"] == "male").astype(int)

        par_fig = go.Figure(go.Parcoords(
            line=dict(
                color=df_par["charges"],
                colorscale=[[0, SUCCESS_COLOR], [0.5, WARN_COLOR], [1, DANGER_COLOR]],
                showscale=True,
                colorbar=dict(title="Charges ($)", tickprefix="$", tickformat=",.0s"),
            ),
            dimensions=[
                dict(label="Age",      values=df_par["age"],        range=[18, 64]),
                dict(label="BMI",      values=df_par["bmi"],        range=[df.bmi.min(), df.bmi.max()]),
                dict(label="Children", values=df_par["children"],   range=[0, 5],  tickvals=[0,1,2,3,4,5]),
                dict(label="Smoker",   values=df_par["smoker_num"], range=[0, 1],  tickvals=[0, 1],
                     ticktext=["No", "Yes"]),
                dict(label="Sex",      values=df_par["sex_num"],    range=[0, 1],  tickvals=[0, 1],
                     ticktext=["F", "M"]),
                dict(label="Charges",  values=df_par["charges"],
                     range=[df.charges.min(), df.charges.max()]),
            ],
        ))
        _apply_defaults(par_fig,
            title="Drag axes to filter — explore multi-dimensional patterns",
            height=420, margin=dict(t=60, b=30, l=60, r=30),
        )
        st.plotly_chart(par_fig, use_container_width=True, config=PLOTLY_CONFIG)

    # ─── TAB 4: Statistics ───────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### 📐 Summary Statistics")
        stats = df[num_cols].describe().T
        stats["null_count"] = df[num_cols].isnull().sum()
        stats = stats.round(2)
        st.dataframe(stats, use_container_width=True)

        st.markdown("---")

        # Interactive Correlation Heatmap
        st.markdown("### 🔥 Interactive Correlation Heatmap")
        corr_method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"], key="corr_method")
        df_num = df[num_cols].copy()
        corr   = df_num.corr(method=corr_method).round(3)

        heatmap_fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale=[[0, DANGER_COLOR], [0.5, "#1E2130"], [1, SUCCESS_COLOR]],
            zmin=-1, zmax=1,
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=13),
            hovertemplate="<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
        ))
        _apply_defaults(heatmap_fig,
            title=f"Correlation Matrix ({corr_method.title()})",
            height=400, margin=dict(t=50, b=30),
            xaxis=dict(side="bottom"),
        )
        st.plotly_chart(heatmap_fig, use_container_width=True, config=PLOTLY_CONFIG)

        st.markdown("---")

        # Pairwise scatter matrix
        st.markdown("### 🔗 Pairwise Scatter Matrix")
        pair_cols = st.multiselect("Select features for scatter matrix",
                                    num_cols, default=num_cols[:3] + (["charges"] if "charges" not in num_cols[:3] else []),
                                    key="pair_cols")
        if len(pair_cols) >= 2:
            pair_color = st.selectbox("Color by (pair)", ["None"] + cat_cols, key="pair_color")
            pair_fig = px.scatter_matrix(
                df, dimensions=pair_cols,
                color=None if pair_color == "None" else pair_color,
                color_discrete_sequence=PALETTE,
                opacity=0.5,
                hover_data=["charges"],
            )
            _apply_defaults(pair_fig,
                height=max(400, 150 * len(pair_cols)),
                margin=dict(t=30, b=10),
            )
            pair_fig.update_traces(diagonal_visible=True, showupperhalf=False,
                                    marker=dict(size=3))
            st.plotly_chart(pair_fig, use_container_width=True, config=PLOTLY_CONFIG)
        else:
            st.info("Select at least 2 features to display the scatter matrix.")


# ---------------------------------------------------------------------------
# PAGE 3 — MODEL PERFORMANCE  (enhanced)
# ---------------------------------------------------------------------------
def page_model_performance(bundle: dict) -> None:
    """Model diagnostics with interactive charts, error analysis, and sensitivity."""
    df = _load_data()
    m = bundle["metrics"]

    st.markdown("## 📈 Model Performance")
    st.markdown("<p style='color:#94A3B8;'>Comprehensive diagnostics for the trained Linear Regression model.</p>",
                unsafe_allow_html=True)

    # KPI row with gauges
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("R² Score",   f"{m['r2']:.4f}",    delta="Goodness of fit")
    kpi2.metric("MAE ($)",    f"{m['mae']:,.0f}",   delta="Mean Abs Error")
    kpi3.metric("RMSE ($)",   f"{m['rmse']:,.0f}",  delta="Root Mean Sq Err")
    kpi4.metric("Train Rows", f"{bundle['train_size']:,}")

    st.markdown("---")

    y_test = bundle["y_test"]
    y_pred = bundle["y_pred"]
    residuals = y_test - y_pred
    abs_errors = np.abs(residuals)

    # ── TABBED DIAGNOSTICS ──
    perf_tabs = st.tabs(["📊 Actual vs Predicted", "📉 Residual Analysis",
                          "🏋️ Feature Importance", "🔬 Error Deep Dive", "🎚️ Sensitivity"])

    # ─── Tab 1: Actual vs Predicted ──────────────────────────────────────
    with perf_tabs[0]:
        col1, col2 = st.columns(2)

        with col1:
            # Interactive scatter with hover showing all details
            avp_fig = go.Figure()
            avp_fig.add_trace(go.Scatter(
                x=y_test, y=y_pred,
                mode="markers",
                marker=dict(
                    size=7, opacity=0.6,
                    color=abs_errors,
                    colorscale=[[0, SUCCESS_COLOR], [0.5, WARN_COLOR], [1, DANGER_COLOR]],
                    colorbar=dict(title="Abs Error ($)", tickprefix="$", tickformat=",.0s",
                                  thickness=15, len=0.8),
                    line=dict(width=0),
                ),
                hovertemplate=(
                    "<b>Actual:</b> $%{x:,.0f}<br>"
                    "<b>Predicted:</b> $%{y:,.0f}<br>"
                    "<b>Error:</b> $%{customdata[0]:,.0f}<br>"
                    "<extra></extra>"
                ),
                customdata=np.stack([residuals], axis=-1),
                name="Test Samples",
            ))
            # Perfect fit line
            line_range = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            avp_fig.add_trace(go.Scatter(
                x=line_range, y=line_range, mode="lines",
                line=dict(color=WARN_COLOR, dash="dash", width=2),
                name="Perfect Fit",
            ))
            _apply_defaults(avp_fig,
                title="Actual vs Predicted (colored by error magnitude)",
                xaxis_title="Actual Charges ($)",
                yaxis_title="Predicted Charges ($)",
                height=420, margin=dict(t=60, b=40),
                xaxis=dict(tickprefix="$", tickformat=",.0s",
                           showgrid=True, gridcolor="rgba(45,55,72,0.2)"),
                yaxis=dict(tickprefix="$", tickformat=",.0s",
                           showgrid=True, gridcolor="rgba(45,55,72,0.2)"),
            )
            st.plotly_chart(avp_fig, use_container_width=True, config=PLOTLY_CONFIG)

        with col2:
            # Cumulative error distribution
            sorted_errors = np.sort(abs_errors)
            cumulative_pct = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100

            cum_fig = go.Figure()
            cum_fig.add_trace(go.Scatter(
                x=sorted_errors, y=cumulative_pct,
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(124,58,237,0.08)",
                line=dict(color=ACCENT_LIGHT, width=2.5),
                hovertemplate="Error ≤ $%{x:,.0f} → %{y:.1f}% of samples<extra></extra>",
                name="Cumulative",
            ))
            # Mark key thresholds
            for pct_val in [50, 75, 90]:
                idx = np.searchsorted(cumulative_pct, pct_val)
                if idx < len(sorted_errors):
                    err_val = sorted_errors[idx]
                    cum_fig.add_annotation(
                        x=err_val, y=pct_val,
                        text=f"P{pct_val}: ${err_val:,.0f}",
                        showarrow=True, arrowhead=2,
                        font=dict(size=10, color=ACCENT_LIGHT),
                        arrowcolor=ACCENT_LIGHT,
                    )
            _apply_defaults(cum_fig,
                title="Cumulative Error Distribution (CDF)",
                xaxis_title="Absolute Error ($)",
                yaxis_title="% of Samples",
                height=420, margin=dict(t=60, b=40),
                xaxis=dict(tickprefix="$", tickformat=",.0s",
                           showgrid=True, gridcolor="rgba(45,55,72,0.2)"),
                yaxis=dict(ticksuffix="%",
                           showgrid=True, gridcolor="rgba(45,55,72,0.2)"),
            )
            st.plotly_chart(cum_fig, use_container_width=True, config=PLOTLY_CONFIG)

    # ─── Tab 2: Residual Analysis ────────────────────────────────────────
    with perf_tabs[1]:
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            # Residuals histogram with KDE
            res_fig = go.Figure()
            res_fig.add_trace(go.Histogram(
                x=residuals, nbinsx=50,
                marker_color=ACCENT, opacity=0.7,
                name="Residuals",
                hovertemplate="Residual: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
            ))
            res_fig.add_vline(x=0, line_dash="dash", line_color=WARN_COLOR, line_width=2)
            res_fig.add_vline(x=float(np.mean(residuals)), line_dash="dot",
                               line_color=INFO_COLOR, line_width=1.5,
                               annotation_text=f"Mean: ${np.mean(residuals):,.0f}",
                               annotation_font_color=INFO_COLOR)
            _apply_defaults(res_fig,
                title="Residuals Distribution",
                xaxis_title="Residual ($)",
                yaxis_title="Count",
                height=380, margin=dict(t=60, b=40),
                xaxis=dict(tickprefix="$", tickformat=",.0s"),
            )
            st.plotly_chart(res_fig, use_container_width=True, config=PLOTLY_CONFIG)

        with col_r2:
            # Residuals vs Predicted — homoscedasticity
            rvp_fig = go.Figure()
            rvp_fig.add_trace(go.Scatter(
                x=y_pred, y=residuals,
                mode="markers",
                marker=dict(size=6, opacity=0.5, color=ACCENT_LIGHT),
                hovertemplate=(
                    "Predicted: $%{x:,.0f}<br>"
                    "Residual: $%{y:,.0f}<extra></extra>"
                ),
                name="Residuals",
            ))
            rvp_fig.add_hline(y=0, line_dash="dash", line_color=WARN_COLOR, line_width=1.5)
            # ±1 std band
            std_r = float(np.std(residuals))
            rvp_fig.add_hrect(y0=-std_r, y1=std_r,
                               fillcolor="rgba(124,58,237,0.03)", line_width=0,
                               annotation_text="±1σ", annotation_position="top left",
                               annotation_font_color="#475569")
            _apply_defaults(rvp_fig,
                title="Residuals vs Predicted (Homoscedasticity)",
                xaxis_title="Predicted ($)",
                yaxis_title="Residual ($)",
                height=380, margin=dict(t=60, b=40),
                xaxis=dict(tickprefix="$", tickformat=",.0s",
                           showgrid=True, gridcolor="rgba(45,55,72,0.2)"),
                yaxis=dict(tickprefix="$", tickformat=",.0s",
                           showgrid=True, gridcolor="rgba(45,55,72,0.2)"),
            )
            st.plotly_chart(rvp_fig, use_container_width=True, config=PLOTLY_CONFIG)

        # QQ Plot
        st.markdown("---")
        st.markdown("#### Q-Q Plot (Normality of Residuals)")
        sorted_res = np.sort(residuals)
        n = len(sorted_res)
        theoretical_q = np.array([float(np.percentile(np.random.standard_normal(10000), 100 * (i + 0.5) / n))
                                   for i in range(n)])

        qq_fig = go.Figure()
        qq_fig.add_trace(go.Scatter(
            x=theoretical_q, y=sorted_res,
            mode="markers",
            marker=dict(size=5, color=ACCENT_LIGHT, opacity=0.6),
            name="Residuals",
            hovertemplate="Theoretical: %{x:.2f}<br>Sample: $%{y:,.0f}<extra></extra>",
        ))
        # Reference line
        qq_min, qq_max = theoretical_q.min(), theoretical_q.max()
        qq_slope = np.std(residuals)
        qq_intercept = np.mean(residuals)
        qq_fig.add_trace(go.Scatter(
            x=[qq_min, qq_max],
            y=[qq_intercept + qq_slope * qq_min, qq_intercept + qq_slope * qq_max],
            mode="lines",
            line=dict(color=WARN_COLOR, dash="dash", width=2),
            name="Normal Reference",
        ))
        _apply_defaults(qq_fig,
            title="Q-Q Plot — Do residuals follow a normal distribution?",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles ($)",
            height=380, margin=dict(t=60, b=40),
            yaxis=dict(tickprefix="$", tickformat=",.0s"),
        )
        st.plotly_chart(qq_fig, use_container_width=True, config=PLOTLY_CONFIG)

    # ─── Tab 3: Feature Importance ───────────────────────────────────────
    with perf_tabs[2]:
        coef_df = pd.DataFrame({
            "Feature":     list(bundle["coef"].keys()),
            "Coefficient": list(bundle["coef"].values()),
            "Abs_Coef":    [abs(v) for v in bundle["coef"].values()],
        }).sort_values("Abs_Coef", ascending=False)

        col_f1, col_f2 = st.columns(2)

        with col_f1:
            # Horizontal bar chart
            bar_colors = [SUCCESS_COLOR if v >= 0 else DANGER_COLOR for v in coef_df["Coefficient"]]
            feat_fig = go.Figure(go.Bar(
                y=coef_df["Feature"], x=coef_df["Coefficient"],
                orientation="h", marker_color=bar_colors,
                text=coef_df["Coefficient"].apply(lambda x: f"{x:+,.0f}"),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Coefficient: %{x:+,.2f}<extra></extra>",
            ))
            _apply_defaults(feat_fig,
                title="Feature Coefficients (Scaled)",
                xaxis_title="Coefficient Value",
                yaxis=dict(autorange="reversed"),
                height=350, margin=dict(t=50, b=30, r=80),
            )
            st.plotly_chart(feat_fig, use_container_width=True, config=PLOTLY_CONFIG)

        with col_f2:
            # Polar / Radar chart of absolute importance
            radar_fig = go.Figure(go.Scatterpolar(
                r=coef_df["Abs_Coef"].tolist() + [coef_df["Abs_Coef"].iloc[0]],
                theta=coef_df["Feature"].tolist() + [coef_df["Feature"].iloc[0]],
                fill="toself",
                fillcolor="rgba(124,58,237,0.13)",
                line=dict(color=ACCENT_LIGHT, width=2),
                marker=dict(size=8, color=ACCENT_LIGHT),
                hovertemplate="<b>%{theta}</b><br>|Coefficient|: %{r:,.0f}<extra></extra>",
            ))
            _apply_defaults(radar_fig,
                title="Feature Importance Radar",
                height=350, margin=dict(t=60, b=30),
                polar=dict(
                    bgcolor=CARD2_BG,
                    radialaxis=dict(showticklabels=True, gridcolor="rgba(45,55,72,0.33)",
                                     tickfont=dict(size=9, color="#64748B")),
                    angularaxis=dict(gridcolor="rgba(45,55,72,0.33)",
                                      tickfont=dict(size=11, color="#94A3B8")),
                ),
            )
            st.plotly_chart(radar_fig, use_container_width=True, config=PLOTLY_CONFIG)

    # ─── Tab 4: Error Deep Dive ──────────────────────────────────────────
    with perf_tabs[3]:
        st.markdown("#### 🔬 Error by Feature Segment")
        segment_col = st.selectbox("Segment by", ["smoker", "sex", "region", "children"], key="err_seg")

        # Merge test data with errors
        test_df = bundle["X_test"].copy()
        # Decode back to original labels
        test_df_orig = df.iloc[test_df.index].copy()
        test_df_orig["abs_error"] = abs_errors
        test_df_orig["residual"] = residuals
        test_df_orig["predicted"] = y_pred

        # Box plot of errors by segment
        err_box = px.box(
            test_df_orig, x=segment_col, y="abs_error",
            color=segment_col,
            color_discrete_sequence=PALETTE,
            points="outliers",
            hover_data=["age", "bmi", "charges", "predicted"],
        )
        _apply_defaults(err_box,
            title=f"Absolute Error Distribution by {segment_col.title()}",
            xaxis_title=segment_col.title(),
            yaxis_title="Absolute Error ($)",
            height=400, margin=dict(t=60, b=40),
            yaxis=dict(tickprefix="$", tickformat=",.0s"),
            showlegend=False,
        )
        st.plotly_chart(err_box, use_container_width=True, config=PLOTLY_CONFIG)

        # Error stats table
        err_stats = test_df_orig.groupby(segment_col)["abs_error"].agg(
            ["mean", "median", "std", "min", "max", "count"]
        ).round(0)
        err_stats.columns = ["Mean Error", "Median Error", "Std Dev", "Min Error", "Max Error", "Count"]
        st.dataframe(err_stats.style.format("${:,.0f}", subset=["Mean Error", "Median Error", "Std Dev", "Min Error", "Max Error"]),
                     use_container_width=True)

    # ─── Tab 5: Sensitivity Analysis ─────────────────────────────────────
    with perf_tabs[4]:
        st.markdown("#### 🎚️ Interactive Sensitivity Analysis")
        st.markdown("<p style='color:#94A3B8;'>See how changing a single feature affects the predicted charge, while holding others constant.</p>",
                    unsafe_allow_html=True)

        sc1, sc2 = st.columns([1, 1])
        with sc1:
            vary_feature = st.selectbox("Feature to vary",
                                         ["age", "bmi", "children"], key="sens_feat")
        with sc2:
            base_smoker = st.selectbox("Base smoker status", ["no", "yes"], key="sens_smoker")

        # Build base input
        base_inputs = {"age": 35, "sex": "male", "bmi": 28.0,
                       "children": 1, "smoker": base_smoker, "region": "northeast"}

        if vary_feature == "age":
            values = list(range(18, 65))
        elif vary_feature == "bmi":
            values = [round(v, 1) for v in np.arange(15, 55, 0.5)]
        else:  # children
            values = list(range(0, 6))

        # Predict for each value
        predictions = []
        for v in values:
            inp = base_inputs.copy()
            inp[vary_feature] = v
            pred_val, _ = predict_charges(bundle, inp)
            predictions.append(pred_val)

        sens_fig = go.Figure()
        sens_fig.add_trace(go.Scatter(
            x=values, y=predictions,
            mode="lines+markers" if vary_feature == "children" else "lines",
            fill="tozeroy",
            fillcolor="rgba(124,58,237,0.06)",
            line=dict(color=ACCENT_LIGHT, width=2.5),
            marker=dict(size=8, color=ACCENT_LIGHT) if vary_feature == "children" else dict(size=0),
            hovertemplate=(
                f"<b>{vary_feature.title()}: %{{x}}</b><br>"
                "Predicted: $%{y:,.0f}<br>"
                "<extra></extra>"
            ),
            name=f"Smoker: {base_smoker}",
        ))

        # Add the other smoker scenario for comparison
        other_smoker = "yes" if base_smoker == "no" else "no"
        other_preds = []
        for v in values:
            inp = base_inputs.copy()
            inp[vary_feature] = v
            inp["smoker"] = other_smoker
            pred_val, _ = predict_charges(bundle, inp)
            other_preds.append(pred_val)

        sens_fig.add_trace(go.Scatter(
            x=values, y=other_preds,
            mode="lines+markers" if vary_feature == "children" else "lines",
            line=dict(color="#475569", width=1.5, dash="dot"),
            marker=dict(size=6, color="#475569") if vary_feature == "children" else dict(size=0),
            hovertemplate=(
                f"<b>{vary_feature.title()}: %{{x}}</b><br>"
                "Predicted: $%{y:,.0f}<br>"
                "<extra></extra>"
            ),
            name=f"Smoker: {other_smoker}",
        ))

        _apply_defaults(sens_fig,
            title=f"Sensitivity: Predicted Charges vs {vary_feature.title()}",
            xaxis_title=vary_feature.title(),
            yaxis_title="Predicted Charge ($)",
            height=400, margin=dict(t=60, b=40),
            yaxis=dict(tickprefix="$", tickformat=",.0s",
                       showgrid=True, gridcolor="rgba(45,55,72,0.2)"),
            xaxis=dict(showgrid=True, gridcolor="rgba(45,55,72,0.2)"),
        )
        st.plotly_chart(sens_fig, use_container_width=True, config=PLOTLY_CONFIG)

        st.caption(f"Base profile: Age={base_inputs['age']}, BMI={base_inputs['bmi']}, "
                   f"Children={base_inputs['children']}, Sex={base_inputs['sex']}, "
                   f"Region={base_inputs['region']}. Varying **{vary_feature}** only.")


# ---------------------------------------------------------------------------
# PAGE 4 — ABOUT
# ---------------------------------------------------------------------------
def page_about() -> None:
    """About page with problem statement, model card, tech stack, and author info."""
    st.markdown("## ℹ️ About InsureIQ")

    # Problem statement
    st.markdown('<div class="card card-accent">', unsafe_allow_html=True)
    st.markdown("""
    ### 🎯 Problem Statement
    Health insurance pricing is complex and opaque. Insurers must balance actuarial risk with
    competitiveness, while policyholders are left guessing why premiums differ. **InsureIQ** applies
    machine learning to the classic *Medical Cost Personal Datasets* benchmark to demystify the key
    drivers of annual insurance charges — empowering both consumers and analysts to make data-driven decisions.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🃏 Model Card")
        st.markdown("""
| Property | Value |
|---|---|
| Algorithm | Linear Regression |
| Library | scikit-learn |
| Training date | June 2025 |
| Dataset rows | 1,338 |
| Features | age, sex, bmi, children, smoker, region |
| Target | charges (USD/year) |
| Test R² | 0.7833 |
| Test MAE | $4,187 |
| Test RMSE | $5,800 |
| Train/Test split | 80 / 20 |
| Preprocessing | StandardScaler + LabelEncoder |
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔬 Features Used")
        feats = {
            "age":      ("🎂", "Numerical", "18–64 years"),
            "sex":      ("🧬", "Categorical", "female / male"),
            "bmi":      ("⚖️",  "Numerical", "Body Mass Index"),
            "children": ("👶", "Ordinal", "0–5 dependents"),
            "smoker":   ("🚬", "Categorical", "yes / no"),
            "region":   ("🌎", "Categorical", "4 US regions"),
        }
        for feat, (icon, ftype, desc) in feats.items():
            st.markdown(f"**{icon} {feat.title()}** — {ftype} · *{desc}*")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Interactive feature distribution overview
    st.markdown("### 📊 Feature Overview")
    df = _load_data()
    overview_feat = st.selectbox("Explore feature distribution", df.columns.tolist(), key="about_feat")

    if df[overview_feat].dtype in [np.int64, np.float64]:
        overview_fig = make_subplots(rows=1, cols=2,
                                      subplot_titles=("Distribution", "Box Plot"),
                                      column_widths=[0.6, 0.4])
        overview_fig.add_trace(
            go.Histogram(x=df[overview_feat], nbinsx=30, marker_color=ACCENT, opacity=0.7,
                          name="Distribution",
                          hovertemplate="%{x}: %{y} records<extra></extra>"),
            row=1, col=1,
        )
        overview_fig.add_trace(
            go.Box(y=df[overview_feat], marker_color=ACCENT_LIGHT, name=overview_feat.title(),
                    boxpoints="outliers",
                    hovertemplate="%{y}<extra></extra>"),
            row=1, col=2,
        )
        _apply_defaults(overview_fig, height=320, margin=dict(t=50, b=30),
                        showlegend=False)
    else:
        counts = df[overview_feat].value_counts()
        overview_fig = make_subplots(rows=1, cols=2,
                                      subplot_titles=("Bar Chart", "Pie Chart"),
                                      specs=[[{"type": "xy"}, {"type": "domain"}]])
        overview_fig.add_trace(
            go.Bar(x=counts.index, y=counts.values, marker_color=PALETTE[:len(counts)],
                    hovertemplate="%{x}: %{y} records<extra></extra>", name="Count"),
            row=1, col=1,
        )
        overview_fig.add_trace(
            go.Pie(labels=counts.index, values=counts.values,
                    marker=dict(colors=PALETTE[:len(counts)]),
                    hole=0.4, textinfo="label+percent",
                    hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"),
            row=1, col=2,
        )
        _apply_defaults(overview_fig, height=320, margin=dict(t=50, b=30),
                        showlegend=False)

    st.plotly_chart(overview_fig, use_container_width=True, config=PLOTLY_CONFIG)

    st.markdown("---")

    # Tech stack
    st.markdown("### 🛠️ Tech Stack")
    techs = [
        ("Python 3.11", "#3776AB", "#E8F4FC"),
        ("Streamlit 1.35", "#FF4B4B", "#FFF0F0"),
        ("scikit-learn 1.4", "#F89939", "#FFF8F0"),
        ("Plotly 5.x", "#3D4DB7", "#F0F2FF"),
        ("Pandas 2.x", "#150458", "#F5F0FF"),
        ("NumPy 1.26", "#013243", "#F0FAFA"),
    ]
    badges_html = " ".join(
        f'<span class="tech-badge" style="color:{tc}; border-color:{tc}44; background:{bg}11;">{name}</span>'
        for name, tc, bg in techs
    )
    st.markdown(badges_html, unsafe_allow_html=True)

    st.markdown("---")

    # Author card
    st.markdown("### 👤 Author")
    st.markdown(f"""
    <div class="card" style="display:flex; align-items:center; gap:1.5rem; max-width:480px;">
        <div style="width:60px; height:60px; border-radius:50%;
             background:linear-gradient(135deg,{ACCENT},{ACCENT_LIGHT});
             display:flex; align-items:center; justify-content:center;
             font-size:1.6rem; flex-shrink:0;">🧑‍💻</div>
        <div>
            <div style="font-weight:700; font-size:1.05rem;">Om</div>
            <div style="color:#94A3B8; font-size:0.82rem;">ML Engineer · Data Scientist</div>
            <div style="margin-top:8px; display:flex; gap:8px; flex-wrap:wrap;">
                <a href="https://linkedin.com" target="_blank"
                   style="background:{ACCENT}22; color:{ACCENT_LIGHT}; border:1px solid {ACCENT}44;
                          border-radius:6px; padding:3px 12px; font-size:0.75rem; text-decoration:none;">
                    🔗 LinkedIn
                </a>
                <a href="https://github.com" target="_blank"
                   style="background:#ffffff11; color:#E2E8F0; border:1px solid #2D3748;
                          border-radius:6px; padding:3px 12px; font-size:0.75rem; text-decoration:none;">
                    🐙 GitHub
                </a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <p style="font-size:0.72rem; color:#475569; text-align:center;">
    © 2025 InsureIQ · MIT License · For educational and portfolio purposes only.
    Predictions are estimates and should not be used for actual insurance underwriting decisions.
    </p>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    """Application entry point."""
    inject_css()
    init_session()

    # Load model — show spinner only on first load
    with st.spinner("🔧 Loading model…"):
        try:
            bundle = load_model_bundle()
        except Exception as exc:
            st.error(f"""
            **Model loading failed:** {exc}

            **Remediation steps:**
            1. Make sure `insurance-checkpoint ml-01.csv` is in the same folder as `app.py`.
            2. Verify that `scikit-learn`, `pandas`, and `numpy` are installed.
            3. Run `pip install -r requirements.txt` to install all dependencies.
            """)
            st.stop()

    render_header()
    page = render_sidebar(bundle)

    if   page == "Predict":            page_predict(bundle)
    elif page == "Data Explorer":      page_data_explorer()
    elif page == "Model Performance":  page_model_performance(bundle)
    elif page == "About":              page_about()


if __name__ == "__main__":
    main()


# =============================================================================
# requirements.txt
# -----------------------------------------------------------------------------
# streamlit>=1.35.0
# scikit-learn>=1.4.0
# pandas>=2.0.0
# numpy>=1.26.0
# plotly>=5.20.0
# statsmodels>=0.14.0
# =============================================================================