import streamlit as st
import pandas as pd
import numpy as np
import joblib
import hopsworks
import os
import re
import sys
import types
import requests
import plotly.graph_objects as go
import shap
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta
from pathlib import Path


def load_env_file(env_path: Path):
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


for candidate in [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]:
    load_env_file(candidate)

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
except Exception:
    pass

st.set_page_config(
    page_title="10Pearl AQI Predictors — Sargodha",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# GLOBAL STYLING
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
    }

    /* Force dark background regardless of Streamlit's own theme / DOM version */
    [data-testid="stAppViewContainer"], [data-testid="stMain"], .main, .stApp {
        background: radial-gradient(circle at top left, #101827 0%, #05070d 60%) !important;
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
    }
    body, p, span, div, label, h1, h2, h3, h4, h5, h6 {
        color: #e6edf3;
    }

    /* Hero header */
    .hero-box {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 28px 32px;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: 34px;
        font-weight: 800;
        color: #ffffff !important;
        margin: 0;
    }
    .brand-tag {
        display: inline-block;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        color: #7fd8ff !important;
        background: rgba(127,216,255,0.12);
        border: 1px solid rgba(127,216,255,0.35);
        padding: 4px 12px;
        border-radius: 999px;
        margin-bottom: 10px;
    }
    .hero-sub {
        font-size: 15px;
        color: #c9d6df !important;
        margin-top: 6px;
    }

    /* Metric cards — solid background so text is always visible, independent of theme */
    .metric-card {
        background: #131c2e;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255,255,255,0.3);
    }
    .metric-label {
        font-size: 13px;
        color: #9fb0c3 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff !important;
    }

    /* Model performance cards */
    .perf-card {
        background: #131c2e;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 20px;
        transition: transform 0.2s ease;
    }
    .perf-card:hover {
        transform: translateY(-4px);
        border-color: rgba(127,216,255,0.4);
    }
    .perf-day-title {
        font-size: 15px;
        font-weight: 700;
        color: #7fd8ff !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 14px;
        text-align: center;
    }
    .perf-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 4px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    .perf-row:last-child {
        border-bottom: none;
    }
    .perf-metric-name {
        font-size: 13px;
        color: #9fb0c3 !important;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .perf-metric-value {
        font-size: 17px;
        color: #ffffff !important;
        font-weight: 700;
    }
    .perf-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-top: 10px;
        text-align: center;
        width: 100%;
    }
    .insight-box {
        background: #101a2c;
        border-left: 4px solid #7fd8ff;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 14px 0;
        font-size: 14px;
        line-height: 1.7;
        color: #cfe0ee !important;
    }
    .insight-box b {
        color: #ffffff !important;
    }

    /* st.expander uses its own light-themed shadow DOM header/container that our
       page CSS can't reach otherwise — force it dark to match the rest of the UI. */
    [data-testid="stExpander"] {
        background-color: #10192b !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 14px !important;
        overflow: hidden;
    }
    [data-testid="stExpander"] summary {
        background-color: #10192b !important;
        color: #e6edf3 !important;
    }
    [data-testid="stExpander"] summary:hover {
        background-color: #16233a !important;
    }
    [data-testid="stExpander"] svg {
        fill: #7fd8ff !important;
    }
    [data-testid="stExpanderDetails"] {
        background-color: #0c1524 !important;
    }

    /* Custom dark table (used instead of st.dataframe, which renders its own
       light-themed grid that page CSS cannot restyle). */
    .dark-table-wrap {
        overflow-x: auto;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    table.dark-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
    }
    table.dark-table thead th {
        background: #16233a;
        color: #9fb0c3 !important;
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 0.6px;
        font-weight: 700;
        padding: 12px 16px;
        text-align: left;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    table.dark-table tbody td {
        padding: 12px 16px;
        color: #e6edf3 !important;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }
    table.dark-table tbody tr:last-child td {
        border-bottom: none;
    }
    table.dark-table tbody tr:nth-child(odd) {
        background: rgba(255,255,255,0.02);
    }
    table.dark-table tbody tr:hover {
        background: rgba(127,216,255,0.06);
    }
    table.dark-table td.num {
        text-align: right;
        font-variant-numeric: tabular-nums;
    }
    .quality-chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        white-space: nowrap;
    }

    /* Big AQI display */
    .aqi-hero {
        border-radius: 24px;
        padding: 34px;
        text-align: center;
        margin-bottom: 10px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
        border: 1px solid rgba(255,255,255,0.12);
        background-color: #10192b;
    }
    .aqi-hero-number {
        font-size: 96px;
        font-weight: 800;
        line-height: 1;
        text-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .aqi-hero-label {
        font-size: 22px;
        font-weight: 600;
        margin-top: 6px;
        letter-spacing: 0.5px;
    }
    .aqi-hero-date {
        font-size: 14px;
        color: rgba(255,255,255,0.85) !important;
        margin-top: 4px;
    }

    .day-tag {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        background: rgba(255,255,255,0.15);
        color: #fff !important;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .hazard-banner {
        padding: 20px 26px;
        border-radius: 16px;
        font-size: 17px;
        font-weight: 600;
        color: white !important;
        margin: 18px 0;
        animation: pulseGlow 1.8s infinite;
        border: 1px solid rgba(255,255,255,0.2);
    }
    @keyframes pulseGlow {
        0%   { box-shadow: 0 0 0px rgba(255,0,0,0.4); }
        50%  { box-shadow: 0 0 28px rgba(255,0,0,0.55); }
        100% { box-shadow: 0 0 0px rgba(255,0,0,0.4); }
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220 0%, #05070d 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] * {
        color: #e6edf3 !important;
    }

    div.stButton > button {
        border-radius: 12px;
        font-weight: 600;
        padding: 10px 18px;
        border: none;
        background: linear-gradient(135deg, #2c5364, #0f2027);
        color: white !important;
        transition: 0.2s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(44,83,100,0.5);
    }

    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: #2c5364; border-radius: 10px; }

    /* Inline markdown code spans (e.g. the sidebar metric values in `backticks`)
       default to a light background in Streamlit — force them dark so they match
       the rest of the theme instead of showing as white pills. */
    code {
        background-color: rgba(127,216,255,0.12) !important;
        color: #7fd8ff !important;
        padding: 2px 7px !important;
        border-radius: 6px !important;
        font-size: 0.9em !important;
        border: 1px solid rgba(127,216,255,0.25);
    }
    section[data-testid="stSidebar"] code {
        background-color: rgba(127,216,255,0.15) !important;
        color: #a8e6ff !important;
    }
</style>
""", unsafe_allow_html=True)


def create_requests_session(retries=3, backoff_factor=1.0):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# Compatibility shim for old scikit-learn pickles that reference the legacy `_loss` module.
def ensure_sklearn_loss_compat():
    try:
        import sklearn._loss._loss as sklearn_loss
        if "_loss" not in sys.modules:
            _loss_module = types.ModuleType("_loss")
            for attr in dir(sklearn_loss):
                if not attr.startswith("_"):
                    setattr(_loss_module, attr, getattr(sklearn_loss, attr))
            sys.modules["_loss"] = _loss_module
    except Exception:
        pass

ensure_sklearn_loss_compat()

# Compatibility shim for old numpy RandomState pickles that were saved with a numpy
# build which pickles the BitGenerator as a *class object*
# (e.g. `<class 'numpy.random._mt19937.MT19937'>`) instead of its string name.
# Older/mismatched numpy installs then fail with:
#   "ValueError: <class 'numpy.random._mt19937.MT19937'> is not a known BitGenerator module."
# This patches the constructor so it also accepts a class (or its str repr) and
# resolves it back to the plain string name numpy expects.
def ensure_numpy_bitgenerator_compat():
    try:
        import numpy.random._pickle as np_random_pickle
        from numpy.random.bit_generator import BitGenerator

        original_ctor = np_random_pickle.__dict__.get("__bit_generator_ctor")
        if original_ctor is None:
            return

        def patched_bit_generator_ctor(bit_generator_name="MT19937"):
            # Some numpy versions pickle RandomState by embedding the *live*,
            # already-constructed BitGenerator instance rather than its name/class
            # (e.g. "<numpy.random._mt19937.MT19937 object at 0x...>"). In that
            # case there's nothing to construct — just hand it straight back.
            if isinstance(bit_generator_name, BitGenerator):
                return bit_generator_name
            if isinstance(bit_generator_name, type):
                bit_generator_name = bit_generator_name.__name__
            elif not isinstance(bit_generator_name, str):
                text = str(bit_generator_name)
                match = re.search(r"\.([A-Za-z0-9_]+)(?:\s+object)?[\s'>]*$", text)
                bit_generator_name = match.group(1) if match else text
            return original_ctor(bit_generator_name)

        np_random_pickle.__dict__["__bit_generator_ctor"] = patched_bit_generator_ctor

        # CRITICAL: `__randomstate_ctor` (and `__generator_ctor`, where present)
        # capture `bit_generator_ctor=__bit_generator_ctor` as a *default parameter
        # value* at the moment numpy itself was first imported — long before this
        # shim runs. Reassigning the module-level name above does NOT change that
        # already-bound default, so pickle calls made through __randomstate_ctor
        # would still silently use the ORIGINAL, unpatched constructor. We fix that
        # by rewriting the bound default itself.
        def patch_ctor_default(func_name):
            func = np_random_pickle.__dict__.get(func_name)
            if func is None or getattr(func, "__defaults__", None) is None:
                return
            new_defaults = tuple(
                patched_bit_generator_ctor if (callable(d) and not isinstance(d, str)) else d
                for d in func.__defaults__
            )
            func.__defaults__ = new_defaults

        patch_ctor_default("__randomstate_ctor")
        patch_ctor_default("__generator_ctor")
    except Exception:
        pass

ensure_numpy_bitgenerator_compat()

# Compatibility shim for a deeper numpy RNG-state pickle mismatch that can surface
# even after the BitGenerator-ctor shim above, e.g.:
#   "ValueError: state is not a legacy MT19937 state"
# This happens because the fitted sklearn model has an internal RandomState/BitGenerator
# object embedded in it (used only during .fit(), never during .predict()), and the
# exact state format written by the training environment's numpy isn't accepted by the
# numpy installed here. Since that RNG state has no effect on prediction output, we make
# joblib's unpickler simply skip restoring it instead of aborting the whole load.
def ensure_joblib_rng_state_compat():
    try:
        import pickle as _pickle_mod
        import joblib.numpy_pickle as _jl_numpy_pickle

        original_load_build = _jl_numpy_pickle.NumpyUnpickler.__dict__.get("load_build")
        if original_load_build is None:
            return

        def patched_load_build(self):
            try:
                original_load_build(self)
            except ValueError as exc:
                msg = str(exc).lower()
                if "legacy" in msg or "bitgenerator" in msg or "bit generator" in msg:
                    # Incompatible/irrelevant RNG state — safe to ignore for inference.
                    pass
                else:
                    raise

        # Reassigning the attribute alone is not enough: joblib registers the
        # *original* function object directly inside its own `dispatch` dict
        # (a class-level copy of pickle.Unpickler.dispatch) at class-definition
        # time, and pickle's main loop calls dispatch[BUILD] rather than looking
        # up self.load_build dynamically. Both must be updated.
        _jl_numpy_pickle.NumpyUnpickler.load_build = patched_load_build
        _jl_numpy_pickle.NumpyUnpickler.dispatch[_pickle_mod.BUILD[0]] = patched_load_build
    except Exception:
        pass

ensure_joblib_rng_state_compat()

# ============================================================
# MODEL PERFORMANCE METRICS (offline evaluation on held-out test data)
# ============================================================
MODEL_PERFORMANCE = {
    "Day 1": {"mae": 9.673568713851495, "r2": 0.806151324143798, "rmse": 12.150241126790617},
    "Day 2": {"mae": 17.890412262642194, "r2": 0.7193661068277716, "rmse": 23.084172918759066},
    "Day 3": {"mae": 18.17306000375878, "r2": 0.7103920160949766, "rmse": 23.237578758354},
}


def render_dark_table(headers, rows, numeric_cols=None):
    """Render an HTML table styled to match the dark dashboard theme, since
    st.dataframe renders its own light-themed grid that page CSS can't reach.

    headers: list[str]
    rows: list[list[str]] — cell content, already formatted as HTML-safe strings
    numeric_cols: set of column indices to right-align
    """
    numeric_cols = numeric_cols or set()
    header_html = "".join(f"<th>{h}</th>" for h in headers)
    body_html = ""
    for row in rows:
        cells = "".join(
            f'<td class="{"num" if i in numeric_cols else ""}">{val}</td>'
            for i, val in enumerate(row)
        )
        body_html += f"<tr>{cells}</tr>"

    st.markdown(f"""
    <div class="dark-table-wrap">
        <table class="dark-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{body_html}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


def get_performance_badge(r2):
    """Map an R² score to a plain-language quality label + color."""
    if r2 >= 0.85:
        return "🏆 EXCELLENT FIT", "#2ecc71"
    elif r2 >= 0.70:
        return "✅ GOOD FIT", "#5dade2"
    elif r2 >= 0.50:
        return "🟡 FAIR FIT", "#f1c40f"
    else:
        return "🟠 NEEDS IMPROVEMENT", "#e67e22"


def render_model_performance():
    """Render a full model-evaluation section: score cards, comparison charts,
    a plain-language insight box, and a detailed metrics table."""
    st.markdown("### 📊 Model Performance & Evaluation")
    st.caption("Metrics computed on held-out historical data (not seen during training). Lower MAE & RMSE = more accurate. Higher R² = better fit.")

    days = list(MODEL_PERFORMANCE.keys())
    day_colors = {"Day 1": "#2ecc71", "Day 2": "#f1c40f", "Day 3": "#e67e22"}

    # ---------------- SCORE CARDS WITH QUALITY BADGES ----------------
    cols = st.columns(3)
    for col, day in zip(cols, days):
        metrics = MODEL_PERFORMANCE[day]
        color = day_colors.get(day, "#7fd8ff")
        badge_label, badge_color = get_performance_badge(metrics["r2"])
        with col:
            st.markdown(f"""
            <div class="perf-card" style="border-top: 3px solid {color};">
                <div class="perf-day-title">{day} Model</div>
                <div class="perf-row">
                    <span class="perf-metric-name">MAE</span>
                    <span class="perf-metric-value">{metrics['mae']:.2f}</span>
                </div>
                <div class="perf-row">
                    <span class="perf-metric-name">RMSE</span>
                    <span class="perf-metric-value">{metrics['rmse']:.2f}</span>
                </div>
                <div class="perf-row">
                    <span class="perf-metric-name">R² Score</span>
                    <span class="perf-metric-value">{metrics['r2']:.3f}</span>
                </div>
                <div class="perf-badge" style="background:{badge_color}26; color:{badge_color} !important; border:1px solid {badge_color}66;">
                    {badge_label}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # ---------------- COMPARISON CHARTS ----------------
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        st.markdown("#### 📉 Error Comparison — MAE vs RMSE")
        mae_vals = [MODEL_PERFORMANCE[d]["mae"] for d in days]
        rmse_vals = [MODEL_PERFORMANCE[d]["rmse"] for d in days]

        err_fig = go.Figure()
        err_fig.add_trace(go.Bar(
            x=days, y=mae_vals, name="MAE",
            marker=dict(color="#5dade2", line=dict(color="rgba(255,255,255,0.15)", width=1)),
            text=[f"{v:.2f}" for v in mae_vals], textposition="outside",
            textfont=dict(color="#dfe9f3"),
        ))
        err_fig.add_trace(go.Bar(
            x=days, y=rmse_vals, name="RMSE",
            marker=dict(color="#ff6b6b", line=dict(color="rgba(255,255,255,0.15)", width=1)),
            text=[f"{v:.2f}" for v in rmse_vals], textposition="outside",
            textfont=dict(color="#dfe9f3"),
        ))
        err_fig.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            font=dict(color="#dfe9f3", family="Poppins, sans-serif"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=30, b=10),
            height=320,
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="AQI points", gridcolor="rgba(255,255,255,0.06)"),
        )
        st.plotly_chart(err_fig, use_container_width=True, key="perf_error_chart")

    with chart_col2:
        st.markdown("#### 🎯 R² — Model Fit by Horizon")
        r2_vals = [MODEL_PERFORMANCE[d]["r2"] for d in days]
        bar_colors = [day_colors.get(d, "#7fd8ff") for d in days]

        r2_fig = go.Figure(go.Bar(
            x=days, y=r2_vals,
            marker=dict(color=bar_colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
            text=[f"{v:.3f}" for v in r2_vals], textposition="outside",
            textfont=dict(color="#dfe9f3"),
        ))
        r2_fig.add_hline(y=0.7, line_dash="dot", line_color="#7fd8ff", opacity=0.6,
                          annotation_text="Good-fit threshold", annotation_font_size=10,
                          annotation_font_color="#7fd8ff")
        r2_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            font=dict(color="#dfe9f3", family="Poppins, sans-serif"),
            margin=dict(l=10, r=10, t=30, b=10),
            height=320,
            showlegend=False,
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="R² Score", range=[0, 1], gridcolor="rgba(255,255,255,0.06)"),
        )
        st.plotly_chart(r2_fig, use_container_width=True, key="perf_r2_chart")

    # ---------------- PLAIN-LANGUAGE INSIGHT BOX ----------------
    best_day = max(days, key=lambda d: MODEL_PERFORMANCE[d]["r2"])
    worst_day = min(days, key=lambda d: MODEL_PERFORMANCE[d]["r2"])
    accuracy_drop = MODEL_PERFORMANCE["Day 1"]["r2"] - MODEL_PERFORMANCE["Day 3"]["r2"]

    st.markdown(f"""
    <div class="insight-box">
    💡 <b>What this means:</b><br>
    • The <b>{best_day}</b> model is the most reliable, explaining
    <b>{MODEL_PERFORMANCE[best_day]['r2']*100:.1f}%</b> of the variance in actual AQI
    (R² = {MODEL_PERFORMANCE[best_day]['r2']:.3f}), with an average error of only
    <b>±{MODEL_PERFORMANCE[best_day]['mae']:.1f} AQI points</b>.<br>
    • Accuracy naturally decreases with forecast horizon — by <b>{worst_day}</b>, R² drops to
    <b>{MODEL_PERFORMANCE[worst_day]['r2']:.3f}</b> (a {accuracy_drop*100:.1f} percentage-point decline from Day 1),
    which is expected since weather and pollution patterns become harder to predict further into the future.<br>
    • RMSE is consistently higher than MAE for every horizon, which means occasional larger errors
    (e.g. sudden pollution spikes or dust events) pull the average up more than typical day-to-day error.
    </div>
    """, unsafe_allow_html=True)

    # ---------------- DETAILED METRICS TABLE ----------------
    with st.expander("📋 View Detailed Metrics Table"):
        headers = ["Model", "MAE", "RMSE", "R² Score", "Quality"]
        rows = []
        for day in days:
            m = MODEL_PERFORMANCE[day]
            badge_label, badge_color = get_performance_badge(m["r2"])
            quality_html = (
                f'<span class="quality-chip" style="background:{badge_color}26; '
                f'color:{badge_color}; border:1px solid {badge_color}66;">{badge_label}</span>'
            )
            rows.append([
                f"<b>{day}</b>",
                f"{m['mae']:.3f}",
                f"{m['rmse']:.3f}",
                f"{m['r2']:.4f}",
                quality_html,
            ])
        render_dark_table(headers, rows, numeric_cols={1, 2, 3})

    st.markdown("")

# ============================================================
# AQI CATEGORY HELPERS (US AQI breakpoints)
# ============================================================
def get_aqi_category(aqi):
    """Return (label, color_hex, emoji, advisory) for a given US AQI value."""
    if aqi <= 50:
        return "Good", "#2ecc71", "🟢", "Air quality is satisfactory. Enjoy outdoor activities."
    elif aqi <= 100:
        return "Moderate", "#f1c40f", "🟡", "Acceptable, but sensitive groups should watch symptoms."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#e67e22", "🟠", "Children, elderly & respiratory patients should limit outdoor exertion."
    elif aqi <= 200:
        return "Unhealthy", "#e74c3c", "🔴", "Everyone may experience health effects. Reduce prolonged outdoor exertion."
    elif aqi <= 300:
        return "Very Unhealthy", "#8e44ad", "🟣", "Health alert! Avoid outdoor activity, wear a mask if going outside."
    else:
        return "Hazardous", "#7d1414", "🟤", "Emergency conditions! Stay indoors, use air purifiers, avoid all outdoor exposure."


def aqi_gauge_chart(value, title):
    label, color, emoji, _ = get_aqi_category(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'font': {'size': 40, 'color': color}},
        title={'text': f"{emoji} {title}", 'font': {'size': 16, 'color': '#dfe9f3'}},
        gauge={
            'axis': {'range': [0, 500], 'tickcolor': '#8fa3b3', 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.35},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(46,204,113,0.25)'},
                {'range': [50, 100], 'color': 'rgba(241,196,15,0.25)'},
                {'range': [100, 150], 'color': 'rgba(230,126,34,0.25)'},
                {'range': [150, 200], 'color': 'rgba(231,76,60,0.25)'},
                {'range': [200, 300], 'color': 'rgba(142,68,173,0.25)'},
                {'range': [300, 500], 'color': 'rgba(125,20,20,0.25)'},
            ],
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#dfe9f3"},
        height=260,
        margin=dict(l=20, r=20, t=50, b=10),
    )
    return fig


def render_shap_explainability(model, feature_names, historical, forecast_daily, day_label):
    """Render a compact SHAP explanation for the current AQI forecast — styled to match the dark dashboard theme."""
    try:
        x_vec = build_feature_vector(historical, forecast_daily, feature_names)

        # GradientBoostingRegressor is tree-based, so TreeExplainer computes exact
        # Shapley values from the tree structure itself — no background sample needed.
        # (Using a background made of copies of the same instance, as before, made every
        # perturbation a no-op and forced all SHAP values to 0.)
        try:
            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            explanation = explainer(x_vec)
        except Exception:
            # Fallback for non-tree models: use real historical rows as background,
            # never copies of the instance being explained.
            hist_feature_rows = []
            for i in range(1, min(21, len(historical) - 5)):
                try:
                    as_of = historical.index.max().date() - timedelta(days=i)
                    hist_slice = historical.loc[historical.index.date <= as_of]
                    if len(hist_slice) < 35:
                        continue
                    hist_feature_rows.append(
                        build_feature_vector(hist_slice, forecast_daily, feature_names)[0]
                    )
                except Exception:
                    continue
            background = np.array(hist_feature_rows) if hist_feature_rows else np.tile(x_vec[0], (5, 1))
            explainer = shap.Explainer(model, background)
            explanation = explainer(x_vec)

        values = np.asarray(explanation.values)
        if values.ndim == 3:
            values = values[0]
        elif values.ndim == 2 and values.shape[0] == 1:
            values = values[0]
        elif values.ndim == 0:
            values = np.asarray([0.0] * len(feature_names))

        if values.size != len(feature_names):
            values = np.asarray(values).reshape(-1)
            if values.size < len(feature_names):
                pad = np.zeros(len(feature_names) - values.size)
                values = np.concatenate([values, pad])

        raw_vals = np.asarray(values, dtype=float)
        abs_vals = np.abs(raw_vals)
        if abs_vals.sum() == 0:
            abs_vals = np.ones_like(abs_vals)

        contrib_pct = pd.Series((abs_vals / abs_vals.sum()) * 100, index=feature_names)
        signed = pd.Series(raw_vals, index=feature_names)

        top = contrib_pct.sort_values(ascending=False).head(8)
        top_signed = signed.loc[top.index]

        # Sort ascending so the biggest contributor lands at the top of the horizontal chart
        order = top.sort_values(ascending=True).index
        top = top.loc[order]
        top_signed = top_signed.loc[order]

        bar_colors = ["#ff6b6b" if v > 0 else "#5dade2" for v in top_signed.values]

        st.markdown(f"### 🔍 SHAP Explainability — {day_label}")
        st.caption("🔴 Red bars push AQI upward · 🔵 Blue bars pull AQI downward. Bar length = relative importance.")

        fig = go.Figure(go.Bar(
            x=top.values,
            y=top.index,
            orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(color="rgba(255,255,255,0.15)", width=1),
            ),
            text=[f"{v:.1f}%" for v in top.values],
            textposition="outside",
            textfont=dict(color="#e6edf3", size=12),
            hovertemplate="<b>%{y}</b><br>Contribution: %{x:.2f}%<extra></extra>",
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            font=dict(color="#dfe9f3", family="Poppins, sans-serif"),
            margin=dict(l=10, r=40, t=20, b=10),
            height=320,
            xaxis=dict(
                title="Relative contribution (%)",
                gridcolor="rgba(255,255,255,0.06)",
                zerolinecolor="rgba(255,255,255,0.15)",
            ),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.03)",
                automargin=True,
            ),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"shap_{day_label}")

    except Exception as exc:
        st.info(f"SHAP explanation for {day_label} is unavailable right now: {exc}")


# ============================================================
# HERO HEADER
# ============================================================
st.markdown(f"""
<div class="hero-box">
    <p class="brand-tag">10PEARL AQI PREDICTORS</p>
    <p class="hero-title">🌬️ Sargodha AQI — 3-Day Live Forecast System</p>
    <p class="hero-sub">📍 Sargodha, Punjab, Pakistan &nbsp;|&nbsp; 🗓️ {datetime.now().strftime('%A, %d %B %Y')} &nbsp;|&nbsp; Powered by Hopsworks + Open-Meteo</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# HOPSWORKS API KEY — read from env / .env / Streamlit secrets
# and allow manual entry in the sidebar for deployment use.
# ============================================================
def resolve_hopsworks_api_key():
    # 1) environment or .env file
    env_key = os.environ.get("HOPSWORKS_API_KEY")
    if env_key and str(env_key).strip():
        return str(env_key).strip()

    # 2) Streamlit Community Cloud secret
    try:
        secret_key = st.secrets.get("HOPSWORKS_API_KEY")
        if secret_key and str(secret_key).strip():
            return str(secret_key).strip()
    except Exception:
        pass

    # 3) manual input in sidebar
    sidebar_key = st.sidebar.text_input(
        "Hopsworks API key",
        type="password",
        help="Paste Hopsworks key here if it is not loaded from the environment or Streamlit secrets.",
        key="hopsworks_api_key_input",
    )
    if sidebar_key and str(sidebar_key).strip():
        return str(sidebar_key).strip()

    return None


st.sidebar.markdown("## 🔑 Hopsworks Connection")
api_key = resolve_hopsworks_api_key()
if api_key:
    st.sidebar.success("✅ Hopsworks API key loaded.")
else:
    st.sidebar.warning("⚠️ HOPSWORKS_API_KEY not found. Set it in .env, Streamlit secrets, or paste it here.")
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info(
    "This dashboard pulls live weather & air-quality data from Open-Meteo, "
    "assembles model features, and runs 3 pre-trained Gradient Boosting "
    "models (Day 1 / Day 2 / Day 3) hosted on Hopsworks to forecast AQI."
)

st.sidebar.markdown("### 📊 Model Metrics (Test Set)")
for _day, _m in MODEL_PERFORMANCE.items():
    st.sidebar.markdown(
        f"**{_day}** — MAE `{_m['mae']:.2f}` · RMSE `{_m['rmse']:.2f}` · R² `{_m['r2']:.3f}`"
    )

# Sargodha Coordinates
LATITUDE = 32.0836
LONGITUDE = 72.6711

# Helper function to extract predict-capable model object
def extract_model(obj):
    if hasattr(obj, "predict"):
        return obj
    elif isinstance(obj, (list, tuple)) and len(obj) > 0:
        for item in obj:
            if hasattr(item, "predict"):
                return item
    elif isinstance(obj, dict):
        for key, val in obj.items():
            if hasattr(val, "predict"):
                return val
    return None

# --- 0. BUILD FEATURE VECTOR FOR EACH MODEL HORIZON ---
@st.cache_data(ttl=86400)
def fetch_historical_daily_data(lat, lon, lookback_days=45):
    end_date = datetime.utcnow().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=lookback_days)

    session = create_requests_session(retries=4, backoff_factor=1.0)
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "auto",
    }
    w_res = session.get(weather_url, params=weather_params, timeout=60)
    w_res.raise_for_status()
    weather_json = w_res.json()
    if "hourly" not in weather_json:
        raise ValueError(f"Weather history not available: {weather_json}")

    weather_df = pd.DataFrame(weather_json["hourly"])
    weather_df["time"] = pd.to_datetime(weather_df["time"])
    weather_df = weather_df.set_index("time").resample("D").mean()

    aqi_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aqi_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
        "timezone": "auto",
    }
    aqi_res = session.get(aqi_url, params=aqi_params, timeout=60)
    aqi_res.raise_for_status()
    aqi_json = aqi_res.json()
    if "hourly" not in aqi_json:
        raise ValueError(f"AQI history not available: {aqi_json}")

    aqi_df = pd.DataFrame(aqi_json["hourly"])
    aqi_df["time"] = pd.to_datetime(aqi_df["time"])
    aqi_df = aqi_df.set_index("time").resample("D").mean()
    aqi_df = aqi_df.rename(columns={"us_aqi": "AQI"})

    combined = pd.concat([weather_df, aqi_df[["AQI", "pm2_5"]]], axis=1)
    combined = combined[["AQI", "pm2_5", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"]]
    combined = combined.dropna()
    if len(combined) < 35:
        raise ValueError("Not enough historical daily data available for lag/rolling feature construction.")
    return combined

@st.cache_data(ttl=600)
def fetch_forecast_weather(lat, lon, horizon_days=5):
    session = create_requests_session(retries=4, backoff_factor=1.0)
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    forecast_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "forecast_days": horizon_days,
        "timezone": "auto",
    }
    forecast_res = session.get(forecast_url, params=forecast_params, timeout=60)
    forecast_res.raise_for_status()
    forecast_json = forecast_res.json()
    if "hourly" not in forecast_json:
        raise ValueError(f"Weather forecast not available: {forecast_json}")

    fc_df = pd.DataFrame(forecast_json["hourly"])
    fc_df["time"] = pd.to_datetime(fc_df["time"])
    fc_df = fc_df.set_index("time").resample("D").mean()
    return fc_df


def build_feature_vector(historical, forecast_daily, feature_names):
    latest_date = historical.index.max().date()
    latest_row = historical.loc[historical.index.date == latest_date]
    if latest_row.empty:
        raise ValueError("Latest historical date missing from assembled daily data.")
    latest_row = latest_row.iloc[0]

    def get_lag(name, lag):
        date = latest_date - timedelta(days=lag)
        if date not in historical.index.date:
            raise ValueError(f"Missing historical data for lag {lag} days ago ({date}).")
        return historical.loc[historical.index.date == date].iloc[0][name]

    def get_roll(stat, window):
        series = historical["AQI"].shift(1).rolling(window=window)
        value = getattr(series, stat)().iloc[-1]
        if pd.isna(value):
            raise ValueError(f"Missing rolling feature {stat} over {window} days.")
        return value

    def canonical_key(name):
        return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")

    feature_vector = []
    for feature in feature_names:
        key = canonical_key(feature)

        if match := re.match(r"^aqi_lag[_-]?([0-9]+)$", key):
            feature_vector.append(get_lag("AQI", int(match.group(1))))
        elif match := re.match(r"^(?:pm2_5|pm25|pm_25)_lag[_-]?([0-9]+)$", key):
            feature_vector.append(get_lag("pm2_5", int(match.group(1))))
        elif match := re.match(r"^(?:temp|temperature|temperature_2m)_lag[_-]?([0-9]+)$", key):
            feature_vector.append(get_lag("temperature_2m", int(match.group(1))))
        elif match := re.match(r"^aqi_rolling_([0-9]+)d_(mean|max|min|std)$", key):
            stat = match.group(2)
            window = int(match.group(1))
            feature_vector.append(get_roll(stat, window))
        elif match := re.match(r"^aqi_roll_(mean|max|min|std)_(\d+)$", key):
            stat = match.group(1)
            window = int(match.group(2))
            feature_vector.append(get_roll(stat, window))
        elif key in {"sin_day", "doy_sin", "month_sin"}:
            if key == "month_sin":
                value = np.sin(2 * np.pi * latest_date.month / 12)
            else:
                day_of_year = latest_date.timetuple().tm_yday
                value = np.sin(2 * np.pi * day_of_year / 365.25)
            feature_vector.append(value)
        elif key in {"cos_day", "doy_cos", "month_cos"}:
            if key == "month_cos":
                value = np.cos(2 * np.pi * latest_date.month / 12)
            else:
                day_of_year = latest_date.timetuple().tm_yday
                value = np.cos(2 * np.pi * day_of_year / 365.25)
            feature_vector.append(value)
        elif key in {"month", "month_number"}:
            feature_vector.append(latest_date.month)
        elif key in {"dayofweek", "day_of_week"}:
            feature_vector.append(latest_date.weekday())
        elif key in {"day_of_year", "doy"}:
            feature_vector.append(latest_date.timetuple().tm_yday)
        elif key in {"is_weekend"}:
            feature_vector.append(int(latest_date.weekday() >= 5))
        elif match := re.match(r"^(?:temp|temperature|temperature_2m)_future_h(\d+)$", key):
            horiz = int(match.group(1))
            target_date = latest_date + timedelta(days=horiz)
            if target_date not in forecast_daily.index.date:
                raise ValueError(f"Forecast weather not available for {target_date}.")
            weather_row = forecast_daily.loc[forecast_daily.index.date == target_date].iloc[0]
            feature_vector.append(weather_row["temperature_2m"])
        elif match := re.match(r"^(?:humidity|relative_humidity|relative_humidity_2m)_future_h(\d+)$", key):
            horiz = int(match.group(1))
            target_date = latest_date + timedelta(days=horiz)
            if target_date not in forecast_daily.index.date:
                raise ValueError(f"Forecast weather not available for {target_date}.")
            weather_row = forecast_daily.loc[forecast_daily.index.date == target_date].iloc[0]
            feature_vector.append(weather_row["relative_humidity_2m"])
        elif match := re.match(r"^(?:wind|wind_speed|wind_speed_10m)_future_h(\d+)$", key):
            horiz = int(match.group(1))
            target_date = latest_date + timedelta(days=horiz)
            if target_date not in forecast_daily.index.date:
                raise ValueError(f"Forecast weather not available for {target_date}.")
            weather_row = forecast_daily.loc[forecast_daily.index.date == target_date].iloc[0]
            feature_vector.append(weather_row["wind_speed_10m"])
        else:
            raise ValueError(f"Unknown feature name: {feature}")

    return np.array(feature_vector).reshape(1, -1)

# --- 1. OPEN-METEO API FETCHING WITH BETTER TIMEOUT & RETRY ---
@st.cache_data(ttl=600)
def fetch_live_weather_and_aqi():
    data = {"pm25": 45.0, "pm10": 90.0, "temp": 30.0, "humidity": 55.0, "wind": 10.0}

    # Weather Fetch
    session = create_requests_session(retries=3, backoff_factor=0.5)
    try:
        weather_url = f"https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        }
        w_res = session.get(weather_url, params=weather_params, timeout=30)
        w_res.raise_for_status()
        w_json = w_res.json()
        curr_w = w_json.get("current", {})
        data["temp"] = curr_w.get("temperature_2m", 30.0)
        data["humidity"] = curr_w.get("relative_humidity_2m", 55.0)
        data["wind"] = curr_w.get("wind_speed_10m", 10.0)
    except Exception as e:
        st.warning(f"Weather API Warning: {e}. Fallback values used.")

    # Air Quality Fetch
    try:
        aqi_url = f"https://air-quality-api.open-meteo.com/v1/air-quality"
        aqi_params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "current": "pm10,pm2_5",
        }
        a_res = session.get(aqi_url, params=aqi_params, timeout=30)
        a_res.raise_for_status()
        a_json = a_res.json()
        curr_a = a_json.get("current", {})
        data["pm25"] = curr_a.get("pm2_5", 45.0)
        data["pm10"] = curr_a.get("pm10", 90.0)
    except Exception as e:
        st.warning(f"Air Quality API Warning: {e}. Fallback values used.")

    return data

# --- 2. HOPSWORKS MODEL LOADING WITH SAFELY UNPACKED MODEL OBJECT ---
@st.cache_resource(show_spinner="Hopsworks Registry se Models download aur extract ho rahe hain...")
def load_all_models(key):
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        api_key_value=key,
        project="colab"
    )
    mr = project.get_model_registry()

    models_dict = {}
    model_info = [
        ("sargodha_aqi_gbr_day1", 4, "Day 1"),
        ("sargodha_aqi_gbr_day2", 3, "Day 2"),
        ("sargodha_aqi_gbr_day3", 3, "Day 3")
    ]

    for m_name, m_ver, label in model_info:
        model_meta = mr.get_model(m_name, version=m_ver)
        model_dir = model_meta.download()

        model_obj = None
        feature_names = None
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if file == "features.pkl":
                    feature_names = joblib.load(file_path)
                elif file.endswith(".pkl") or file.endswith(".joblib"):
                    if file == "features.pkl":
                        continue
                    raw_obj = joblib.load(file_path)
                    extracted = extract_model(raw_obj)
                    if extracted is not None:
                        model_obj = extracted
            if model_obj is not None and feature_names is not None:
                break

        if model_obj is None:
            raise ValueError(f"No model object found in downloaded artifact for {m_name}.")
        if feature_names is None:
            raise ValueError(f"No features.pkl found for {m_name}; cannot build the model input vector.")

        models_dict[label] = {
            "model": model_obj,
            "features": feature_names,
        }

    return models_dict

models = None
if api_key:
    try:
        models = load_all_models(api_key)
        st.sidebar.success("✅ Day 1, 2, 3 Models Ready!")
    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
        import traceback
        st.sidebar.code(traceback.format_exc())
else:
    st.info("👈 HOPSWORKS_API_KEY set nahi hai. Terminal me export karo ya project root me .env file create karo.")

# --- 3. LIVE DATA UI ---
st.markdown("### 📡 Live Weather & Air Quality — Sargodha")

col_refresh, _ = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 Refresh Live Data"):
        st.cache_data.clear()

live_data = fetch_live_weather_and_aqi()

m1, m2, m3, m4, m5 = st.columns(5)
metric_defs = [
    (m1, "PM2.5", f"{live_data['pm25']:.1f} µg/m³", "🫧"),
    (m2, "PM10", f"{live_data['pm10']:.1f} µg/m³", "🌫️"),
    (m3, "Temperature", f"{live_data['temp']:.1f} °C", "🌡️"),
    (m4, "Humidity", f"{live_data['humidity']:.0f} %", "💧"),
    (m5, "Wind Speed", f"{live_data['wind']:.1f} km/h", "🍃"),
]
for col, label, value, icon in metric_defs:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{icon} {label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# --- 4. PREDICTION & ALARMING ALERT SYSTEM ---
st.markdown("### 🚀 Run Forecast")
run_forecast = st.button("🚀 Run 3-Day AQI Forecast", type="primary")

if run_forecast:
    if not models or len(models) < 3:
        st.error("Models load nahi hue! HOPSWORKS_API_KEY environment variable verify karo.")
    else:
        try:
            with st.spinner("Fetching historical & forecast weather data, building features, running models..."):
                historical = fetch_historical_daily_data(LATITUDE, LONGITUDE)
                forecast_daily = fetch_forecast_weather(LATITUDE, LONGITUDE, horizon_days=5)

                preds = []
                for label in ["Day 1", "Day 2", "Day 3"]:
                    model_info = models[label]
                    model = model_info["model"]
                    feature_names = model_info["features"]
                    x_vec = build_feature_vector(historical, forecast_daily, feature_names)
                    preds.append(float(model.predict(x_vec)[0]))

            today = datetime.now()
            dates = [(today + timedelta(days=i + 1)) for i in range(3)]
            date_labels = [d.strftime("%d %b (%a)") for d in dates]
            preds = [round(v, 1) for v in preds]

            # ---------------- HAZARD BANNER ----------------
            max_aqi = max(preds)
            worst_day_idx = preds.index(max_aqi)
            worst_label, worst_color, worst_emoji, worst_advisory = get_aqi_category(max_aqi)

            if max_aqi > 200:
                st.markdown(f"""
                <div class="hazard-banner" style="background: linear-gradient(135deg, #7d1414, #b71c1c);">
                    🚨 SEVERE ALERT — {worst_label.upper()} AIR EXPECTED ON {date_labels[worst_day_idx]}!<br>
                    Peak Predicted AQI: <b>{max_aqi}</b> {worst_emoji} &nbsp;|&nbsp; {worst_advisory}
                </div>
                """, unsafe_allow_html=True)
            elif max_aqi > 150:
                st.markdown(f"""
                <div class="hazard-banner" style="background: linear-gradient(135deg, #e67e22, #d35400);">
                    ⚠️ UNHEALTHY AIR QUALITY WARNING — {date_labels[worst_day_idx]}<br>
                    Peak Predicted AQI: <b>{max_aqi}</b> {worst_emoji} &nbsp;|&nbsp; {worst_advisory}
                </div>
                """, unsafe_allow_html=True)
            elif max_aqi > 100:
                st.markdown(f"""
                <div class="hazard-banner" style="background: linear-gradient(135deg, #f39c12, #f1c40f); color:#3a2c00;">
                    🟡 MODERATE ALERT — Sensitive groups take care on {date_labels[worst_day_idx]}<br>
                    Peak Predicted AQI: <b>{max_aqi}</b> {worst_emoji}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success(f"✅ **ACCEPTABLE AIR QUALITY.** Maximum predicted AQI over next 3 days is **{max_aqi}** {worst_emoji}")

            # ---------------- BIG AQI HERO CARDS ----------------
            st.markdown("### 📅 3-Day AQI Forecast")
            hcols = st.columns(3)
            for i, col in enumerate(hcols):
                label, color, emoji, advisory = get_aqi_category(preds[i])
                with col:
                    st.markdown(f"""
                    <div class="aqi-hero" style="background: linear-gradient(160deg, {color}33, {color}0d);">
                        <span class="day-tag">📆 Day {i+1} · {date_labels[i]}</span>
                        <div class="aqi-hero-number" style="color:{color};">{preds[i]:.0f}</div>
                        <div class="aqi-hero-label" style="color:{color};">{emoji} {label}</div>
                        <div class="aqi-hero-date">{advisory}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # ---------------- GAUGE CHARTS ----------------
            st.markdown("### 🧭 AQI Gauges")
            gcols = st.columns(3)
            for i, col in enumerate(gcols):
                with col:
                    st.plotly_chart(aqi_gauge_chart(preds[i], f"Day {i+1} — {date_labels[i]}"), use_container_width=True)

            # ---------------- TREND CHART: HISTORY + FORECAST ----------------
            st.markdown("### 📈 AQI Trend — Last 14 Days + 3-Day Model Forecast")

            hist_tail = historical.tail(14).copy()
            hist_dates = list(hist_tail.index.date)
            hist_values = list(hist_tail["AQI"].values)

            trend_fig = go.Figure()

            # Historical actual AQI
            trend_fig.add_trace(go.Scatter(
                x=hist_dates,
                y=hist_values,
                mode="lines+markers",
                name="Historical AQI",
                line=dict(color="#5dade2", width=3),
                marker=dict(size=6),
            ))

            # Bridge point connecting history to forecast
            bridge_x = [hist_dates[-1]] + [d.date() for d in dates]
            bridge_y = [hist_values[-1]] + preds

            trend_fig.add_trace(go.Scatter(
                x=bridge_x,
                y=bridge_y,
                mode="lines+markers",
                name="Model Forecast",
                line=dict(color="#f39c12", width=3, dash="dash"),
                marker=dict(size=10, symbol="diamond"),
            ))

            # Hazard threshold reference lines
            for level, color, name in [(100, "#f1c40f", "Moderate"), (150, "#e67e22", "Unhealthy (Sensitive)"),
                                        (200, "#e74c3c", "Unhealthy"), (300, "#8e44ad", "Very Unhealthy")]:
                trend_fig.add_hline(y=level, line_dash="dot", line_color=color, opacity=0.5,
                                     annotation_text=name, annotation_font_color=color, annotation_font_size=10)

            trend_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                font=dict(color="#dfe9f3"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Date"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="AQI"),
                height=420,
            )
            st.plotly_chart(trend_fig, use_container_width=True)

            # ---------------- SHAP EXPLAINABILITY ----------------
            st.markdown("### 🧠 Model Interpretation")
            for label in ["Day 1", "Day 2", "Day 3"]:
                model_info = models[label]
                render_shap_explainability(
                    model_info["model"],
                    model_info["features"],
                    historical,
                    forecast_daily,
                    label,
                )

            # ---------------- RAW TABLE ----------------
            with st.expander("📋 View Raw Forecast Table"):
                headers = ["Forecast Date", "Predicted AQI", "Category"]
                rows = []
                for i, p in enumerate(preds):
                    cat_label, cat_color, cat_emoji, _ = get_aqi_category(p)
                    cat_html = (
                        f'<span class="quality-chip" style="background:{cat_color}26; '
                        f'color:{cat_color}; border:1px solid {cat_color}66;">{cat_emoji} {cat_label}</span>'
                    )
                    rows.append([date_labels[i], f"{p:.1f}", cat_html])
                render_dark_table(headers, rows, numeric_cols={1})

        except Exception as err:
            st.error(f"Prediction Error: {err}")

st.divider()

# --- MODEL PERFORMANCE / EVALUATION SECTION ---
# Shown only on demand: click the button to reveal the full evaluation
# section (score cards, charts, insight box, metrics table) below it.
if "show_model_performance" not in st.session_state:
    st.session_state["show_model_performance"] = False

btn_label = (
    "🔽 Hide Model Performance & Evaluation"
    if st.session_state["show_model_performance"]
    else "📊 Show Model Performance & Evaluation"
)
if st.button(btn_label, key="toggle_model_performance", type="primary"):
    st.session_state["show_model_performance"] = not st.session_state["show_model_performance"]

if st.session_state["show_model_performance"]:
    render_model_performance()