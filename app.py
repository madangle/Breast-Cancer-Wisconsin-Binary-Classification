"""
Breast Cancer Diagnosis Lab — Streamlit App
Calls modular functions from machine_learning.py
"""

import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from machine_learning import (
    load_default_dataset,
    load_csv_dataset,
    split_and_scale,
    train_all_models,
    plot_confusion_matrix,
    plot_roc_curves,
    SCALE_REQUIRED,
)

warnings.filterwarnings("ignore")

# Page config 
st.set_page_config(
    page_title="Cancer Diagnosis | Breast Cancer Wisconsin Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:       #F7F5F0;
    --surface:  #FFFFFF;
    --border:   #E2DDD6;
    --accent:   #2E6B4F;
    --red:      #C0392B;
    --ink:      #1A1915;
    --muted:    #7A766E;
    --good-bg:  #EEF7F2;
    --bad-bg:   #FDF0EF;
    --good-bdr: #B8DCC9;
    --bad-bdr:  #F0C0BB;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--ink);
}
.stApp { background-color: var(--bg) !important; }
section[data-testid="stSidebar"],
button[data-testid="collapsedControl"] { display: none !important; }

/* ── Header ── */
.top-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.4rem 0 1rem 0;
    border-bottom: 2px solid var(--ink);
    margin-bottom: 0;
}
.brand-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.75rem; font-weight: 400;
    color: var(--ink); letter-spacing: -0.01em;
}
.brand-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem; font-weight: 300;
    color: var(--muted); letter-spacing: 0.08em;
    text-transform: uppercase; margin-left: 0.8rem;
}
.badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem; font-weight: 400;
    letter-spacing: 0.12em; text-transform: uppercase;
    background: var(--good-bg); color: var(--accent);
    border: 1px solid var(--good-bdr);
    border-radius: 2px; padding: 0.3rem 0.7rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important; padding: 0 !important; margin-bottom: 1.8rem !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important; font-weight: 400 !important;
    letter-spacing: 0.1em !important; text-transform: uppercase !important;
    color: var(--muted) !important; background: transparent !important;
    border: none !important; border-bottom: 2px solid transparent !important;
    padding: 0.9rem 1.4rem !important; margin: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: var(--ink) !important;
    border-bottom: 2px solid var(--ink) !important;
    background: transparent !important;
}

/* ── Section head ── */
.sec-head {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem; font-weight: 400;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 0.9rem;
    padding-bottom: 0.45rem; border-bottom: 1px solid var(--border);
}

/* ── Selectbox ── */
.stSelectbox label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.62rem !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; color: var(--muted) !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important; color: var(--ink) !important;
}

/* ── Button ── */
.stButton > button {
    background: var(--accent) !important; color: #fff !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important; font-weight: 400 !important;
    letter-spacing: 0.18em !important; text-transform: uppercase !important;
    border: none !important; border-radius: 3px !important;
    padding: 0.7rem 2rem !important; width: 100% !important;
    transition: background 0.18s ease !important;
}
.stButton > button:hover { background: #255c42 !important; }

/* ── File uploader ── */
.stFileUploader label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.62rem !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; color: var(--muted) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 4px !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.8rem 1rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.58rem !important; text-transform: uppercase !important;
    letter-spacing: 0.12em !important; color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.5rem !important; color: var(--ink) !important;
}

/* ── Result banners ── */
.result-good {
    background: var(--good-bg); border: 1px solid var(--good-bdr);
    border-left: 4px solid var(--accent);
    border-radius: 4px; padding: 1.4rem 1.8rem; margin-top: 0.5rem;
    display: flex; align-items: center; gap: 1.5rem;
}
.result-bad {
    background: var(--bad-bg); border: 1px solid var(--bad-bdr);
    border-left: 4px solid var(--red);
    border-radius: 4px; padding: 1.4rem 1.8rem; margin-top: 0.5rem;
    display: flex; align-items: center; gap: 1.5rem;
}
.r-icon { font-size: 2rem; line-height: 1; }
.r-verdict {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem; font-weight: 400; color: var(--ink);
}
.r-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--muted); margin-top: 0.3rem;
}
.conf-row { margin-top: 0.8rem; display: flex; align-items: center; gap: 0.8rem; }
.conf-track {
    flex: 1; height: 5px; background: var(--border);
    border-radius: 3px; overflow: hidden;
}
.conf-fill-g { height: 5px; background: var(--accent); border-radius: 3px; }
.conf-fill-r { height: 5px; background: var(--red);    border-radius: 3px; }
.conf-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem; color: var(--ink); min-width: 2.8rem; text-align: right;
}

/* ── Stat strip ── */
.stat-strip {
    display: flex; gap: 1px; background: var(--border);
    border: 1px solid var(--border); border-radius: 4px;
    overflow: hidden; margin-bottom: 1.6rem;
}
.stat-cell { flex: 1; background: var(--surface); padding: 0.85rem 1rem; }
.s-label {
    font-family: 'DM Mono', monospace; font-size: 0.55rem;
    font-weight: 300; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 0.2rem;
}
.s-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem; font-weight: 400; color: var(--ink);
}
.s-good { color: var(--accent); }
.s-bad  { color: var(--red); }

/* ── Dataframe ── */
.stDataFrame thead tr th {
    background: var(--bg) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.6rem !important; letter-spacing: 0.12em !important;
    text-transform: uppercase !important; color: var(--muted) !important;
    border-bottom: 1px solid var(--border) !important;
}
.stDataFrame tbody tr td {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important; color: var(--ink) !important;
    border-bottom: 1px solid var(--border) !important;
    background: var(--surface) !important;
}
hr { border-color: var(--border) !important; margin: 1.6rem 0 !important; }
.stAlert {
    background: var(--surface) !important; border: 1px solid var(--border) !important;
    border-radius: 4px !important; font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important; color: var(--muted) !important;
}
</style>
""", unsafe_allow_html=True)


# Session state 
if "results" not in st.session_state:
    st.session_state.results     = None
if "trained_data" not in st.session_state:
    st.session_state.trained_data = None


# Header 
st.markdown("""
<div class="top-header">
    <div>
        <span class="brand-title">🔬 Cancer Diagnosis</span>
        <span class="brand-sub">Breast Cancer Wisconsin · Binary Classification</span>
    </div>
    <span class="badge">30 Features · 6 Classifiers</span>
</div>
""", unsafe_allow_html=True)


# Tabs 
tab_train, tab_eval, tab_compare, tab_about = st.tabs([
    "01 · Dataset & Train",
    "02 · Evaluate Model",
    "03 · Compare All",
    "04 · About",
])


# ═══════════════════════════════════════════════════
#  TAB 1 — Dataset & Train
# ═══════════════════════════════════════════════════
with tab_train:
    st.markdown('<div class="sec-head">Data Source</div>', unsafe_allow_html=True)

    src_col, info_col = st.columns([3, 2])

    with src_col:
        uploaded = st.file_uploader(
            "Upload CSV (optional — 30 features + target column)",
            type=["csv"],
            help="Must contain the 30 Breast Cancer Wisconsin feature columns plus a 'target' column (0=Malignant, 1=Benign).",
        )

    with info_col:
        if uploaded is None:
            st.info(
                "**No file uploaded.**  \n"
                "The built-in sklearn Breast Cancer dataset will be used  \n"
                "(569 samples · 30 features · binary labels)."
            )
        else:
            st.success(f"File accepted: **{uploaded.name}**")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-head">Training</div>', unsafe_allow_html=True)

    run_col, _ = st.columns([2, 4])
    with run_col:
        train_btn = st.button("Train All 6 Models →")

    if train_btn:
        with st.spinner("Loading data and training models …"):
            try:
                # Load 
                if uploaded is not None:
                    X, y, feature_names = load_csv_dataset(
                        io.StringIO(uploaded.read().decode("utf-8"))
                    )
                    data_source = f"CSV upload — {uploaded.name}"
                else:
                    X, y, feature_names = load_default_dataset()
                    data_source = "Built-in sklearn Breast Cancer dataset"

                # Split & scale 
                X_train, X_test, X_tr_sc, X_te_sc, y_train, y_test, scaler = split_and_scale(X, y)

                # Train 
                results = train_all_models(X_train, X_test, X_tr_sc, X_te_sc, y_train, y_test)

                # Cache 
                st.session_state.results = results
                st.session_state.trained_data = {
                    "source":       data_source,
                    "n_total":      len(X),
                    "n_train":      len(X_train),
                    "n_test":       len(X_test),
                    "n_features":   X.shape[1],
                    "class_dist":   y.value_counts().to_dict(),
                    "feature_names": feature_names,
                }
                st.success("All 6 models trained successfully!")

            except Exception as e:
                st.error(f"Training failed: {e}")

    # Show dataset summary if trained
    if st.session_state.trained_data is not None:
        td = st.session_state.trained_data
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-head">Dataset Summary</div>', unsafe_allow_html=True)

        benign_n    = td["class_dist"].get(1, 0)
        malignant_n = td["class_dist"].get(0, 0)
        strip_html = f"""
        <div class="stat-strip">
            <div class="stat-cell"><div class="s-label">Source</div>
                <div class="s-value" style="font-size:0.82rem;font-family:'DM Mono',monospace;">{td['source'][:32]}…</div></div>
            <div class="stat-cell"><div class="s-label">Total Samples</div>
                <div class="s-value">{td['n_total']}</div></div>
            <div class="stat-cell"><div class="s-label">Training Set</div>
                <div class="s-value">{td['n_train']}</div></div>
            <div class="stat-cell"><div class="s-label">Test Set</div>
                <div class="s-value">{td['n_test']}</div></div>
            <div class="stat-cell"><div class="s-label">Features</div>
                <div class="s-value">{td['n_features']}</div></div>
            <div class="stat-cell"><div class="s-label">Benign</div>
                <div class="s-value s-good">{benign_n}</div></div>
            <div class="stat-cell"><div class="s-label">Malignant</div>
                <div class="s-value s-bad">{malignant_n}</div></div>
        </div>"""
        st.markdown(strip_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
#  TAB 2 — Evaluate Single Model
# ═══════════════════════════════════════════════════
with tab_eval:
    if st.session_state.results is None:
        st.info("Train the models first in **01 · Dataset & Train**.")
    else:
        results = st.session_state.results

        st.markdown('<div class="sec-head">Select Classifier</div>', unsafe_allow_html=True)
        sel_col, _ = st.columns([3, 3])
        with sel_col:
            selected = st.selectbox("Model", list(results.keys()), label_visibility="visible")

        res = results[selected]
        m   = res["metrics"]

        # Metrics strip 
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-head">Performance Metrics</div>', unsafe_allow_html=True)

        strip_html = f"""
        <div class="stat-strip">
            <div class="stat-cell"><div class="s-label">Accuracy</div>
                <div class="s-value">{m['Accuracy']:.4f}</div></div>
            <div class="stat-cell"><div class="s-label">AUC-ROC</div>
                <div class="s-value">{m['AUC']:.4f}</div></div>
            <div class="stat-cell"><div class="s-label">Precision</div>
                <div class="s-value">{m['Precision']:.4f}</div></div>
            <div class="stat-cell"><div class="s-label">Recall</div>
                <div class="s-value">{m['Recall']:.4f}</div></div>
            <div class="stat-cell"><div class="s-label">F1 Score</div>
                <div class="s-value">{m['F1']:.4f}</div></div>
            <div class="stat-cell"><div class="s-label">MCC</div>
                <div class="s-value">{m['MCC']:.4f}</div></div>
        </div>"""
        st.markdown(strip_html, unsafe_allow_html=True)

        # Confusion matrix + classification report 
        cm_col, rep_col = st.columns([2, 3])

        with cm_col:
            st.markdown('<div class="sec-head">Confusion Matrix</div>', unsafe_allow_html=True)
            fig = plot_confusion_matrix(res["confusion_matrix"], selected)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with rep_col:
            st.markdown('<div class="sec-head">Classification Report</div>', unsafe_allow_html=True)
            report = res["classification_report"]
            rows = []
            for label in ["Malignant", "Benign", "macro avg", "weighted avg"]:
                key = label.lower() if label in ["Malignant", "Benign"] else label
                if key in report:
                    d = report[key]
                    rows.append({
                        "Class":     label,
                        "Precision": f"{d['precision']:.4f}",
                        "Recall":    f"{d['recall']:.4f}",
                        "F1-Score":  f"{d['f1-score']:.4f}",
                        "Support":   int(d["support"]),
                    })
            rep_df = pd.DataFrame(rows)
            st.dataframe(rep_df, use_container_width=True, hide_index=True)

            # Accuracy row
            acc_row = report.get("accuracy", None)
            if acc_row is not None:
                st.markdown(
                    f'<p style="font-family:\'DM Mono\',monospace;font-size:0.72rem;'
                    f'color:#7A766E;margin-top:0.5rem;">'
                    f'Overall Accuracy: <strong style="color:#1A1915;">{acc_row:.4f}</strong></p>',
                    unsafe_allow_html=True,
                )

        # Scaling note ─
        if selected in SCALE_REQUIRED:
            st.markdown(
                '<p style="font-family:\'DM Mono\',monospace;font-size:0.62rem;'
                'color:#7A766E;margin-top:0.8rem;">'
                '⚠ StandardScaler applied for this model.</p>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════
#  TAB 3 — Compare All Models
# ═══════════════════════════════════════════════════
with tab_compare:
    if st.session_state.results is None:
        st.info("Train the models first in **01 · Dataset & Train**.")
    else:
        results = st.session_state.results
        rows = [{"Model": n, **r["metrics"]} for n, r in results.items()]
        compare_df = pd.DataFrame(rows)

        # Summary metrics ──
        st.markdown('<div class="sec-head">Summary</div>', unsafe_allow_html=True)
        best  = compare_df.loc[compare_df["Accuracy"].idxmax()]
        worst = compare_df.loc[compare_df["Accuracy"].idxmin()]
        avg   = compare_df["Accuracy"].mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Models",        len(compare_df))
        m2.metric("Best Accuracy", f"{best['Accuracy']:.4f}",  delta=best["Model"])
        m3.metric("Avg Accuracy",  f"{avg:.4f}")
        m4.metric("Lowest",        f"{worst['Accuracy']:.4f}", delta=worst["Model"], delta_color="inverse")

        st.markdown("<br>", unsafe_allow_html=True)

        # Table + bar chart 
        tbl_col, chart_col = st.columns([5, 4])
        with tbl_col:
            st.markdown('<div class="sec-head">All Metrics</div>', unsafe_allow_html=True)
            st.dataframe(compare_df, use_container_width=True, hide_index=True)

        with chart_col:
            st.markdown('<div class="sec-head">Accuracy Comparison</div>', unsafe_allow_html=True)
            st.bar_chart(
                compare_df.set_index("Model")[["Accuracy"]].sort_values("Accuracy"),
                color="#2E6B4F", height=260,
            )

        # ROC curves ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-head">ROC Curves</div>', unsafe_allow_html=True)
        roc_fig = plot_roc_curves(results)
        roc_col, _ = st.columns([3, 2])
        with roc_col:
            st.pyplot(roc_fig, use_container_width=True)
        plt.close(roc_fig)

        # Metric heatmap ───
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-head">Metrics Heatmap</div>', unsafe_allow_html=True)

        heat_df = compare_df.set_index("Model")
        fig_h, ax_h = plt.subplots(figsize=(8, 3.2))
        fig_h.patch.set_facecolor("#F7F5F0")
        ax_h.set_facecolor("#F7F5F0")
        sns_map = plt.get_cmap("YlGn")
        import seaborn as sns
        sns.heatmap(
            heat_df, annot=True, fmt=".3f", cmap="YlGn",
            linewidths=0.4, linecolor="#E2DDD6",
            annot_kws={"size": 8},
            ax=ax_h, cbar=False,
        )
        ax_h.tick_params(colors="#7A766E", labelsize=8)
        ax_h.set_xlabel("", labelpad=0)
        ax_h.set_ylabel("", labelpad=0)
        plt.tight_layout()
        st.pyplot(fig_h, use_container_width=True)
        plt.close(fig_h)


# ═══════════════════════════════════════════════════
#  TAB 4 — About
# ═══════════════════════════════════════════════════
with tab_about:
    st.markdown('<div class="sec-head">Dataset & Methodology</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Breast Cancer Wisconsin (Diagnostic)**  
569 samples · 30 numerical features · binary labels  
*(0 = Malignant, 1 = Benign)*

Features are computed from digitised images of fine-needle aspirate (FNA) biopsies.
They describe characteristics of the cell nuclei such as radius, texture, perimeter,
area, smoothness, compactness, concavity, symmetry, and fractal dimension —
each captured as mean, standard error, and worst value.

**CSV Upload**  
Upload test-only data for evaluation in memory-constrained environments.
The CSV must include all 30 feature columns matching `sklearn.datasets.load_breast_cancer()`
plus a `target` column (0/1).
        """)
    with c2:
        st.markdown("""
**Classifiers**

| Model | Scaling |
|---|---|
| Logistic Regression | ✓ StandardScaler |
| Decision Tree | — |
| k-Nearest Neighbours | ✓ StandardScaler |
| Naïve Bayes | — |
| Random Forest | — |
| XGBoost | ✓ StandardScaler |

**Evaluation Metrics**  
Accuracy · AUC-ROC · Precision · Recall · F1 Score · MCC  
Confusion matrix and per-class classification report for each model.

`random_state=42` · 80/20 stratified split
        """)
    st.markdown("---")
    st.caption("Cancer Diagnosis Lab · Streamlit · UCI ML Repository · sklearn · © 2025")
