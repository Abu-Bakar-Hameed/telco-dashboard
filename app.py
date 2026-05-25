"""
app.py — Telco Customer Churn Prediction Dashboard
Streamlit multi-section interactive web application.
Run: streamlit run app.py
"""

import os
import sys
import warnings
import time

warnings.filterwarnings("ignore")
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── local imports ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_raw_data,
    get_data_profile,
    preprocess,
    preprocess_single_input,
    train_all_models,
    tune_models,
    evaluate_model,
    save_model,
    load_model,
    save_scaler,
    load_scaler,
    save_feature_names,
    load_feature_names,
    plot_churn_distribution,
    plot_tenure_distribution,
    plot_monthly_charges,
    plot_contract_churn,
    plot_internet_churn,
    plot_payment_churn,
    plot_correlation_heatmap,
    plot_senior_citizen,
    plot_metrics_bar,
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrices,
    plot_feature_importance,
    build_leaderboard,
    get_models,
)

# ── page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📡 Telco Customer Churn Prediction Dashboard")
st.caption("EDA • Machine Learning • Model Comparison • Predictions")

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1f3c 50%, #0a2a2a 100%);
    padding: 2.5rem 2rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid #1a3a5c;
}
.main-header h1 {
    font-family: 'Space Mono', monospace;
    color: #00c2cb;
    font-size: 2.2rem;
    letter-spacing: -1px;
    margin: 0;
}
.main-header p { color: #8ab4c4; margin: 0.4rem 0 0; font-size: 1rem; }

.metric-card {
    background: linear-gradient(145deg, #0d1f3c, #0a2a2a);
    border: 1px solid #1a3a5c;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.metric-card .val {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #00c2cb;
}
.metric-card .lbl { color: #8ab4c4; font-size: 0.85rem; margin-top: 4px; }

.churn-badge-high {
    background: rgba(255, 75, 110, 0.13);
    border: 1px solid #ff4b6e;
    color: #ff4b6e;
    padding: 0.6rem 1.4rem;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    display: inline-block;
    margin-top: 0.5rem;
}
.churn-badge-low {
    background: rgba(0, 194, 203, 0.13);
    border: 1px solid #00c2cb;
    color: #00c2cb;
    padding: 0.6rem 1.4rem;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    display: inline-block;
    margin-top: 0.5rem;
}
.section-title {
    font-family: 'Space Mono', monospace;
    color: #00c2cb;
    border-left: 4px solid #00c2cb;
    padding-left: 0.7rem;
    margin-bottom: 1rem;
}
.champion-banner {
    background: linear-gradient(90deg, #0a2a2a, #0d1f3c);
    border: 2px solid #00c2cb;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}
.stButton>button {
    background: linear-gradient(135deg, #00c2cb, #0077b6);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    padding: 0.55rem 1.5rem;
    transition: all 0.2s;
}
.stButton>button:hover { opacity: 0.88; transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='text-align:center; color:#4CAF50;'>📡 Navigation</h2>",
        unsafe_allow_html=True,
    )

    # FIX: option labels must exactly match the if/elif page checks below
    page = option_menu(
        menu_title=None,
        options=[
            "Home",
            "Dataset Explorer",
            "EDA Dashboard",
            "Model Training",
            "Model Comparison",
            "Churn Predictor",
        ],
        icons=["house", "database", "bar-chart", "cpu", "trophy", "activity"],
        default_index=0,
        styles={
            "container": {
                "padding": "8px",
                "background-color": "#0E1117",
                "border-radius": "10px",
            },
            "icon": {"color": "#4CAF50", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "4px",
                "border-radius": "8px",
                "padding": "8px",
                "color": "#ffffff",
            },
            "nav-link-selected": {
                "background-color": "#1f77b4",
                "color": "white",
                "font-weight": "bold",
            },
        },
    )

    st.markdown("---")

    st.markdown(
        """
        <div style='padding:10px; background-color:#111827; border-radius:10px;'>
            <h4 style='color:#4CAF50;'>📊 Dataset Info</h4>
            <p style='margin:0; color:white;'>
                <b>Telco Customer Churn</b><br>
                Records: 7,043<br>
                Features: 21
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style='padding:10px; margin-top:10px; background-color:#111827; border-radius:10px;'>
            <h4 style='color:#4CAF50;'>🤖 Models</h4>
            <p style='margin:0; color:white;'>
                Logistic Regression<br>
                Decision Tree<br>
                Random Forest<br>
                KNN<br>
                SVM
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Data loading (cached) ──────────────────────────────────────────────────────
@st.cache_data
def get_raw():
    return load_raw_data()


@st.cache_data
def get_preprocessed():
    raw = load_raw_data()
    X_train, X_test, y_train, y_test, feat, scaler, df_clean = preprocess(raw)
    return X_train, X_test, y_train, y_test, feat, scaler, df_clean


df_raw = get_raw()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":

    st.markdown("""
    <div class="main-header">
        <h1>📡 Telco Customer Churn Intelligence Platform</h1>
        <p>
            AI-Powered Customer Retention Analytics System
            · Machine Learning Insights · Interactive Business Dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # KPI CARDS
    # ─────────────────────────────────────────────

    col1, col2, col3, col4 = st.columns(4)

    total_customers = len(df_raw)
    churned_customers = (df_raw["Churn"] == "Yes").sum()
    retention_rate = round(
        ((total_customers - churned_customers) / total_customers) * 100, 1
    )
    total_features = df_raw.shape[1] - 1

    metrics = [
        (f"{total_customers:,}", "👥 Total Customers"),
        (f"{churned_customers:,}", "⚠️ Churned Customers"),
        (f"{retention_rate}%", "💙 Retention Rate"),
        (f"{total_features}", "📊 Business Features"),
    ]

    for col, (value, label) in zip([col1, col2, col3, col4], metrics):
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="val">{value}</div>
                <div class="lbl">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ─────────────────────────────────────────────
    # HERO SECTION
    # ─────────────────────────────────────────────

    left, right = st.columns([1.6, 1])

    with left:

        st.markdown(
            '<h2 class="section-title">🚀 Why This Project Matters</h2>',
            unsafe_allow_html=True,
        )

        st.markdown("""
Customer churn is one of the biggest challenges faced by telecom companies.

This intelligent analytics platform helps businesses identify customers
who are likely to leave the service before churn actually happens.

Using Machine Learning and advanced customer behavior analysis,
companies can take proactive retention actions such as:

✅ Personalized offers  
✅ Customer loyalty campaigns  
✅ Contract optimization  
✅ Discount recommendations  
✅ Customer support prioritization  

The goal is simple:

> **Reduce customer loss and increase long-term revenue growth.**
        """)

        st.markdown("---")

        st.markdown(
            '<h2 class="section-title">🧠 Key Capabilities</h2>',
            unsafe_allow_html=True,
        )

        feature_col1, feature_col2 = st.columns(2)

        with feature_col1:
            st.markdown("""
### 📊 Smart Analytics
- Interactive EDA dashboards
- Customer behavior insights
- Churn trend visualization
- Correlation analysis

### 🤖 Machine Learning
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine
            """)

        with feature_col2:
            st.markdown("""
### 📈 Model Evaluation
- Accuracy comparison
- Precision & Recall
- ROC-AUC analysis
- Confusion matrices
- Feature importance

### 🔮 AI Prediction Engine
- Real-time churn prediction
- Confidence scoring
- Business risk analysis
- Retention recommendations
            """)

    with right:

        st.plotly_chart(
            plot_churn_distribution(df_raw),
            use_container_width=True
        )

        st.markdown("---")

        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg,#0d1f3c,#0a2a2a);
                padding:1.2rem;
                border-radius:14px;
                border:1px solid #1a3a5c;
            ">

            <h3 style="color:#00c2cb;">📌 Project Highlights</h3>

            <ul style="line-height:2;">
                <li>✔ End-to-End ML Workflow</li>
                <li>✔ Interactive Streamlit Dashboard</li>
                <li>✔ Advanced Plotly Visualizations</li>
                <li>✔ Multiple ML Model Comparison</li>
                <li>✔ Hyperparameter Optimization</li>
                <li>✔ Real-Time Prediction System</li>
                <li>✔ Business Intelligence Insights</li>
            </ul>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ─────────────────────────────────────────────
    # BUSINESS IMPACT SECTION
    # ─────────────────────────────────────────────

    st.markdown(
        '<h2 class="section-title">📈 Business Impact</h2>',
        unsafe_allow_html=True,
    )

    impact1, impact2, impact3 = st.columns(3)

    with impact1:
        st.markdown("""
### 💰 Revenue Protection

Predicting churn early helps telecom companies
reduce customer loss and protect recurring revenue streams.
        """)

    with impact2:
        st.markdown("""
### 🎯 Better Decision Making

Business teams can use data-driven insights
to improve retention strategies and customer satisfaction.
        """)

    with impact3:
        st.markdown("""
### ⚡ Operational Efficiency

Machine Learning automates churn detection,
reducing manual analysis time and improving scalability.
        """)

    st.markdown("---")

    # ─────────────────────────────────────────────
    # FOOTER SECTION
    # ─────────────────────────────────────────────

    st.markdown(
        """
        <div style='
            text-align:center;
            padding:1rem;
            border-radius:12px;
            background:#0E1117;
            border:1px solid #1f2937;
        '>

        <h3 style='color:#00c2cb;'>
            📡 AI-Powered Telecom Customer Intelligence System
        </h3>

        <p style='color:#9ca3af;'>
            Built with Streamlit · Plotly · Scikit-Learn · Pandas · NumPy
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dataset Explorer":
    st.markdown('<h2 class="section-title">Dataset Explorer</h2>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 Raw Data", "📈 Statistics", "🔍 Column Profiles", "⚠️ Data Quality"]
    )

    with tab1:
        st.markdown(f"**Shape:** {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
        search = st.text_input("🔎 Filter customerID (optional)")
        view   = df_raw[df_raw["customerID"].str.contains(search)] if search else df_raw
        st.dataframe(view, use_container_width=True, height=450)

    with tab2:
        st.markdown("**Numeric Summary Statistics**")
        st.dataframe(df_raw.describe().round(2), use_container_width=True)

    with tab3:
        col = st.selectbox("Select a column", df_raw.columns.tolist())
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Data Type:** `{df_raw[col].dtype}`")
            st.markdown(f"**Unique Values:** {df_raw[col].nunique()}")
            st.markdown(f"**Null Count:** {df_raw[col].isnull().sum()}")
        with c2:
            if df_raw[col].dtype == "object":
                vc = df_raw[col].value_counts().reset_index()
                vc.columns = [col, "Count"]
                st.dataframe(vc, use_container_width=True)
            else:
                st.plotly_chart(
                    px.histogram(df_raw, x=col, title=f"Distribution: {col}"),
                    use_container_width=True,
                )

    with tab4:
        profile = get_data_profile(df_raw)
        st.markdown(f"**Duplicate Rows:** {profile['duplicate_rows']}")
        null_df = pd.DataFrame.from_dict(
            profile["null_counts"], orient="index", columns=["Nulls"]
        )
        null_df = null_df[null_df["Nulls"] > 0]
        if null_df.empty:
            st.success("✅ No explicit null values — note: TotalCharges may contain empty strings.")
        else:
            st.dataframe(null_df, use_container_width=True)

        st.markdown("**Churn Class Balance**")
        churn_df = pd.DataFrame.from_dict(
            profile["churn_distribution"], orient="index", columns=["Count"]
        )
        churn_df["Pct"] = (
            (churn_df["Count"] / churn_df["Count"].sum() * 100).round(1).astype(str) + "%"
        )
        st.dataframe(churn_df, use_container_width=True)
        st.plotly_chart(plot_churn_distribution(df_raw), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA Dashboard":
    st.markdown(
        '<h2 class="section-title">Exploratory Data Analysis</h2>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        ["📐 Univariate Analysis", "🔗 Bivariate Analysis", "🌡️ Correlation Analysis"]
    )

    # ── TAB 1: UNIVARIATE ──────────────────────────────────────────────────────
    with tab1:
        st.markdown("### Customer Distribution Overview")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_tenure_distribution(df_raw), use_container_width=True)
        with c2:
            st.plotly_chart(plot_monthly_charges(df_raw), use_container_width=True)

        st.plotly_chart(plot_senior_citizen(df_raw), use_container_width=True)

        st.markdown("---")
        st.markdown("### Monthly Charges vs Churn")
        fig_mc = px.histogram(
            df_raw,
            x="MonthlyCharges",
            nbins=50,
            color="Churn",
            color_discrete_map={"Yes": "#ff4b6e", "No": "#00c2cb"},
            title="Monthly Charges Distribution by Churn Status",
            marginal="box",
        )
        st.plotly_chart(fig_mc, use_container_width=True)

    # ── TAB 2: BIVARIATE ───────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Relationship Between Features and Churn")
        st.info("Select a feature to analyze how it impacts churn behavior.")

        cat_cols = [
            col for col in df_raw.columns
            if col not in ["customerID", "Churn"]
            and (df_raw[col].dtype == "object" or df_raw[col].nunique() <= 10)
        ]

        if not cat_cols:
            st.warning("No categorical or low-cardinality features found.")

        else:
            sel_col = st.selectbox(
                "Select Feature",
                options=cat_cols,
                index=0,
                key="bivariate_selector"
            )

            st.markdown(f"#### 📊 {sel_col} vs Churn Analysis")

            ct = (
                df_raw.groupby([sel_col, "Churn"], dropna=False)
                .size()
                .reset_index(name="Count")
            )

            fig_biv = px.bar(
                ct,
                x=sel_col,
                y="Count",
                color="Churn",
                barmode="group",
                title=f"{sel_col} vs Churn",
                color_discrete_map={"Yes": "#ff4b6e", "No": "#00c2cb"},
            )

            fig_biv.update_layout(xaxis_tickangle=-30, template="plotly_white")

            st.plotly_chart(fig_biv, use_container_width=True)

        # ─────────────────────────────────────
        # 🔥 ADD YOUR PRE-BUILT BUSINESS PLOTS HERE
        # ─────────────────────────────────────

        st.markdown("---")
        st.markdown("### 🔥 Key Business Drivers of Churn")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.plotly_chart(plot_contract_churn(df_raw), use_container_width=True)

        with col2:
            st.plotly_chart(plot_internet_churn(df_raw), use_container_width=True)

        with col3:
            st.plotly_chart(plot_payment_churn(df_raw), use_container_width=True)

        # ── TAB 3: CORRELATION ─────────────────────────────────────────────────────
        with tab3:
            st.markdown("### Feature Correlation Analysis")
            st.plotly_chart(plot_correlation_heatmap(df_raw), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL TRAINING  (single merged block — FIX: was duplicated)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Training":
    st.markdown('<h2 class="section-title">Model Training</h2>', unsafe_allow_html=True)

    X_train, X_test, y_train, y_test, feat, scaler, df_clean = get_preprocessed()

    st.markdown(f"""
    - **Training samples:** {len(X_train):,}
    - **Test samples:** {len(X_test):,}
    - **Features after encoding:** {len(feat)}
    - **Stratified split:** 80/20
    """)

    model_names    = list(get_models().keys())
    selected       = st.selectbox("Select a model to train", model_names, index=0)
    include_tuning = st.checkbox(
        "Also run GridSearchCV tuning (Random Forest only, takes ~60s)", value=False
    )

    # ── Train single model ─────────────────────────────────────────────────────
    if st.button("▶ Train Selected Model"):
        with st.spinner(f"Training {selected}..."):
            model   = get_models()[selected]
            t0      = time.time()
            model.fit(X_train, y_train)
            elapsed = round(time.time() - t0, 2)
            metrics = evaluate_model(model, X_test, y_test)
            save_model(model, selected)
            save_scaler(scaler)
            save_feature_names(feat)

        st.success(f"✅ Trained in {elapsed}s — model serialized to `models/`")

        # Gauge charts
        score_meta = [
            ("accuracy",  "Accuracy",  "#1D9E75"),
            ("precision", "Precision", "#378ADD"),
            ("recall",    "Recall",    "#7F77DD"),
            ("f1",        "F1 Score",  "#D85A30"),
            ("roc_auc",   "ROC-AUC",   "#D4537E"),
        ]

        fig_gauges = make_subplots(
            rows=1,
            cols=5,
            specs=[[{"type": "indicator"}] * 5],
            horizontal_spacing=0.04,
        )
        for i, (key, label, color) in enumerate(score_meta, start=1):
            val = round(metrics[key] * 100, 2)
            fig_gauges.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=val,
                    number={"suffix": "%", "font": {"size": 22, "color": color}},
                    title={"text": f"<b>{label}</b>", "font": {"size": 13}},
                    gauge={
                        "axis": {"range": [0, 100], "visible": False},
                        "bar":  {"color": color, "thickness": 0.25},
                        "bgcolor": "rgba(0,0,0,0.06)",
                        "borderwidth": 0,
                    },
                ),
                row=1,
                col=i,
            )
        fig_gauges.update_layout(
            height=220,
            margin=dict(t=60, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_gauges, use_container_width=True)

        # Performance bar chart
        bar_labels = [m[1] for m in score_meta]
        bar_values = [round(metrics[m[0]] * 100, 2) for m in score_meta]
        bar_colors = [m[2] for m in score_meta]

        bar_fig = go.Figure(
            go.Bar(
                x=bar_labels,
                y=bar_values,
                marker_color=bar_colors,
                text=[f"{v:.2f}%" for v in bar_values],
                textposition="outside",
                cliponaxis=False,
            )
        )
        bar_fig.update_layout(
            title=f"Model Performance Overview — {selected}",
            yaxis=dict(range=[0, 115], title="Score (%)", ticksuffix="%"),
            xaxis_title="Metric",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=40, l=50, r=20),
            height=340,
            showlegend=False,
        )
        st.plotly_chart(bar_fig, use_container_width=True)

        # Confusion matrix + feature importance
        c1, c2 = st.columns(2)
        with c1:
            cm = metrics["confusion_matrix"]
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            cell_labels = [
                [f"TN<br>{tn}", f"FP<br>{fp}"],
                [f"FN<br>{fn}", f"TP<br>{tp}"],
            ]
            fig_cm = go.Figure(
                go.Heatmap(
                    z=cm,
                    text=cell_labels,
                    texttemplate="%{text}",
                    colorscale=[[0, "#fce8e8"], [1, "#c8f0e4"]],
                    x=["Pred: No", "Pred: Yes"],
                    y=["Actual: No", "Actual: Yes"],
                    showscale=False,
                    xgap=3,
                    ygap=3,
                )
            )
            fig_cm.update_layout(
                title=f"Confusion Matrix — {selected}",
                height=320,
                margin=dict(t=50, b=40, l=80, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(size=14),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            sensitivity = tp / (tp + fn) if (tp + fn) else 0
            specificity = tn / (tn + fp) if (tn + fp) else 0
            st.markdown(
                f"**Sensitivity:** `{sensitivity:.4f}` &nbsp;|&nbsp; "
                f"**Specificity:** `{specificity:.4f}` &nbsp;|&nbsp; "
                f"**Total samples:** `{tn + fp + fn + tp:,}`"
            )

        with c2:
            if selected in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
                st.plotly_chart(
                    plot_feature_importance(model, feat), use_container_width=True
                )
            else:
                st.info("Feature importance plot available for tree-based models only.")

        # GridSearchCV tuning (Random Forest only)
        if include_tuning and selected == "Random Forest":
            with st.spinner("Running GridSearchCV… (this may take 1–2 min)"):
                tuned_result = tune_models(X_train, y_train)
                if tuned_result is not None:
                    tuned_models, best_params = tuned_result
                    st.markdown("**Best Hyperparameters Found:**")
                    st.json(best_params)
                    for name, tm in tuned_models.items():
                        m2 = evaluate_model(tm, X_test, y_test)
                        save_model(tm, name)
                        st.markdown(
                            f"**{name}** → F1: `{m2['f1']:.4f}` | ROC-AUC: `{m2['roc_auc']:.4f}`"
                        )

    # ── Train ALL 5 models ─────────────────────────────────────────────────────
    st.markdown("---")

    if st.button("🚀 Train ALL 5 Models"):
        with st.spinner("Training all models..."):
            trained_models, results = train_all_models(X_train, X_test, y_train, y_test)
            save_scaler(scaler)
            save_feature_names(feat)
            st.session_state["results"] = results
            st.session_state["trained"] = trained_models

        st.success("✅ All models trained!")

        lb = build_leaderboard(results)

        st.dataframe(
            lb.style.highlight_max(
                subset=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
            ),
            use_container_width=True,
        )

        # Grouped percentage bar chart
        metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        fig_percent = go.Figure()
        for m in metric_cols:
            fig_percent.add_trace(
                go.Bar(
                    name=m,
                    x=lb["Model"],
                    y=(lb[m] * 100).round(2),
                    text=(lb[m] * 100).round(2).astype(str) + "%",
                    textposition="outside",
                )
            )
        fig_percent.update_layout(
            title="📊 Model Performance Comparison (%)",
            barmode="group",
            yaxis=dict(range=[0, 115], ticksuffix="%"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=500,
            legend_title="Metrics",
        )
        st.plotly_chart(fig_percent, use_container_width=True)

        # F1 ranking chart
        fig_f1 = go.Figure(
            go.Bar(
                x=lb["Model"],
                y=lb["F1"],
                text=[f"{v:.3f}" for v in lb["F1"]],
                textposition="outside",
            )
        )
        fig_f1.update_layout(
            title="🏆 F1 Score Comparison Across Models",
            yaxis=dict(range=[0, 1]),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=350,
        )
        st.plotly_chart(fig_f1, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.markdown(
        '<h2 class="section-title">Model Comparison Dashboard</h2>',
        unsafe_allow_html=True,
    )

    results = st.session_state.get("results", None)
    trained = st.session_state.get("trained", None)

    # Load saved models if session state is empty
    if results is None:
        X_train, X_test, y_train, y_test, feat_names, scaler, _ = get_preprocessed()
        model_names = list(get_models().keys())
        loaded = {n: load_model(n) for n in model_names}

        if all(m is not None for m in loaded.values()):
            results = {n: evaluate_model(m, X_test, y_test) for n, m in loaded.items()}
            trained = loaded
        else:
            st.warning("⚠️ No trained models found. Go to Model Training and train models first.")
            st.stop()

    lb = build_leaderboard(results)

    # Champion banner
    champion  = lb.iloc[0]["Model"]
    champ_f1  = lb.iloc[0]["F1"]
    champ_auc = lb.iloc[0]["ROC-AUC"]

    st.markdown(f"""
    <div class="champion-banner">
        🏆 <strong>Champion Model: {champion}</strong><br>
        F1 Score: <strong>{champ_f1:.4f}</strong> | ROC-AUC: <strong>{champ_auc:.4f}</strong><br>
        <span style="font-size:0.85rem">Selected based on highest F1 score for balanced churn prediction.</span>
    </div>
    """, unsafe_allow_html=True)

    # Leaderboard table
    st.markdown("### 📊 Full Metrics Leaderboard")
    st.dataframe(lb, use_container_width=True)

    # Per-metric bar charts
    models = lb["Model"]

    def metric_bar(metric_name, title, color):
        fig = go.Figure(
            go.Bar(
                x=models,
                y=lb[metric_name],
                text=(lb[metric_name] * 100).round(2).astype(str) + "%",
                textposition="outside",
                marker_color=color,
            )
        )
        fig.update_layout(
            title=title,
            yaxis=dict(range=[0, 1.15], tickformat=".0%"),
            height=350,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(metric_bar("Accuracy",  "📊 Accuracy",  "#1D9E75"), use_container_width=True)
        st.plotly_chart(metric_bar("Precision", "🎯 Precision", "#378ADD"), use_container_width=True)
        st.plotly_chart(metric_bar("Recall",    "🔍 Recall",    "#7F77DD"), use_container_width=True)
    with col2:
        st.plotly_chart(metric_bar("F1",      "⚖️ F1 Score", "#D85A30"), use_container_width=True)
        st.plotly_chart(metric_bar("ROC-AUC", "📈 ROC-AUC",  "#D4537E"), use_container_width=True)

    # Confusion matrix for best model
    st.markdown("### 🎯 Confusion Matrix (Best Model)")
    best_model_name = lb.iloc[0]["Model"]
    cm              = results[best_model_name]["confusion_matrix"]
    tn, fp, fn, tp  = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    fig_cm = go.Figure(
        go.Heatmap(
            z=cm,
            text=[[f"TN<br>{tn}", f"FP<br>{fp}"], [f"FN<br>{fn}", f"TP<br>{tp}"]],
            texttemplate="%{text}",
            colorscale=[[0, "#fce8e8"], [1, "#c8f0e4"]],
            showscale=False,
            x=["Pred: No", "Pred: Yes"],
            y=["Actual: No", "Actual: Yes"],
            xgap=3,
            ygap=3,
        )
    )
    fig_cm.update_layout(
        title=f"Confusion Matrix — {best_model_name}",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # Advanced analysis tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Metrics Overview",
        "📈 ROC Curves",
        "🔄 PR Curves",
        "🎯 Confusion Matrices",
        "🌟 Feature Importance",
    ])

    with tab1:
        st.plotly_chart(plot_metrics_bar(results), use_container_width=True)

    with tab2:
        st.plotly_chart(plot_roc_curves(results), use_container_width=True)

    with tab3:
        st.plotly_chart(plot_pr_curves(results), use_container_width=True)

    with tab4:
        st.plotly_chart(plot_confusion_matrices(results), use_container_width=True)

    with tab5:
        tree_models = {
            n: m for n, m in trained.items() if hasattr(m, "feature_importances_")
        }
        if tree_models:
            best_tree       = max(tree_models, key=lambda n: results[n]["roc_auc"])
            feat_names_file = load_feature_names()
            if feat_names_file:
                st.plotly_chart(
                    plot_feature_importance(tree_models[best_tree], feat_names_file),
                    use_container_width=True,
                )
        else:
            st.info("No tree-based models available.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CHURN PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Churn Predictor":
    st.markdown(
        '<h2 class="section-title">🔮 Live Churn Predictor</h2>',
        unsafe_allow_html=True,
    )
    st.markdown("Predict customer churn risk using trained machine learning models.")

    scaler_loaded = load_scaler()
    feat_loaded   = load_feature_names()
    model_names   = list(get_models().keys())
    available     = [n for n in model_names if load_model(n) is not None]

    if not available or scaler_loaded is None or feat_loaded is None:
        st.warning("⚠️ No trained models found. Please train models first.")
        st.stop()

    chosen_model = st.selectbox("🤖 Select Model", available, index=0)
    st.markdown("---")

    # Customer input form
    st.subheader("🧾 Customer Profile")
    c1, c2, c3 = st.columns(3)

    with c1:
        gender     = st.selectbox("Gender", ["Male", "Female"])
        senior     = st.selectbox("Senior Citizen", ["0", "1"])
        partner    = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure     = st.slider("Tenure (months)", 1, 72, 12)
        phone      = st.selectbox("Phone Service", ["Yes", "No"])

    with c2:
        multiple_lines    = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet          = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security   = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup     = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support      = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    with c3:
        streaming_tv     = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract         = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless        = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment          = st.selectbox("Payment Method", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ])
        monthly_charges  = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
        total_charges    = st.number_input(
            "Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges)
        )

    st.markdown("---")

    if st.button("🔮 Predict Churn Risk", use_container_width=True):
        raw_input = {
            "gender": gender,
            "SeniorCitizen": int(senior),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": str(total_charges),
        }

        model   = load_model(chosen_model)
        X_input = preprocess_single_input(raw_input, feat_loaded, scaler_loaded)
        prob    = model.predict_proba(X_input)[0][1]
        pred    = "Churn Risk" if prob >= 0.5 else "Retained Account"

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns([1.2, 1])

        with col1:
            icon = "🚨" if prob >= 0.5 else "✅"
            st.markdown(
                f"### {icon} {pred}\n\n"
                f"**Confidence Score:** `{prob * 100:.2f}%`  \n"
                f"**Model Used:** `{chosen_model}`"
            )

            if prob >= 0.5:
                st.error("⚠️ High churn risk detected")
                st.markdown("""
                **Recommended Retention Actions:**
                - Offer contract upgrade incentives
                - Apply loyalty discount on monthly charges
                - Assign dedicated customer success manager
                - Promote auto-pay migration
                """)
            else:
                st.success("✅ Low churn risk")
                st.markdown("""
                **Customer Growth Strategy:**
                - Upsell bundled services
                - Regular engagement campaigns
                - Early renewal incentives
                """)

        with col2:
            bar_color = "#ff4b6e" if prob >= 0.5 else "#00c2cb"
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%"},
                    title={"text": "Churn Probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": bar_color},
                        "steps": [
                            {"range": [0,  40], "color": "rgba(0, 194, 203, 0.2)"},
                            {"range": [40, 65], "color": "rgba(255, 215, 0, 0.2)"},
                            {"range": [65, 100], "color": "rgba(255, 75, 110, 0.2)"},
                        ],
                        "threshold": {
                            "value": 50,
                            "line": {"color": "black", "width": 2},
                        },
                    },
                )
            )
            fig_gauge.update_layout(
                height=320,
                margin=dict(t=40, b=10, l=10, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_gauge, use_container_width=True)