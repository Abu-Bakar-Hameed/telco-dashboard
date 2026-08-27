"""
utils.py — Telco Customer Churn Project
Data extraction, preprocessing, model training, and evaluation helpers.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────
def load_raw_data() -> pd.DataFrame:
    """Load raw CSV from disk."""
    return pd.read_csv(DATA_PATH)


def get_data_profile(df: pd.DataFrame) -> dict:
    """Return a summary profile of the raw dataframe."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "churn_distribution": (
            df["Churn"].value_counts().to_dict() if "Churn" in df.columns else {}
        ),
    }


# ─────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
TARGET_COL   = "Churn"
DROP_COLS    = ["customerID"]


def preprocess(df: pd.DataFrame):
    """
    Full preprocessing pipeline.
    Returns (X_train, X_test, y_train, y_test, feature_names, scaler, preprocessed_df).
    """
    df = df.copy()

    # Fix TotalCharges: coerce to numeric, fill NaN with median
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Drop customerID
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Encode target
    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)

    # One-Hot Encode categoricals
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    # Scale numerics
    scaler = StandardScaler()
    df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_names, scaler, df


def preprocess_single_input(raw_input: dict, feature_names: list, scaler) -> np.ndarray:
    """
    Preprocess a single user-provided record for prediction.
    raw_input: dict with column names matching original CSV schema.
    """
    df = pd.DataFrame([raw_input])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    # Align to training feature columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]
    return df.values


# ─────────────────────────────────────────────────────────
# 3. MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────
def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(
            probability=True, random_state=42, class_weight="balanced"
        ),
    }


# ─────────────────────────────────────────────────────────
# 4. TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test) -> dict:
    """Compute all required classification metrics."""
    y_pred = model.predict(X_test)

    # predict_proba may fail on stale pickles from a different sklearn version.
    # Fall back to decision_function (e.g. SVM / old LR) when it does.
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except (AttributeError, TypeError):
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            # Normalise to [0, 1] with a sigmoid so downstream ROC code works
            y_prob = 1 / (1 + np.exp(-scores))
        else:
            # Last resort: use hard predictions as a degenerate probability
            y_prob = y_pred.astype(float)

    cm     = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1":        round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": cm,
        "fpr": fpr, "tpr": tpr,
        "pr_precision": prec_c, "pr_recall": rec_c,
    }


def train_all_models(X_train, X_test, y_train, y_test) -> tuple:
    """Train all models and return (trained_dict, metrics_dict)."""
    models  = get_models()
    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        trained[name] = model
        save_model(model, name)
    return trained, results


def tune_models(X_train, y_train) -> tuple:
    """GridSearchCV tuning for Random Forest. Returns (tuned_models, best_params)."""
    tuned       = {}
    best_params = {}

    # Random Forest
    rf_grid = {
        "n_estimators":      [100, 200],
        "max_depth":         [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    rf_cv = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_grid, cv=3, scoring="f1", n_jobs=1, verbose=0
    )
    rf_cv.fit(X_train, y_train)
    tuned["Random Forest (Tuned)"] = rf_cv.best_estimator_
    best_params["Random Forest"]   = rf_cv.best_params_

    return tuned, best_params


# ─────────────────────────────────────────────────────────
# 5. SERIALIZATION
# ─────────────────────────────────────────────────────────
def save_model(model, name: str):
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(model, os.path.join(MODEL_DIR, f"{safe_name}.pkl"))


def load_model(name: str):
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(MODEL_DIR, f"{safe_name}.pkl")
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        # Stale pickle from an incompatible sklearn version — discard it so
        # the caller falls back to training fresh models.
        return None


def save_scaler(scaler):
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))


def load_scaler():
    path = os.path.join(MODEL_DIR, "scaler.pkl")
    return joblib.load(path) if os.path.exists(path) else None


def save_feature_names(feature_names: list):
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))


def load_feature_names() -> list:
    path = os.path.join(MODEL_DIR, "feature_names.pkl")
    return joblib.load(path) if os.path.exists(path) else []


# ─────────────────────────────────────────────────────────
# 6. PLOTLY VISUALIZATIONS
# ─────────────────────────────────────────────────────────
PALETTE = px.colors.qualitative.Bold


def plot_churn_distribution(df_raw: pd.DataFrame) -> go.Figure:
    counts = df_raw["Churn"].value_counts().reset_index()
    counts.columns = ["Churn", "Count"]
    fig = px.pie(counts, names="Churn", values="Count",
                 title="Churn Distribution",
                 color_discrete_sequence=["#00c2cb", "#ff4b6e"])
    fig.update_traces(textinfo="percent+label", hole=0.4)
    return fig


def plot_tenure_distribution(df_raw: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df_raw, x="tenure", color="Churn",
                       barmode="overlay", nbins=40,
                       title="Tenure Distribution by Churn",
                       color_discrete_map={"Yes": "#ff4b6e", "No": "#00c2cb"})
    fig.update_layout(bargap=0.05)
    return fig


def plot_monthly_charges(df_raw: pd.DataFrame) -> go.Figure:
    fig = px.box(df_raw, x="Churn", y="MonthlyCharges", color="Churn",
                 title="Monthly Charges vs Churn",
                 color_discrete_map={"Yes": "#ff4b6e", "No": "#00c2cb"})
    return fig


def plot_contract_churn(df_raw: pd.DataFrame) -> go.Figure:
    ct = df_raw.groupby(["Contract", "Churn"]).size().reset_index(name="Count")
    fig = px.bar(ct, x="Contract", y="Count", color="Churn",
                 barmode="group", title="Contract Type vs Churn",
                 color_discrete_map={"Yes": "#ff4b6e", "No": "#00c2cb"})
    return fig


def plot_internet_churn(df_raw: pd.DataFrame) -> go.Figure:
    ct = df_raw.groupby(["InternetService", "Churn"]).size().reset_index(name="Count")
    fig = px.bar(ct, x="InternetService", y="Count", color="Churn",
                 barmode="group", title="Internet Service vs Churn",
                 color_discrete_map={"Yes": "#ff4b6e", "No": "#00c2cb"})
    return fig


def plot_payment_churn(df_raw: pd.DataFrame) -> go.Figure:
    ct = df_raw.groupby(["PaymentMethod", "Churn"]).size().reset_index(name="Count")
    fig = px.bar(ct, x="PaymentMethod", y="Count", color="Churn",
                 barmode="group", title="Payment Method vs Churn",
                 color_discrete_map={"Yes": "#ff4b6e", "No": "#00c2cb"})
    fig.update_layout(xaxis_tickangle=-30)
    return fig


def plot_correlation_heatmap(df_raw: pd.DataFrame) -> go.Figure:
    num = df_raw[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]].copy()
    num["TotalCharges"] = pd.to_numeric(num["TotalCharges"], errors="coerce")
    corr = num.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Numeric Feature Correlation Matrix")
    return fig


def plot_senior_citizen(df_raw: pd.DataFrame) -> go.Figure:
    ct = df_raw.groupby(["SeniorCitizen", "Churn"]).size().reset_index(name="Count")
    ct["SeniorCitizen"] = ct["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})
    fig = px.bar(ct, x="SeniorCitizen", y="Count", color="Churn",
                 barmode="group", title="Senior Citizen vs Churn",
                 color_discrete_map={"Yes": "#ff4b6e", "No": "#00c2cb"})
    return fig


# ─────────────────────────────────────────────────────────
# 7. COMPARISON CHARTS
# ─────────────────────────────────────────────────────────
def plot_metrics_bar(results: dict) -> go.Figure:
    """Accuracy vs F1 grouped bar chart."""
    names = list(results.keys())
    acc   = [results[n]["accuracy"] for n in names]
    f1    = [results[n]["f1"]       for n in names]
    fig   = go.Figure(data=[
        go.Bar(name="Accuracy", x=names, y=acc, marker_color="#00c2cb"),
        go.Bar(name="F1 Score", x=names, y=f1,  marker_color="#ff4b6e"),
    ])
    fig.update_layout(
        barmode="group",
        title="Accuracy vs F1 Score — All Models",
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


def plot_roc_curves(results: dict) -> go.Figure:
    """Multi-model ROC curves on one chart."""
    fig = go.Figure()
    for name, m in results.items():
        fig.add_trace(go.Scatter(
            x=m["fpr"], y=m["tpr"], mode="lines",
            name=f"{name} (AUC={m['roc_auc']:.3f})"
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="grey"), name="Random"
    ))
    fig.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return fig


def plot_pr_curves(results: dict) -> go.Figure:
    """Precision-Recall curves."""
    fig = go.Figure()
    for name, m in results.items():
        fig.add_trace(go.Scatter(
            x=m["pr_recall"], y=m["pr_precision"],
            mode="lines", name=name
        ))
    fig.update_layout(
        title="Precision-Recall Curves — All Models",
        xaxis_title="Recall",
        yaxis_title="Precision",
    )
    return fig


def plot_confusion_matrices(results: dict) -> go.Figure:
    """Side-by-side confusion matrices."""
    names = list(results.keys())
    cols  = 3
    rows  = (len(names) + cols - 1) // cols
    fig   = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=names,
        vertical_spacing=0.12,
    )
    for idx, name in enumerate(names):
        r, c  = divmod(idx, cols)
        cm    = results[name]["confusion_matrix"]
        annot = [[str(v) for v in row] for row in cm]
        hm = go.Heatmap(
            z=cm, text=annot, texttemplate="%{text}",
            colorscale="Blues", showscale=False,
            x=["Pred No", "Pred Yes"],
            y=["Actual No", "Actual Yes"],
        )
        fig.add_trace(hm, row=r + 1, col=c + 1)
    fig.update_layout(title="Confusion Matrices — All Models", height=350 * rows)
    return fig


def plot_feature_importance(model, feature_names: list, top_n: int = 20) -> go.Figure:
    """Feature importance for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return go.Figure().add_annotation(
            text="Model has no feature_importances_", showarrow=False
        )
    imp = model.feature_importances_
    df  = pd.DataFrame({"Feature": feature_names, "Importance": imp})
    df  = df.sort_values("Importance", ascending=False).head(top_n)
    fig = px.bar(
        df, x="Importance", y="Feature", orientation="h",
        title=f"Top {top_n} Feature Importances",
        color="Importance", color_continuous_scale="teal",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def build_leaderboard(results: dict) -> pd.DataFrame:
    rows = []
    for name, m in results.items():
        rows.append({
            "Model":     name,
            "Accuracy":  m["accuracy"],
            "Precision": m["precision"],
            "Recall":    m["recall"],
            "F1":        m["f1"],
            "ROC-AUC":   m["roc_auc"],
        })
    df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    df.index += 1
    return df
