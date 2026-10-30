
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ============================================================
# FRAUDSHIELD AI - CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FraudShield AI | Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATHS = [
    BASE_DIR / "data" / "creditcard.csv",
    BASE_DIR / "creditcard.csv",
]

MODEL_PATHS = [
    BASE_DIR / "fraudshield_model.joblib",
    BASE_DIR / "outputs" / "fraudshield_model.joblib",
    BASE_DIR / "retainiq_model.joblib",
]

BG = "#080D19"
PANEL = "#101A2B"
CYAN = "#39E6D0"
RED = "#FF5576"
PURPLE = "#9D8CFF"
WHITE = "#F2F6FF"
MUTED = "#91A0B8"
GREEN = "#42D99A"
AMBER = "#FFC66D"

# ============================================================
# PREMIUM UI
# ============================================================

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background:
            radial-gradient(ellipse at 10% 0%, #14243a 0%, transparent 42%),
            radial-gradient(ellipse at 95% 10%, #191332 0%, transparent 36%),
            {BG};
        color: {WHITE};
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #10192A, #080D19);
        border-right: 1px solid #26344A;
    }}

    [data-testid="stSidebar"] * {{
        color: {WHITE};
    }}

    .block-container {{
        max-width: 1450px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }}

    @keyframes fadeUp {{
        from {{ opacity: 0; transform: translateY(16px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes glow {{
        0%, 100% {{ box-shadow: 0 0 12px #39e6d020; }}
        50% {{ box-shadow: 0 0 28px #39e6d045; }}
    }}

    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: .48; }}
    }}

    .hero {{
        position: relative;
        overflow: hidden;
        padding: 30px;
        border: 1px solid #2C465B;
        border-radius: 24px;
        background: linear-gradient(120deg, #12283A, #15172B 72%, #211631);
        animation: fadeUp .7s ease both, glow 5s ease-in-out infinite;
        margin-bottom: 25px;
    }}

    .hero:after {{
        content: "";
        position: absolute;
        width: 220px;
        height: 220px;
        right: -65px;
        top: -100px;
        border-radius: 50%;
        border: 1px solid #39E6D035;
        box-shadow: 0 0 0 28px #39E6D009, 0 0 0 58px #39E6D007;
        pointer-events: none;
    }}

    .eyebrow {{
        color: {CYAN};
        letter-spacing: 3px;
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
    }}

    .hero h1 {{
        color: {WHITE};
        font-size: clamp(30px, 4vw, 48px);
        line-height: 1.12;
        margin: 12px 0;
        font-weight: 900;
    }}

    .hero p {{
        color: #B7C5D9;
        font-size: 14px;
        max-width: 700px;
        line-height: 1.8;
    }}

    .status-pill {{
        display: inline-block;
        color: {CYAN};
        background: #39E6D012;
        border: 1px solid #39E6D044;
        border-radius: 50px;
        padding: 7px 12px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-top: 8px;
    }}

    .metric-card {{
        background: linear-gradient(145deg, #142137, #0D1524);
        border: 1px solid #293A52;
        border-radius: 18px;
        padding: 20px;
        min-height: 125px;
        animation: fadeUp .6s ease both;
        transition: transform .2s ease, border-color .2s ease;
    }}

    .metric-card:hover {{
        transform: translateY(-4px);
        border-color: {CYAN};
    }}

    .metric-label {{
        color: {MUTED};
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .metric-value {{
        color: {WHITE};
        font-size: clamp(23px, 2.5vw, 32px);
        font-weight: 800;
        margin: 10px 0 5px;
    }}

    .metric-note {{
        color: {MUTED};
        font-size: 11px;
    }}

    .section-label {{
        color: {CYAN};
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 4px;
    }}

    .panel {{
        background: linear-gradient(145deg, #111D30, #0D1422);
        border: 1px solid #29384E;
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 18px;
        animation: fadeUp .6s ease both;
    }}

    .risk-box {{
        padding: 24px;
        border-radius: 18px;
        text-align: center;
        background: #121F30;
        border: 1px solid #31445D;
        margin: 12px 0;
    }}

    .risk-number {{
        font-size: 48px;
        font-weight: 900;
        margin: 10px 0;
    }}

    .live-dot {{
        display: inline-block;
        height: 8px;
        width: 8px;
        background: {GREEN};
        border-radius: 50%;
        margin-right: 7px;
        animation: pulse 1.5s infinite;
    }}

    div[data-testid="stPlotlyChart"] {{
        background: transparent;
        border-radius: 16px;
    }}

    div.stButton > button {{
        border: 1px solid #39E6D080;
        background: linear-gradient(100deg, #123B45, #242147);
        color: #F5FFFF;
        font-weight: 700;
        border-radius: 12px;
        padding: 10px 18px;
        transition: all .2s ease;
    }}

    div.stButton > button:hover {{
        border-color: {CYAN};
        transform: translateY(-2px);
        box-shadow: 0 5px 20px #39E6D025;
    }}

    [data-testid="stMetric"] {{
        background: #111D30;
        border: 1px solid #293A52;
        border-radius: 14px;
        padding: 14px;
    }}

    h1, h2, h3 {{
        color: {WHITE};
    }}

    .footer {{
        text-align: center;
        color: #73829A;
        border-top: 1px solid #243147;
        padding-top: 20px;
        margin-top: 35px;
        font-size: 10px;
        letter-spacing: 1px;
        line-height: 2;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset():
    for path in DATA_PATHS:
        if path.exists():
            try:
                return pd.read_csv(path), str(path), None
            except Exception as exc:
                return None, str(path), str(exc)

    return None, None, None


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def find_estimator(saved, depth=0):
    """Find a fitted estimator inside a saved model object."""
    if depth > 4:
        return None

    if hasattr(saved, "predict"):
        return saved

    if isinstance(saved, dict):
        preferred_keys = [
            "model",
            "best_model",
            "pipeline",
            "classifier",
            "best_estimator",
            "estimator",
            "trained_model",
            "random_forest",
            "rf_model",
        ]

        for key in preferred_keys:
            if key in saved:
                candidate = find_estimator(saved[key], depth + 1)
                if candidate is not None:
                    return candidate

        for value in saved.values():
            candidate = find_estimator(value, depth + 1)
            if candidate is not None:
                return candidate

    return None


@st.cache_resource(show_spinner=False)
def load_model():
    errors = []

    for path in MODEL_PATHS:
        if path.exists():
            try:
                saved = joblib.load(path)
                estimator = find_estimator(saved)

                if estimator is not None:
                    return estimator, str(path), None

                if isinstance(saved, dict):
                    detail = (
                        "The saved file contains a dictionary, but no "
                        "predict-capable model was found. Keys: "
                        + ", ".join(map(str, saved.keys()))
                    )
                else:
                    detail = (
                        "Unsupported model object: "
                        + type(saved).__name__
                    )

                errors.append(f"{path.name}: {detail}")

            except Exception as exc:
                errors.append(f"{path.name}: {exc}")

    if errors:
        return None, None, "\n".join(errors)

    return None, None, None


df, dataset_path, dataset_error = load_dataset()
model, model_path, model_error = load_model()

# ============================================================
# GENERAL HELPERS
# ============================================================

def money(value):
    return f"${value:,.2f}"


def percent(value):
    return f"{value:.2f}%"


def style_chart(fig, height=340):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=WHITE, size=11),
        margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=WHITE),
        ),
        hoverlabel=dict(
            bgcolor="#172338",
            font_color=WHITE,
        ),
    )

    fig.update_xaxes(
        gridcolor="#263449",
        zerolinecolor="#263449",
    )
    fig.update_yaxes(
        gridcolor="#263449",
        zerolinecolor="#263449",
    )

    return fig


def card(label, value, note, accent=CYAN):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{accent}">
                {value}
            </div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_fraud_probability(features):
    """Return estimated probability of class 1 (fraud)."""
    if model is None:
        raise RuntimeError("No trained model is loaded.")

    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        missing = [c for c in expected if c not in features.columns]

        if missing:
            raise ValueError(
                "The model requires additional feature columns: "
                + ", ".join(missing[:10])
            )

        features = features[expected]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)

        classes = list(getattr(model, "classes_", [0, 1]))

        if hasattr(model, "named_steps"):
            final_estimator = list(model.named_steps.values())[-1]
            classes = list(
                getattr(final_estimator, "classes_", classes)
            )

        if 1 not in classes:
            raise RuntimeError("Fraud class 1 was not found in the model.")

        return float(probabilities[0][classes.index(1)])

    if hasattr(model, "decision_function"):
        score = float(model.decision_function(features)[0])
        return float(
            1 / (1 + np.exp(-np.clip(score, -50, 50)))
        )

    prediction = int(model.predict(features)[0])
    return float(prediction)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:12px 0 25px">
            <div style="font-size:34px">🛡️</div>
            <div style="font-size:23px;font-weight:900">
                FraudShield <span style="color:{CYAN}">AI</span>
            </div>
            <div class="eyebrow"
                 style="font-size:8px;letter-spacing:2px">
                FINANCIAL RISK INTELLIGENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='section-label'>WORKSPACE</div>",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Command Center",
            "Transaction Scanner",
            "Model Intelligence",
            "About FraudShield",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if model is not None:
        st.success("AI model loaded")
    else:
        st.warning("AI model unavailable")

    if df is not None:
        st.caption(f"{len(df):,} dataset records loaded")
    else:
        st.caption("Dataset not found")

    st.markdown(
        f"""
        <div class="panel" style="padding:14px;margin-top:22px">
            <div class="section-label">SYSTEM STATUS</div>
            <p style="font-size:12px;margin-bottom:4px">
                <span class="live-dot"></span>Workspace active
            </p>
            <div style="font-size:10px;color:{MUTED}">
                Local educational prototype<br>
                Human review recommended
            </div>
        </div>
        <div style="font-size:9px;color:{MUTED};line-height:2">
            FRAUDSHIELD AI<br>PORTFOLIO EDITION
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# HERO
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <div class="eyebrow">INTELLIGENT FINANCIAL PROTECTION</div>
        <h1>See the risk.<br>
        <span style="color:{CYAN}">Stop the fraud.</span></h1>
        <p>
            An AI-powered transaction intelligence workspace for
            exploring suspicious financial activity, assessing
            transaction risk, and understanding machine-learning
            model performance.
        </p>
        <span class="status-pill">✦ MACHINE LEARNING</span>
        &nbsp;
        <span class="status-pill">◉ RISK ANALYTICS</span>
        &nbsp;
        <span class="status-pill">⌁ INTERACTIVE DASHBOARD</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA VALIDATION
# ============================================================

if df is None:
    st.error(
        "Dataset not found. Place creditcard.csv inside the "
        "data folder or the main project folder."
    )
    st.info("Expected path: FraudShield-AI/data/creditcard.csv")

    if dataset_error:
        st.code(dataset_error)

    st.stop()

if "Class" not in df.columns:
    st.error(
        "The dataset must contain a 'Class' column: "
        "0 = legitimate, 1 = fraud."
    )
    st.stop()

total_transactions = len(df)
fraud_count = int((df["Class"] == 1).sum())
legitimate_count = int((df["Class"] == 0).sum())

historical_rate = (
    fraud_count / total_transactions * 100
    if total_transactions else 0
)

# ============================================================
# PAGE 1: COMMAND CENTER
# ============================================================

if page == "Command Center":

    st.markdown(
        "<div class='section-label'>OVERVIEW / 01</div>",
        unsafe_allow_html=True,
    )
    st.header("Fraud Command Center")
    st.caption(
        "Explore historical transaction patterns in the loaded dataset. "
        "These are dataset statistics, not live banking activity."
    )

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        card(
            "Transactions",
            f"{total_transactions:,}",
            "Total dataset records",
            WHITE,
        )

    with k2:
        card(
            "Fraud Cases",
            f"{fraud_count:,}",
            "Records labelled fraud",
            RED,
        )

    with k3:
        card(
            "Legitimate",
            f"{legitimate_count:,}",
            "Records labelled legitimate",
            GREEN,
        )

    with k4:
        card(
            "Historical Fraud Rate",
            percent(historical_rate),
            "Share of labelled fraud records",
            AMBER,
        )

    st.write("")
    left, right = st.columns([1.05, 1])

    with left:
        st.markdown("### Transaction composition")

        pie_fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Legitimate", "Fraud"],
                    values=[legitimate_count, fraud_count],
                    hole=0.72,
                    sort=False,
                    marker=dict(
                        colors=[CYAN, RED],
                        line=dict(color=BG, width=4),
                    ),
                    textinfo="percent",
                    hovertemplate=(
                        "%{label}<br>"
                        "Transactions: %{value:,}<br>"
                        "Share: %{percent}<extra></extra>"
                    ),
                )
            ]
        )

        pie_fig.update_layout(
            annotations=[
                dict(
                    text=f"<b>{total_transactions:,}</b><br>records",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=18, color=WHITE),
                )
            ],
            showlegend=True,
        )

        st.plotly_chart(
            style_chart(pie_fig, 350),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with right:
        st.markdown("### Fraud vs legitimate")

        bar_data = pd.DataFrame({
            "Transaction Type": ["Legitimate", "Fraud"],
            "Number of Records": [legitimate_count, fraud_count],
        })

        bar_fig = px.bar(
            bar_data,
            x="Transaction Type",
            y="Number of Records",
            color="Transaction Type",
            color_discrete_map={
                "Legitimate": CYAN,
                "Fraud": RED,
            },
            text="Number of Records",
        )

        bar_fig.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
            cliponaxis=False,
        )

        st.plotly_chart(
            style_chart(bar_fig, 350),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown("### Transaction amount intelligence")

    if "Amount" in df.columns:
        amount_data = df[["Amount", "Class"]].copy()
        amount_data["Transaction Type"] = amount_data["Class"].map({
            0: "Legitimate",
            1: "Fraud",
        })
        amount_data = amount_data.dropna(subset=["Amount"])

        amount_fig = px.histogram(
            amount_data,
            x="Amount",
            color="Transaction Type",
            nbins=45,
            barmode="overlay",
            opacity=0.8,
            color_discrete_map={
                "Legitimate": CYAN,
                "Fraud": RED,
            },
            labels={"Amount": "Transaction amount"},
        )

        st.plotly_chart(
            style_chart(amount_fig, 360),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.caption(
            "Amounts use the units in the source dataset. "
            "This chart is descriptive and does not establish causation."
        )
    else:
        st.info("The dataset does not contain an Amount column.")

    st.markdown("### Recent dataset records")

    preview = df.tail(8).copy()
    preview["Transaction Type"] = preview["Class"].map({
        0: "Legitimate",
        1: "Fraud",
    })

    display_columns = [
        col for col in ["Time", "Amount", "Transaction Type", "Class"]
        if col in preview.columns
    ]

    st.dataframe(
        preview[display_columns].iloc[::-1],
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# PAGE 2: TRANSACTION SCANNER
# ============================================================

elif page == "Transaction Scanner":

    st.markdown(
        "<div class='section-label'>RISK ANALYSIS / 02</div>",
        unsafe_allow_html=True,
    )
    st.header("Transaction Risk Scanner")
    st.caption(
        "Select a transaction record and estimate its fraud risk "
        "using your saved machine-learning model."
    )

    if model is None:
        st.error(
            "The saved model could not be loaded. Check that "
            "fraudshield_model.joblib exists and contains a trained model."
        )

        if model_error:
            st.code(model_error)

        st.info(
            "If the model is stored in a dictionary, inspect the keys "
            "shown above and check train_model.py."
        )
        st.stop()

    feature_columns = [c for c in df.columns if c != "Class"]

    if not feature_columns:
        st.error("No model input features were found.")
        st.stop()

    st.markdown(
        """
        <div class="panel">
            <div class="section-label">SCAN CONFIGURATION</div>
            Select a transaction record from the dataset.
            The selected record's feature values will be sent to the model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    max_index = len(df) - 1

    with st.form("transaction_scanner_form"):
        selected_index = st.number_input(
            "Dataset transaction row",
            min_value=0,
            max_value=max_index,
            value=0,
            step=1,
            help="Choose a row index from the loaded dataset.",
        )

        show_features = st.checkbox(
            "Preview model input features",
            value=False,
        )

        submitted = st.form_submit_button(
            "⚡ Analyse Transaction Risk",
            use_container_width=True,
        )

    if show_features:
        st.dataframe(
            df.iloc[[int(selected_index)]][feature_columns],
            use_container_width=True,
        )

    if submitted:
        try:
            selected_row = df.iloc[[int(selected_index)]]
            features = selected_row[feature_columns].copy()

            features = features.apply(pd.to_numeric, errors="coerce")

            if features.isna().any().any():
                st.error(
                    "This transaction contains missing or invalid feature "
                    "values. Choose another dataset row."
                )
                st.stop()

            with st.spinner("Analysing transaction signals..."):
                time.sleep(0.3)
                risk_probability = get_fraud_probability(features)

            risk_probability = float(np.clip(risk_probability, 0, 1))
            risk_pct = risk_probability * 100

            if risk_pct >= 70:
                risk_name = "HIGH RISK"
                risk_color = RED
                risk_message = (
                    "This record has a high estimated fraud probability. "
                    "Additional verification may be appropriate."
                )
            elif risk_pct >= 30:
                risk_name = "MODERATE RISK"
                risk_color = AMBER
                risk_message = (
                    "This record falls in the moderate-risk band. "
                    "Further review may be appropriate."
                )
            else:
                risk_name = "LOW RISK"
                risk_color = GREEN
                risk_message = (
                    "This record has a lower estimated fraud probability. "
                    "A low score does not guarantee safety."
                )

            st.markdown(
                f"""
                <div class="risk-box">
                    <div class="section-label">MODEL RISK ASSESSMENT</div>
                    <div class="risk-number" style="color:{risk_color}">
                        {risk_pct:.2f}%
                    </div>
                    <h2 style="color:{risk_color};margin:0">
                        {risk_name}
                    </h2>
                    <p style="color:{MUTED};margin-top:12px">
                        {risk_message}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(risk_probability)

            a, b, c = st.columns(3)

            with a:
                if "Amount" in selected_row.columns:
                    st.metric(
                        "Transaction Amount",
                        money(float(selected_row["Amount"].iloc[0])),
                    )
                else:
                    st.metric("Transaction Amount", "Not available")

            with b:
                actual_label = int(selected_row["Class"].iloc[0])
                st.metric(
                    "Dataset Label",
                    "Fraud" if actual_label == 1 else "Legitimate",
                )

            with c:
                st.metric("Dataset Row", f"#{int(selected_index)}")

            with st.expander("View complete transaction features"):
                st.dataframe(features, use_container_width=True)

            st.warning(
                "Educational prototype only. This score is a model estimate, "
                "not a confirmed fraud finding. Do not use it to make real "
                "financial decisions without proper validation and human review."
            )

        except Exception as exc:
            st.error("The model could not score this transaction.")
            st.code(f"{type(exc).__name__}: {exc}")
            st.info(
                "The saved model and dashboard must use the same input "
                "features and preprocessing pipeline. If the error mentions "
                "feature names or column counts, check train_model.py."
            )

# ============================================================
# PAGE 3: MODEL INTELLIGENCE
# ============================================================

elif page == "Model Intelligence":

    st.markdown(
        "<div class='section-label'>MODEL PERFORMANCE / 03</div>",
        unsafe_allow_html=True,
    )
    st.header("Model Intelligence")
    st.caption(
        "Inspect classification metrics and the confusion matrix "
        "using a held-out evaluation split."
    )

    if model is None:
        st.warning(
            "No saved model is available. Train and save the model first."
        )

        if model_error:
            st.code(model_error)

        st.stop()

    st.success("Saved model loaded successfully.")

    if model_path:
        st.caption(f"Model file: {Path(model_path).name}")

    if st.button("Run evaluation on a held-out split"):
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                roc_auc_score,
                confusion_matrix,
                classification_report,
            )

            X = df.drop(columns=["Class"]).copy()
            y = df["Class"].copy()

            X = X.apply(pd.to_numeric, errors="coerce")
            valid_rows = X.notna().all(axis=1)

            X = X.loc[valid_rows]
            y = y.loc[valid_rows]

            if y.nunique() < 2:
                st.error(
                    "Evaluation requires both legitimate and fraud examples."
                )
                st.stop()

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )

            if hasattr(model, "feature_names_in_"):
                expected = list(model.feature_names_in_)
                missing = [c for c in expected if c not in X_test.columns]

                if missing:
                    raise ValueError(
                        "Missing model features: " + ", ".join(missing[:10])
                    )

                X_test = X_test[expected]

            with st.spinner("Evaluating model..."):
                predictions = model.predict(X_test)

                accuracy = accuracy_score(y_test, predictions)
                precision = precision_score(
                    y_test, predictions, zero_division=0
                )
                recall = recall_score(
                    y_test, predictions, zero_division=0
                )
                f1 = f1_score(
                    y_test, predictions, zero_division=0
                )

                auc = None

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_test)
                    classes = list(getattr(model, "classes_", [0, 1]))

                    if hasattr(model, "named_steps"):
                        estimator = list(model.named_steps.values())[-1]
                        classes = list(
                            getattr(estimator, "classes_", classes)
                        )

                    if 1 in classes:
                        fraud_index = classes.index(1)
                        auc = roc_auc_score(
                            y_test, probs[:, fraud_index]
                        )

            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.metric("Accuracy", percent(accuracy * 100))

            with m2:
                st.metric("Precision", percent(precision * 100))

            with m3:
                st.metric("Recall", percent(recall * 100))

            with m4:
                st.metric("F1-Score", percent(f1 * 100))

            if auc is not None:
                st.metric("ROC-AUC", f"{auc:.4f}")

            matrix = confusion_matrix(
                y_test, predictions, labels=[0, 1]
            )

            cm_fig = go.Figure(
                data=go.Heatmap(
                    z=matrix,
                    x=["Predicted Legitimate", "Predicted Fraud"],
                    y=["Actual Legitimate", "Actual Fraud"],
                    colorscale=[
                        [0, "#132239"],
                        [0.5, "#225C68"],
                        [1, "#39E6D0"],
                    ],
                    showscale=False,
                    text=matrix,
                    texttemplate="%{text:,}",
                    hovertemplate=(
                        "Actual: %{y}<br>"
                        "Prediction: %{x}<br>"
                        "Count: %{z:,}<extra></extra>"
                    ),
                )
            )

            cm_fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted class",
                yaxis_title="Actual class",
            )

            st.plotly_chart(
                style_chart(cm_fig, 400),
                use_container_width=True,
                config={"displayModeBar": False},
            )

            st.markdown("### Classification report")

            report = classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["Legitimate", "Fraud"],
                output_dict=True,
                zero_division=0,
            )

            report_df = pd.DataFrame(report).transpose()

            st.dataframe(
                report_df.round(4),
                use_container_width=True,
            )

            st.info(
                "Important: this evaluation is a reliable held-out estimate "
                "only if these test records were not used during training "
                "or model selection. If the saved model was trained on the "
                "full dataset, retrain it using a proper train/test split."
            )

        except Exception as exc:
            st.error("Evaluation failed.")
            st.code(f"{type(exc).__name__}: {exc}")
            st.caption(
                "Check that train_model.py uses the same feature columns "
                "and preprocessing as this dashboard."
            )

    st.markdown("### What do these metrics mean?")

    metric_info = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-score", "ROC-AUC"],
        "Meaning": [
            "How many flagged transactions were actually fraud.",
            "How many actual fraud cases the model detected.",
            "Balance between precision and recall.",
            "Ability to rank fraud cases above legitimate cases.",
        ],
        "Why it matters": [
            "Helps understand false alarms.",
            "Important because missed fraud can be costly.",
            "Useful when comparing precision and recall together.",
            "Measures ranking performance across thresholds.",
        ],
    })

    st.dataframe(
        metric_info,
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# PAGE 4: ABOUT FRAUDSHIELD
# ============================================================

elif page == "About FraudShield":

    st.markdown(
        "<div class='section-label'>PROJECT PROFILE / 04</div>",
        unsafe_allow_html=True,
    )
    st.header("Inside FraudShield AI")

    st.markdown(
        """
        <div class="panel">
            <div class="eyebrow">PROJECT MISSION</div>
            <h2>Make suspicious activity easier to investigate.</h2>
            FraudShield AI is an educational machine-learning project
            that explores transaction classification, imbalanced datasets,
            model evaluation, and interactive risk visualisation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Project pipeline")

    pipeline = [
        ("01", "DATA", "Load and inspect transaction records."),
        ("02", "PREPROCESSING", "Prepare features for model input."),
        ("03", "MACHINE LEARNING", "Train and compare classification models."),
        ("04", "EVALUATION", "Measure precision, recall, F1-score and ROC-AUC."),
        ("05", "RISK INTERFACE", "Display estimated transaction risk interactively."),
    ]

    for number, title, description in pipeline:
        st.markdown(
            f"""
            <div class="panel" style="padding:16px;margin-bottom:10px">
                <span style="color:{CYAN};font-weight:900;font-size:20px">
                    {number}
                </span>
                &nbsp;&nbsp;
                <span style="font-weight:800">{title}</span>
                <div style="color:{MUTED};font-size:12px;margin-top:8px">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Technology stack")

    t1, t2, t3 = st.columns(3)

    with t1:
        st.markdown(
            """
            <div class="panel">
                <h3>🐍 Python</h3>
                <p style="color:#91A0B8;font-size:12px">
                    Pandas · NumPy · Scikit-learn
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t2:
        st.markdown(
            """
            <div class="panel">
                <h3>📊 Analytics</h3>
                <p style="color:#91A0B8;font-size:12px">
                    Plotly · Metrics · Confusion matrix
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t3:
        st.markdown(
            """
            <div class="panel">
                <h3>✨ Interface</h3>
                <p style="color:#91A0B8;font-size:12px">
                    Streamlit · CSS · Interactive charts
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Limitations")

    st.warning(
        "This is not a production fraud detection system. The public dataset "
        "may not represent current financial transactions. A production "
        "system would require stronger validation, monitoring, privacy "
        "controls, explainability, threshold calibration, and human review."
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="footer">
        FRAUDSHIELD AI · FINANCIAL RISK INTELLIGENCE<br>
        EDUCATIONAL MACHINE-LEARNING PROTOTYPE · HUMAN REVIEW REQUIRED<br>
        <span style="color:{CYAN}">
            BUILT FOR LEARNING · DESIGNED FOR INSIGHT
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
