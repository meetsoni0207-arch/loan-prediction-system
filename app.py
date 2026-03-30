from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "best_model.joblib"
METADATA_PATH = PROJECT_ROOT / "artifacts" / "models" / "model_metadata.joblib"
APP_THRESHOLD = 0.80


def probability_to_score(probability: float) -> int:
    return int(max(300, min(850, round(850 - probability * 550))))


def get_risk_band(probability: float) -> dict[str, str]:
    if probability >= 0.80:
        return {
            "label": "High Risk",
            "color": "#b42318",
            "surface": "#fff1f0",
            "range": "80% to 100%",
        }
    if probability >= 0.60:
        return {
            "label": "Review Required",
            "color": "#b54708",
            "surface": "#fff7e8",
            "range": "60% to below 80%",
        }
    if probability >= 0.30:
        return {
            "label": "Intermediate Risk",
            "color": "#b69208",
            "surface": "#fffbea",
            "range": "30% to below 60%",
        }
    return {
        "label": "Low Risk",
        "color": "#027a48",
        "surface": "#ecfdf3",
        "range": "0% to below 30%",
    }


def get_risk_profile(probability: float, threshold: float) -> dict[str, str]:
    band = get_risk_band(probability)
    if probability >= threshold:
        description = "This case crosses the model threshold and should go through manual review or tighter pricing."
    else:
        description = "This applicant remains below the model threshold and looks comparatively safer."
    return {
        "label": band["label"],
        "color": band["color"],
        "surface": band["surface"],
        "description": f"{description} Risk band: {band['range']}.",
    }


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, metadata


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(18, 52, 77, 0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(8, 145, 178, 0.10), transparent 24%),
                linear-gradient(180deg, #f7fafb 0%, #eef4f6 100%);
            color: #102a43;
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }

        .hero {
            padding: 1.6rem 1.8rem;
            border-radius: 26px;
            background: linear-gradient(135deg, #102a43 0%, #156082 100%);
            color: #f8fbfc;
            box-shadow: 0 22px 48px rgba(16, 42, 67, 0.16);
            margin-bottom: 1.25rem;
        }

        .hero h1 {
            margin: 0 0 0.45rem 0;
            font-size: 2.2rem;
            letter-spacing: -0.02em;
            color: #f8fbfc !important;
        }

        .hero p {
            margin: 0;
            max-width: 820px;
            color: rgba(248, 251, 252, 0.9);
        }

        .hero strong,
        .hero-chip {
            color: #f8fbfc !important;
        }

        .hero-chip-row {
            display: flex;
            gap: 0.65rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }

        .hero-chip {
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 999px;
            padding: 0.45rem 0.85rem;
            font-size: 0.88rem;
        }

        .panel {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(247, 250, 252, 0.96));
            border: 1px solid rgba(16, 42, 67, 0.10);
            border-radius: 24px;
            padding: 1.15rem 1.2rem;
            box-shadow: 0 12px 30px rgba(16, 42, 67, 0.08);
            margin-bottom: 1rem;
        }

        .panel-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #102a43;
            margin-bottom: 0.85rem;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid rgba(16, 42, 67, 0.08);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 24px rgba(16, 42, 67, 0.08);
        }

        .metric-label {
            color: #627d98;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .metric-value {
            color: #102a43;
            font-size: 2rem;
            font-weight: 700;
            margin-top: 0.35rem;
            line-height: 1.1;
        }

        .metric-copy {
            color: #627d98;
            font-size: 0.9rem;
            margin-top: 0.4rem;
        }

        .risk-banner {
            border-radius: 20px;
            padding: 1rem 1.05rem;
            border: 1px solid rgba(16, 42, 67, 0.08);
            margin-bottom: 1rem;
        }

        .risk-title {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .chart-caption {
            color: #627d98;
            font-size: 0.9rem;
            margin-top: 0.55rem;
        }

        .panel-title,
        .metric-label,
        .metric-value,
        .metric-copy,
        .risk-title,
        .section-card,
        .section-card *,
        .panel,
        .panel *,
        .stSelectbox label,
        .stSlider label,
        .stNumberInput label,
        label {
            color: #102a43 !important;
        }

        [data-testid="stWidgetLabel"] p,
        [data-testid="stForm"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stForm"] [data-testid="stMarkdownContainer"] li,
        [data-testid="stForm"] [data-testid="stMarkdownContainer"] span {
            color: #102a43 !important;
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-baseweb="input"] input,
        div[data-baseweb="select"] input,
        div[data-baseweb="base-input"] input {
            color: #102a43 !important;
            -webkit-text-fill-color: #102a43 !important;
            opacity: 1 !important;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] > div,
        [data-testid="stNumberInput"] div[data-baseweb="input"] > div {
            background: #ffffff !important;
            border-color: rgba(16, 42, 67, 0.14) !important;
        }

        [data-testid="stNumberInput"] input {
            background: #ffffff !important;
            color: #102a43 !important;
            -webkit-text-fill-color: #102a43 !important;
        }

        [data-testid="stNumberInput"] button,
        [data-testid="stSelectbox"] svg {
            color: #102a43 !important;
        }

        [data-testid="stSelectbox"] [data-baseweb="select"] {
            background: #ffffff !important;
            border-radius: 14px !important;
        }

        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            min-height: 52px;
        }

        [data-testid="stSelectbox"] [data-baseweb="select"] > div * {
            color: #102a43 !important;
            fill: #102a43 !important;
            opacity: 1 !important;
        }

        [data-testid="stSelectbox"] [class*="singleValue"],
        [data-testid="stSelectbox"] [class*="placeholder"],
        [data-testid="stSelectbox"] [class*="valueContainer"],
        [data-testid="stSelectbox"] [class*="ValueContainer"],
        [data-testid="stSelectbox"] [class*="control"] {
            color: #102a43 !important;
            -webkit-text-fill-color: #102a43 !important;
            opacity: 1 !important;
        }

        [data-testid="stTable"] table,
        [data-testid="stTable"] th,
        [data-testid="stTable"] td,
        [data-testid="stDataFrame"] *,
        table, thead, tbody, tr, th, td {
            color: #102a43 !important;
        }

        [data-testid="stTable"] table,
        [data-testid="stDataFrame"] {
            background: #ffffff !important;
            border-radius: 16px !important;
            overflow: hidden;
            border: 1px solid rgba(16, 42, 67, 0.08);
        }

        [data-testid="stTable"] th,
        [data-testid="stDataFrame"] thead th {
            background: #eaf2f5 !important;
            font-weight: 700 !important;
        }

        [data-testid="stForm"] {
            background: transparent;
            border: none;
        }

        [data-testid="stForm"] > div {
            border: none !important;
            padding: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []


def build_input_frame() -> tuple[pd.DataFrame, bool]:
    left_spacer, center, right_spacer = st.columns([0.08, 0.84, 0.08])
    with center:
        st.markdown('<div class="panel-title">Underwriting Inputs</div>', unsafe_allow_html=True)
        with st.form("application_form", clear_on_submit=False):
            fin1, fin2, fin3 = st.columns(3)
            with fin1:
                income = st.number_input("Annual Income", min_value=1000, value=60000, step=1000)
                credit_score = st.slider("Credit Score", 300, 850, 680)
            with fin2:
                loan_amount = st.number_input("Requested Loan Amount", min_value=1000, value=25000, step=1000)
                interest_rate = st.slider("Interest Rate", 1.0, 35.0, 10.0)
            with fin3:
                dti_ratio = st.slider("DTI Ratio", 0.0, 1.0, 0.25)
                num_credit_lines = st.slider("Number of Credit Lines", 1, 15, 4)

            borrower1, borrower2, borrower3 = st.columns(3)
            with borrower1:
                age = st.slider("Age", 18, 75, 35)
                months_employed = st.slider("Months Employed", 0, 240, 48)
            with borrower2:
                education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
                employment_type = st.selectbox(
                    "Employment Type",
                    ["Full-time", "Part-time", "Self-employed", "Unemployed"],
                )
            with borrower3:
                marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
                loan_term = st.selectbox("Loan Term", [12, 24, 36, 48, 60, 72])

            app1, app2, app3 = st.columns(3)
            with app1:
                has_mortgage = st.selectbox("Has Mortgage", ["No", "Yes"])
            with app2:
                has_dependents = st.selectbox("Has Dependents", ["No", "Yes"])
            with app3:
                has_cosigner = st.selectbox("Has Co-Signer", ["No", "Yes"])

            loan_purpose = st.selectbox("Loan Purpose", ["Home", "Auto", "Education", "Business", "Other"])
            submitted = st.form_submit_button("Evaluate Applicant", use_container_width=True, type="primary")

    frame = pd.DataFrame(
        [
            {
                "Age": age,
                "Income": income,
                "LoanAmount": loan_amount,
                "CreditScore": credit_score,
                "MonthsEmployed": months_employed,
                "NumCreditLines": num_credit_lines,
                "InterestRate": interest_rate,
                "LoanTerm": loan_term,
                "DTIRatio": dti_ratio,
                "Education": education,
                "EmploymentType": employment_type,
                "MaritalStatus": marital_status,
                "HasMortgage": has_mortgage,
                "HasDependents": has_dependents,
                "LoanPurpose": loan_purpose,
                "HasCoSigner": has_cosigner,
            }
        ]
    )

    frame["Loan_to_Income"] = frame["LoanAmount"] / frame["Income"].clip(lower=1)
    frame["Credit_per_Line"] = frame["CreditScore"] / frame["NumCreditLines"].clip(lower=1)
    frame["Income_per_Dependent"] = frame["Income"] / (
        1 + frame["HasDependents"].map({"Yes": 1, "No": 0}).fillna(0)
    )
    frame["Rate_x_DTI"] = frame["InterestRate"] * frame["DTIRatio"]
    return frame, submitted


def render_hero(metadata: dict) -> None:
    model_name = metadata.get("best_model_name") or metadata.get("scoring", {}).get("best_model_name", "Best Model")
    st.markdown(
        f"""
        <section class="hero">
            <h1>Loan Default Risk Dashboard</h1>
            <p>
                Applicant evaluation workspace powered by the trained machine learning model.
                The current production model is <strong>{model_name}</strong>, selected for maximum validation accuracy.
            </p>
            <div class="hero-chip-row">
                <div class="hero-chip">Accuracy-first model selection</div>
                <div class="hero-chip">Decision threshold: {APP_THRESHOLD:.2f}</div>
                <div class="hero-chip">Credit risk score range: 300 to 850</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, subtle: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-copy">{subtle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_default_label(probability: float, threshold: float) -> tuple[int, str]:
    predicted_default = 1 if probability >= threshold else 0
    meaning = "Default" if predicted_default == 1 else "No Default"
    return predicted_default, meaning


def get_loan_safety_message(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "Not safe to give the loan under the current model threshold."
    return "Safer to give the loan under the current model threshold."


def render_snapshot(applicant: pd.DataFrame) -> None:
    snapshot = pd.DataFrame(
        {
            "Field": [
                "Annual Income",
                "Requested Loan Amount",
                "Credit Score",
                "Interest Rate",
                "DTI Ratio",
                "Employment Length",
                "Education",
                "Employment Type",
                "Marital Status",
                "Loan Purpose",
                "Loan to Income",
                "Credit per Line",
            ],
            "Value": [
                format_currency(float(applicant.at[0, "Income"])),
                format_currency(float(applicant.at[0, "LoanAmount"])),
                f"{int(applicant.at[0, 'CreditScore'])}",
                f"{float(applicant.at[0, 'InterestRate']):.2f}%",
                f"{float(applicant.at[0, 'DTIRatio']):.2f}",
                f"{int(applicant.at[0, 'MonthsEmployed'])} months",
                applicant.at[0, "Education"],
                applicant.at[0, "EmploymentType"],
                applicant.at[0, "MaritalStatus"],
                applicant.at[0, "LoanPurpose"],
                f"{float(applicant.at[0, 'Loan_to_Income']):.2f}",
                f"{float(applicant.at[0, 'Credit_per_Line']):.2f}",
            ],
        }
    )
    st.dataframe(snapshot, use_container_width=True, hide_index=True)


def append_history(applicant: pd.DataFrame, probability: float, score: int, decision: str) -> None:
    row = {
        "Run": len(st.session_state.prediction_history) + 1,
        "Income": float(applicant.at[0, "Income"]),
        "LoanAmount": float(applicant.at[0, "LoanAmount"]),
        "CreditScore": int(applicant.at[0, "CreditScore"]),
        "PD": round(probability, 4),
        "RiskScore": score,
        "Decision": decision,
    }
    st.session_state.prediction_history.insert(0, row)
    st.session_state.prediction_history = st.session_state.prediction_history[:12]


def render_prediction_charts(probability: float, threshold: float) -> None:
    comparison = pd.DataFrame(
        {
            "Metric": ["Default Probability", "Decision Threshold"],
            "Value": [probability, threshold],
        }
    ).set_index("Metric")
    st.bar_chart(comparison, height=260)
    st.caption("This chart compares the applicant's predicted probability of default against the model threshold.")


def render_risk_ranges(threshold: float) -> None:
    ranges = pd.DataFrame(
        [
            {"Risk Band": "Low Risk", "Probability Range": "0% to below 30%", "Typical View": "Generally safer profile"},
            {"Risk Band": "Intermediate Risk", "Probability Range": "30% to below 60%", "Typical View": "Borderline, review context"},
            {"Risk Band": "Review Required", "Probability Range": "60% to below 80%", "Typical View": "Needs stricter review"},
            {"Risk Band": "High Risk", "Probability Range": "80% to 100%", "Typical View": "Very risky lending case"},
        ]
    )
    st.dataframe(ranges, use_container_width=True, hide_index=True)
    st.caption(
        f"These are business-facing probability bands. The current model decision threshold is {threshold:.2f} ({threshold:.0%})."
    )


def render_default_meaning() -> None:
    default_mapping = pd.DataFrame(
        [
            {"Default": 0, "Meaning": "No Default"},
            {"Default": 1, "Meaning": "Default"},
        ]
    )
    st.dataframe(default_mapping, use_container_width=True, hide_index=True)
    st.caption("Target label used by the model: `0` means the loan is not expected to default, `1` means default.")


def render_example_profiles() -> None:
    examples = pd.DataFrame(
        [
            {
                "Example": "Low Risk",
                "Income": "$95,000",
                "Loan Amount": "$18,000",
                "Credit Score": "760",
                "Interest Rate": "7%",
                "DTI": "0.18",
                "Employment": "84 months, Full-time",
                "Typical Signal": "Higher income, lower loan burden, stronger credit history",
            },
            {
                "Example": "Intermediate Risk",
                "Income": "$58,000",
                "Loan Amount": "$42,000",
                "Credit Score": "660",
                "Interest Rate": "13%",
                "DTI": "0.36",
                "Employment": "30 months, Part-time",
                "Typical Signal": "Moderate leverage and average credit quality",
            },
            {
                "Example": "High Risk",
                "Income": "$28,000",
                "Loan Amount": "$85,000",
                "Credit Score": "520",
                "Interest Rate": "24%",
                "DTI": "0.72",
                "Employment": "6 months, Unemployed/unstable",
                "Typical Signal": "High loan-to-income pressure, weaker credit, expensive borrowing",
            },
        ]
    )
    st.dataframe(examples, use_container_width=True, hide_index=True)
    st.caption("These are illustrative examples to help users understand what input patterns tend to look safer or riskier.")


def render_history() -> None:
    history = pd.DataFrame(st.session_state.prediction_history)
    if history.empty:
        st.info("Prediction history will appear here after you evaluate an applicant.")
        return

    history_display = history.copy()
    history_display["PD"] = history_display["PD"].map(lambda value: f"{value:.2%}")
    history_display["Income"] = history_display["Income"].map(format_currency)
    history_display["LoanAmount"] = history_display["LoanAmount"].map(format_currency)
    st.dataframe(history_display, use_container_width=True, hide_index=True)

    chart_data = history.iloc[::-1][["Run", "PD", "RiskScore"]].set_index("Run")
    st.line_chart(chart_data, height=240)
    st.caption("History chart tracks probability of default and credit risk score across recent evaluations.")


def main() -> None:
    st.set_page_config(
        page_title="Loan Default Risk Dashboard",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_styles()
    initialize_state()

    if not MODEL_PATH.exists():
        st.error("Model artifacts are missing. Run `python train.py` first.")
        return

    model, metadata = load_artifacts()
    render_hero(metadata)
    applicant, submitted = build_input_frame()

    probability = float(model.predict_proba(applicant)[:, 1][0])
    score = probability_to_score(probability)
    risk_profile = get_risk_profile(probability, APP_THRESHOLD)
    decision = "Review" if probability >= APP_THRESHOLD else "Proceed"
    default_value, default_meaning = get_default_label(probability, APP_THRESHOLD)
    safety_message = get_loan_safety_message(probability, APP_THRESHOLD)

    if submitted:
        append_history(applicant, probability, score, decision)

    metric1, metric2, metric3 = st.columns(3, gap="large")
    with metric1:
        render_metric_card(
            "Default Probability",
            f"{probability:.2%}",
            f"Predicted class: {default_value} = {default_meaning}",
        )
    with metric2:
        render_metric_card("Decision", decision, safety_message)
    with metric3:
        render_metric_card("Credit Risk Score", f"{score}", "Business-facing score derived from the predicted default probability")

    left, right = st.columns([1.15, 1.0], gap="large")

    with left:
        st.markdown(
            f"""
            <div class="risk-banner" style="background:{risk_profile['surface']}; color:{risk_profile['color']};">
                <div class="risk-title">{risk_profile['label']}</div>
                <div>{risk_profile['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="panel-title">Prediction Visual</div>', unsafe_allow_html=True)
        render_prediction_charts(probability, APP_THRESHOLD)

        st.markdown('<div class="panel-title">Prediction History</div>', unsafe_allow_html=True)
        render_history()

    with right:
        st.markdown('<div class="panel-title">Applicant Input Snapshot</div>', unsafe_allow_html=True)
        render_snapshot(applicant)

        st.markdown('<div class="panel-title">Default Meaning</div>', unsafe_allow_html=True)
        render_default_meaning()

        st.markdown('<div class="panel-title">Risk Bands</div>', unsafe_allow_html=True)
        render_risk_ranges(APP_THRESHOLD)

        st.markdown('<div class="panel-title">Example Profiles</div>', unsafe_allow_html=True)
        render_example_profiles()


if __name__ == "__main__":
    main()
