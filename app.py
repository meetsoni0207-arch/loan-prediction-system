from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_ROOT / "models" / "best_model.joblib"
METADATA_PATH = ARTIFACTS_ROOT / "models" / "model_metadata.joblib"
METRICS_PATH = ARTIFACTS_ROOT / "metrics.json"
APP_THRESHOLD = 0.80

def probability_to_score(probability: float) -> int:
    return int(max(300, min(850, round(850 - probability * 550))))

def get_risk_profile(probability: float, threshold: float) -> dict[str, str]:
    if probability >= max(0.65, threshold + 0.15):
        return {
            "label": "High Risk",
            "color": "#b42318",
            "surface": "#fff1f0",
            "description": "This applicant sits well above the decision threshold and should be treated as high default risk.",
        }
    if probability >= threshold:
        return {
            "label": "Review Required",
            "color": "#b54708",
            "surface": "#fff7e8",
            "description": "This case crosses the model threshold and should go through manual review or tighter pricing.",
        }
    return {
        "label": "Lower Risk",
        "color": "#027a48",
        "surface": "#ecfdf3",
        "description": "This applicant remains below the decision threshold and looks comparatively safer.",
    }

def format_currency(value: float) -> str:
    return f"${value:,.0f}"

@st.cache_resource
def load_model_artifacts():
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, metadata

@st.cache_data
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))

def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 12% 10%, rgba(39, 110, 140, 0.18), transparent 26%),
                radial-gradient(circle at 88% 14%, rgba(26, 156, 129, 0.12), transparent 24%),
                linear-gradient(180deg, #f4f8fb 0%, #e9f0f4 52%, #edf5f7 100%);
            color: #102a43;
        }

        [data-testid="stSidebar"] {
            display: none;
        }

        [data-testid="stHeader"] {
            background: transparent !important;
            height: 0rem !important;
            border: none !important;
        }

        [data-testid="stToolbar"] {
            top: 0.55rem;
            right: 0.85rem;
        }

        [data-testid="stDecoration"] {
            display: none;
        }

        .block-container {
            padding-top: 0.75rem;
            padding-bottom: 2rem;
            max-width: 1240px;
        }

        .hero {
            padding: 2rem 2.2rem;
            border-radius: 30px;
            background: linear-gradient(135deg, #113756 0%, #1d6283 55%, #28708f 100%);
            color: #f8fbfc;
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: 0 24px 60px rgba(17, 55, 86, 0.18);
            margin-bottom: 1.45rem;
        }

        .hero h1 {
            margin: 0 0 0.45rem 0;
            font-size: 2.6rem;
            letter-spacing: -0.02em;
            color: #f8fbfc !important;
        }

        .hero p {
            margin: 0;
            max-width: 930px;
            font-size: 1.06rem;
            line-height: 1.75;
            color: rgba(248, 251, 252, 0.9) !important;
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
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 999px;
            padding: 0.6rem 1rem;
            font-size: 0.94rem;
            backdrop-filter: blur(8px);
        }

        .panel-title {
            font-size: 1.08rem;
            font-weight: 700;
            color: #102a43;
            margin-bottom: 0.9rem;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(17, 55, 86, 0.07);
            border-radius: 24px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 18px 36px rgba(16, 42, 67, 0.07);
            backdrop-filter: blur(10px);
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

        .section-card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(17, 55, 86, 0.07);
            border-radius: 24px;
            padding: 1.1rem 1.1rem;
            box-shadow: 0 18px 38px rgba(16, 42, 67, 0.06);
            backdrop-filter: blur(10px);
            margin-bottom: 1rem;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.9rem;
        }

        .info-tile {
            background: #f8fbfc;
            border: 1px solid rgba(16, 42, 67, 0.08);
            border-radius: 18px;
            padding: 0.95rem 1rem;
        }

        .info-tile h3 {
            margin: 0 0 0.3rem 0;
            font-size: 1rem;
            color: #102a43 !important;
        }

        .info-tile p {
            margin: 0;
            color: #486581 !important;
            font-size: 0.94rem;
        }

        .team-card {
            background: linear-gradient(180deg, #ffffff, #f6fafc);
            border: 1px solid rgba(16, 42, 67, 0.08);
            border-radius: 20px;
            padding: 1rem;
            box-shadow: 0 10px 20px rgba(16, 42, 67, 0.05);
        }

        .team-name {
            font-size: 1.05rem;
            font-weight: 700;
            color: #102a43 !important;
        }

        .team-role {
            color: #156082 !important;
            font-weight: 600;
            margin-top: 0.2rem;
        }

        .nav-button-row {
            margin-bottom: 1rem;
        }

        .nav-button-row + div div.stButton > button {
            border-radius: 999px !important;
            min-height: 2.75rem !important;
            border: 1px solid rgba(17, 55, 86, 0.10) !important;
            background: rgba(255, 255, 255, 0.78) !important;
            color: #234e68 !important;
            font-weight: 700 !important;
            box-shadow: 0 10px 28px rgba(16, 42, 67, 0.06);
            backdrop-filter: blur(10px);
        }

        .nav-button-row + div div.stButton > button:hover {
            border-color: rgba(29, 103, 130, 0.35) !important;
            color: #10324f !important;
            background: rgba(255, 255, 255, 0.92) !important;
        }

        .nav-button-row + div div.stButton > button:focus {
            box-shadow: 0 0 0 0.15rem rgba(29, 103, 130, 0.18) !important;
        }

        .panel-title,
        .metric-label,
        .metric-value,
        .metric-copy,
        .risk-title,
        .section-card,
        .section-card *,
        .stSelectbox label,
        .stSlider label,
        .stNumberInput label,
        label,
        p,
        li,
        h2,
        h3 {
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

        [data-testid="stSelectbox"] [data-baseweb="select"] {
            background: #ffffff !important;
            border-radius: 14px !important;
        }

        [data-testid="stTable"] table,
        [data-testid="stTable"] th,
        [data-testid="stTable"] td,
        [data-testid="stDataFrame"] *,
        table, thead, tbody, tr, th, td {
            color: #102a43 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def initialize_state() -> None:
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"

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

def render_page_hero(title: str, description: str, chips: list[str]) -> None:
    chip_markup = "".join(f'<div class="hero-chip">{chip}</div>' for chip in chips)
    st.markdown(
        f"""
        <section class="hero">
            <h1>{title}</h1>
            <p>{description}</p>
            <div class="hero-chip-row">{chip_markup}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

def render_navigation() -> str:
    st.markdown('<div class="nav-button-row">', unsafe_allow_html=True)
    col1, col2, col3, spacer = st.columns([1, 1, 1, 5], gap="small")

    with col1:
        if st.button("Home", key="nav_home"):
            st.session_state.current_page = "Home"
    with col2:
        if st.button("Predict", key="nav_predict"):
            st.session_state.current_page = "Predict"
    with col3:
        if st.button("About", key="nav_about"):
            st.session_state.current_page = "About"

    st.markdown("</div>", unsafe_allow_html=True)
    return st.session_state.current_page

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
            submitted = st.form_submit_button("Evaluate Applicant")

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
    st.dataframe(snapshot, width=700, hide_index=True)

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
            {"Risk Band": "Low Risk", "Probability Range": f"Below {max(threshold - 0.15, 0):.2f}"},
            {"Risk Band": "Intermediate Risk", "Probability Range": f"{max(threshold - 0.15, 0):.2f} to below {threshold:.2f}"},
            {"Risk Band": "Review Required", "Probability Range": f"{threshold:.2f} to below {min(threshold + 0.15, 1):.2f}"},
            {"Risk Band": "High Risk", "Probability Range": f"{min(threshold + 0.15, 1):.2f} and above"},
        ]
    )
    st.dataframe(ranges, width=700, hide_index=True)
    st.caption("These bands help interpret predicted default probability ranges in business terms.")

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
    st.dataframe(examples, width=700, hide_index=True)
    st.caption("These illustrative examples show what safer and riskier applicant patterns can look like.")

def render_history() -> None:
    history = pd.DataFrame(st.session_state.prediction_history)
    if history.empty:
        st.info("Prediction history will appear here after you evaluate an applicant.")
        return

    history_display = history.copy()
    history_display["PD"] = history_display["PD"].map(lambda value: f"{value:.2%}")
    history_display["Income"] = history_display["Income"].map(format_currency)
    history_display["LoanAmount"] = history_display["LoanAmount"].map(format_currency)
    st.dataframe(history_display, width=700, hide_index=True)

    chart_data = history.iloc[::-1][["Run", "PD", "RiskScore"]].set_index("Run")
    st.line_chart(chart_data, height=240)
    st.caption("History chart tracks probability of default and credit risk score across recent evaluations.")

def render_home_page(metrics: dict, metadata: dict) -> None:
    dataset = metrics.get("dataset", {})
    test_metrics = metrics.get("test_metrics", {})
    business = metrics.get("business_insights", {})
    model_name = metadata.get("best_model_name") or metrics.get("scoring", {}).get("best_model_name", "Logistic Regression")
    dataset_rows = dataset.get("shape_after_drop_loanid", ["-", "-"])[0]

    render_page_hero(
        "Loan Prediction System",
        "This project predicts the probability that a loan applicant may default by combining borrower details, financial behavior, and a trained machine learning model into one decision-support dashboard.",
        [
            f"Production model: {model_name}",
            f"Test accuracy: {test_metrics.get('accuracy', 0.0):.2%}",
            f"Dataset size: {dataset_rows:,}" if isinstance(dataset_rows, int) else f"Dataset size: {dataset_rows}",
        ],
    )

    left, right = st.columns([1.2, 1.0], gap="large")
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Project Description</div>', unsafe_allow_html=True)
        st.write(
            """
            The application is built to support credit-risk analysis. It helps estimate default risk before a loan is approved,
            making it easier for financial teams to review applicants consistently and faster.
            """
        )
        st.write(
            """
            Users can explore the project on the Home page, run live predictions on the Predict page, and review the complete
            project background, dataset, workflow, performance, and team details on the About page.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Why This Project Matters</div>', unsafe_allow_html=True)
        st.markdown(
            """
            - Reduces manual effort in identifying high-risk applicants.
            - Turns borrower information into a measurable probability of default.
            - Supports better decision-making with explainable metrics and model outputs.
            - Provides a simple interface for demonstrating an end-to-end ML project.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        metric1, metric2 = st.columns(2)
        with metric1:
            render_metric_card("Default Rate", f"{business.get('default_rate', 0.0):.2%}", "Observed share of defaulted borrowers in the dataset")
        with metric2:
            render_metric_card("Decision Threshold", f"{metadata.get('threshold', 0.0):.2f}", "Probability cutoff used to flag cases for review")

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Key Highlights</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="info-grid">
                <div class="info-tile">
                    <h3>Dataset</h3>
                    <p>Large borrower dataset with financial, demographic, and repayment-risk features.</p>
                </div>
                <div class="info-tile">
                    <h3>Prediction</h3>
                    <p>Live probability of default, business decision, and a derived credit risk score.</p>
                </div>
                <div class="info-tile">
                    <h3>Top Signal</h3>
                    <p>Loan-to-income pressure and interest rate strongly influence the model outcome.</p>
                </div>
                <div class="info-tile">
                    <h3>Use Case</h3>
                    <p>Useful for loan screening demos, ML showcases, and credit-risk analytics projects.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

def render_predict_page(model, metadata: dict) -> None:
    render_page_hero(
        "Predict Applicant Risk",
        "Enter borrower and loan details below to estimate default probability, business decision status, and a credit risk score from the trained model.",
        [
            "Live scoring interface",
            f"Threshold: {metadata['threshold']:.2f}",
            "Prediction history included",
        ],
    )

    applicant, submitted = build_input_frame()

    probability = float(model.predict_proba(applicant)[:, 1][0])
    score = probability_to_score(probability)
    risk_profile = get_risk_profile(probability, metadata["threshold"])
    decision = "Review" if probability >= metadata["threshold"] else "Proceed"

    if submitted:
        append_history(applicant, probability, score, decision)

    metric1, metric2, metric3 = st.columns(3, gap="large")
    with metric1:
        render_metric_card("Default Probability", f"{probability:.2%}", "Predicted probability of default for this applicant")
    with metric2:
        render_metric_card("Decision", decision, risk_profile["description"])
    with metric3:
        render_metric_card("Credit Risk Score", f"{score}", "Business-facing score derived from the predicted default probability")

    left, right = st.columns([1.15, 1.0], gap="large")

    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
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
        render_prediction_charts(probability, metadata["threshold"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Prediction History</div>', unsafe_allow_html=True)
        render_history()
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Applicant Input Snapshot</div>', unsafe_allow_html=True)
        render_snapshot(applicant)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Risk Bands</div>', unsafe_allow_html=True)
        render_risk_ranges(metadata["threshold"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Example Profiles</div>', unsafe_allow_html=True)
        render_example_profiles()
        st.markdown("</div>", unsafe_allow_html=True)
        
def render_about_page(metrics: dict, metadata: dict) -> None:
    dataset = metrics.get("dataset", {})
    scoring = metrics.get("scoring", {})
    test_metrics = metrics.get("test_metrics", {})
    validation_best = metrics.get("best_validation_model", {})
    business = metrics.get("business_insights", {})
    dataset_rows = dataset.get("shape_after_drop_loanid", ["-", "-"])[0]
    dataset_cols = dataset.get("shape_after_drop_loanid", ["-", "-"])[1]
    dataset_rows_display = f"{dataset_rows:,}" if isinstance(dataset_rows, int) else str(dataset_rows)

    render_page_hero(
        "About The Project",
        "This page summarizes the problem statement, dataset, modeling pipeline, performance, technology choices, and the team behind the loan prediction system.",
        [
            f"Best model: {validation_best.get('name', 'Logistic Regression')}",
            f"Accuracy: {test_metrics.get('accuracy', 0.0):.2%}",
            f"ROC-AUC: {test_metrics.get('roc_auc', 0.0):.3f}",
        ],
    )

    overview, problem = st.columns(2, gap="large")
    with overview:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Project Overview</div>', unsafe_allow_html=True)
        st.write(
            """
            The loan prediction system is an end-to-end machine learning project that evaluates whether an applicant is likely
            to default on a loan. It combines data preprocessing, feature engineering, model training, and an interactive Streamlit interface.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with problem:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Problem Statement</div>', unsafe_allow_html=True)
        st.write(
            """
            Financial institutions need fast and consistent risk screening. Manual review alone can be slow and inconsistent,
            so this project predicts default risk from applicant and loan attributes to support better lending decisions.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Dataset</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        - Source file used in training: `{dataset.get('path', 'Not available')}`
        - Records after dropping `LoanID`: `{dataset_rows_display}`
        - Columns after dropping `LoanID`: `{dataset_cols}`
        - Default rate in the dataset: `{dataset.get('default_rate', 0.0):.2%}`
        - Missing values: no missing values reported in the saved metrics artifact.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    workflow, performance = st.columns([1.2, 1.0], gap="large")
    with workflow:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Model Workflow</div>', unsafe_allow_html=True)
        st.markdown(
            """
            1. Collect and clean applicant financial and demographic data.
            2. Engineer features such as `Loan_to_Income`, `Credit_per_Line`, `Income_per_Dependent`, and `Rate_x_DTI`.
            3. Compare multiple machine learning models including Logistic Regression, XGBoost, Decision Tree, and Random Forest.
            4. Select the best-performing validation model and save it as the production model artifact.
            5. Serve the model through Streamlit for live loan-risk prediction.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with performance:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Performance</div>', unsafe_allow_html=True)
        metric1, metric2 = st.columns(2)
        with metric1:
            render_metric_card("Accuracy", f"{test_metrics.get('accuracy', 0.0):.2%}", "Saved test accuracy from training artifacts")
        with metric2:
            render_metric_card("Precision", f"{test_metrics.get('precision', 0.0):.2%}", "Positive-class precision on the test split")
        metric3, metric4 = st.columns(2)
        with metric3:
            render_metric_card("Recall", f"{test_metrics.get('recall', 0.0):.2%}", "Positive-class recall on the test split")
        with metric4:
            render_metric_card("ROC-AUC", f"{test_metrics.get('roc_auc', 0.0):.3f}", "Ranking quality of the selected model")
        st.caption(
            f"The current production model is {validation_best.get('name', 'Logistic Regression')} with threshold {scoring.get('threshold', metadata.get('threshold', 0.0)):.2f}."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    insights, tech = st.columns(2, gap="large")
    with insights:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Business Insights</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            - Average credit score of defaulted borrowers: `{business.get('avg_credit_score_defaulted', 0.0):.1f}`
            - Average credit score of non-defaulted borrowers: `{business.get('avg_credit_score_non_defaulted', 0.0):.1f}`
            - Average interest rate for defaulted borrowers: `{business.get('avg_interest_rate_defaulted', 0.0):.2f}`
            - Average loan-to-income ratio for defaulted borrowers: `{business.get('avg_loan_to_income_defaulted', 0.0):.2f}`
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with tech:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Tech Stack</div>', unsafe_allow_html=True)
        st.markdown(
            """
            - Python
            - Streamlit
            - Pandas
            - Joblib
            - Scikit-learn based model pipeline
            - XGBoost for model comparison
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Team Members</div>', unsafe_allow_html=True)
    team1, team2, team3 = st.columns(3, gap="large")
    with team1:
        st.markdown(
            """
            <div class="team-card">
                <div class="team-name">Yash Panchal</div>
                <div class="team-role">System Design and Frontend Development</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with team2:
        st.markdown(
            """
            <div class="team-card">
                <div class="team-name">Meet Soni</div>
                <div class="team-role">Data Preprocessing & Testing</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with team3:
        st.markdown(
            """
            <div class="team-card">
                <div class="team-name">Dhyey Makwana</div>
                <div class="team-role">Model Develop and Training</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

def main() -> None:
    st.set_page_config(
        page_title="Loan Prediction System",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_styles()
    initialize_state()

    metrics = load_metrics()
    metadata = {"threshold": APP_THRESHOLD}

    if METADATA_PATH.exists():
        _, metadata = load_model_artifacts()
        metadata["threshold"] = APP_THRESHOLD

    page = render_navigation()

    if page == "Home":
        render_home_page(metrics, metadata)
        return

    if page == "About":
        render_about_page(metrics, metadata)
        return

    if not MODEL_PATH.exists():
        st.error("Model artifacts are missing. Run `python train.py` first.")
        return

    model, metadata = load_model_artifacts()
    metadata["threshold"] = APP_THRESHOLD
    render_predict_page(model, metadata)

if __name__ == "__main__":
    main()

# from __future__ import annotations

# from pathlib import Path

# import joblib
# import pandas as pd
# import streamlit as st

# PROJECT_ROOT = Path(__file__).resolve().parent
# MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "best_model.joblib"
# METADATA_PATH = PROJECT_ROOT / "artifacts" / "models" / "model_metadata.joblib"
# APP_THRESHOLD = 0.80


# def probability_to_score(probability: float) -> int:
#     return int(max(300, min(850, round(850 - probability * 550))))


# def get_risk_band(probability: float) -> dict[str, str]:
#     if probability >= 0.80:
#         return {
#             "label": "High Risk",
#             "color": "#b42318",
#             "surface": "#fff1f0",
#             "range": "80% to 100%",
#         }
#     if probability >= 0.60:
#         return {
#             "label": "Review Required",
#             "color": "#b54708",
#             "surface": "#fff7e8",
#             "range": "60% to below 80%",
#         }
#     if probability >= 0.30:
#         return {
#             "label": "Intermediate Risk",
#             "color": "#b69208",
#             "surface": "#fffbea",
#             "range": "30% to below 60%",
#         }
#     return {
#         "label": "Low Risk",
#         "color": "#027a48",
#         "surface": "#ecfdf3",
#         "range": "0% to below 30%",
#     }


# def get_risk_profile(probability: float, threshold: float) -> dict[str, str]:
#     band = get_risk_band(probability)
#     if probability >= threshold:
#         description = "This case crosses the model threshold and should go through manual review or tighter pricing."
#     else:
#         description = "This applicant remains below the model threshold and looks comparatively safer."
#     return {
#         "label": band["label"],
#         "color": band["color"],
#         "surface": band["surface"],
#         "description": f"{description} Risk band: {band['range']}.",
#     }


# def format_currency(value: float) -> str:
#     return f"${value:,.0f}"


# @st.cache_resource
# def load_artifacts():
#     model = joblib.load(MODEL_PATH)
#     metadata = joblib.load(METADATA_PATH)
#     return model, metadata


# def apply_styles() -> None:
#     st.markdown(
#         """
#         <style>
#         .stApp {
#             background:
#                 radial-gradient(circle at top left, rgba(18, 52, 77, 0.10), transparent 28%),
#                 radial-gradient(circle at top right, rgba(8, 145, 178, 0.10), transparent 24%),
#                 linear-gradient(180deg, #f7fafb 0%, #eef4f6 100%);
#             color: #102a43;
#         }

#         [data-testid="stSidebar"] {
#             display: none;
#         }

#         .block-container {
#             padding-top: 1.5rem;
#             padding-bottom: 2rem;
#             max-width: 1280px;
#         }

#         .hero {
#             padding: 1.6rem 1.8rem;
#             border-radius: 26px;
#             background: linear-gradient(135deg, #102a43 0%, #156082 100%);
#             color: #f8fbfc;
#             box-shadow: 0 22px 48px rgba(16, 42, 67, 0.16);
#             margin-bottom: 1.25rem;
#         }

#         .hero h1 {
#             margin: 0 0 0.45rem 0;
#             font-size: 2.2rem;
#             letter-spacing: -0.02em;
#             color: #f8fbfc !important;
#         }

#         .hero p {
#             margin: 0;
#             max-width: 820px;
#             color: rgba(248, 251, 252, 0.9);
#         }

#         .hero strong,
#         .hero-chip {
#             color: #f8fbfc !important;
#         }

#         .hero-chip-row {
#             display: flex;
#             gap: 0.65rem;
#             flex-wrap: wrap;
#             margin-top: 1rem;
#         }

#         .hero-chip {
#             background: rgba(255, 255, 255, 0.12);
#             border: 1px solid rgba(255, 255, 255, 0.15);
#             border-radius: 999px;
#             padding: 0.45rem 0.85rem;
#             font-size: 0.88rem;
#         }

#         .panel {
#             background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(247, 250, 252, 0.96));
#             border: 1px solid rgba(16, 42, 67, 0.10);
#             border-radius: 24px;
#             padding: 1.15rem 1.2rem;
#             box-shadow: 0 12px 30px rgba(16, 42, 67, 0.08);
#             margin-bottom: 1rem;
#         }

#         .panel-title {
#             font-size: 1.05rem;
#             font-weight: 700;
#             color: #102a43;
#             margin-bottom: 0.85rem;
#         }

#         .metric-card {
#             background: rgba(255, 255, 255, 0.96);
#             border: 1px solid rgba(16, 42, 67, 0.08);
#             border-radius: 22px;
#             padding: 1rem 1.1rem;
#             box-shadow: 0 10px 24px rgba(16, 42, 67, 0.08);
#         }

#         .metric-label {
#             color: #627d98;
#             font-size: 0.78rem;
#             letter-spacing: 0.08em;
#             text-transform: uppercase;
#         }

#         .metric-value {
#             color: #102a43;
#             font-size: 2rem;
#             font-weight: 700;
#             margin-top: 0.35rem;
#             line-height: 1.1;
#         }

#         .metric-copy {
#             color: #627d98;
#             font-size: 0.9rem;
#             margin-top: 0.4rem;
#         }

#         .risk-banner {
#             border-radius: 20px;
#             padding: 1rem 1.05rem;
#             border: 1px solid rgba(16, 42, 67, 0.08);
#             margin-bottom: 1rem;
#         }

#         .risk-title {
#             font-size: 1rem;
#             font-weight: 700;
#             margin-bottom: 0.25rem;
#         }

#         .chart-caption {
#             color: #627d98;
#             font-size: 0.9rem;
#             margin-top: 0.55rem;
#         }

#         .panel-title,
#         .metric-label,
#         .metric-value,
#         .metric-copy,
#         .risk-title,
#         .section-card,
#         .section-card *,
#         .panel,
#         .panel *,
#         .stSelectbox label,
#         .stSlider label,
#         .stNumberInput label,
#         label {
#             color: #102a43 !important;
#         }

#         [data-testid="stWidgetLabel"] p,
#         [data-testid="stForm"] [data-testid="stMarkdownContainer"] p,
#         [data-testid="stForm"] [data-testid="stMarkdownContainer"] li,
#         [data-testid="stForm"] [data-testid="stMarkdownContainer"] span {
#             color: #102a43 !important;
#         }

#         div[data-baseweb="select"] span,
#         div[data-baseweb="select"] div,
#         div[data-baseweb="input"] input,
#         div[data-baseweb="select"] input,
#         div[data-baseweb="base-input"] input {
#             color: #102a43 !important;
#             -webkit-text-fill-color: #102a43 !important;
#             opacity: 1 !important;
#         }

#         div[data-baseweb="select"] > div,
#         div[data-baseweb="base-input"] > div,
#         [data-testid="stNumberInput"] div[data-baseweb="input"] > div {
#             background: #ffffff !important;
#             border-color: rgba(16, 42, 67, 0.14) !important;
#         }

#         [data-testid="stNumberInput"] input {
#             background: #ffffff !important;
#             color: #102a43 !important;
#             -webkit-text-fill-color: #102a43 !important;
#         }

#         [data-testid="stNumberInput"] button,
#         [data-testid="stSelectbox"] svg {
#             color: #102a43 !important;
#         }

#         [data-testid="stSelectbox"] [data-baseweb="select"] {
#             background: #ffffff !important;
#             border-radius: 14px !important;
#         }

#         [data-testid="stSelectbox"] [data-baseweb="select"] > div {
#             min-height: 52px;
#         }

#         [data-testid="stSelectbox"] [data-baseweb="select"] > div * {
#             color: #102a43 !important;
#             fill: #102a43 !important;
#             opacity: 1 !important;
#         }

#         [data-testid="stSelectbox"] [class*="singleValue"],
#         [data-testid="stSelectbox"] [class*="placeholder"],
#         [data-testid="stSelectbox"] [class*="valueContainer"],
#         [data-testid="stSelectbox"] [class*="ValueContainer"],
#         [data-testid="stSelectbox"] [class*="control"] {
#             color: #102a43 !important;
#             -webkit-text-fill-color: #102a43 !important;
#             opacity: 1 !important;
#         }

#         [data-testid="stTable"] table,
#         [data-testid="stTable"] th,
#         [data-testid="stTable"] td,
#         [data-testid="stDataFrame"] *,
#         table, thead, tbody, tr, th, td {
#             color: #102a43 !important;
#         }

#         [data-testid="stTable"] table,
#         [data-testid="stDataFrame"] {
#             background: #ffffff !important;
#             border-radius: 16px !important;
#             overflow: hidden;
#             border: 1px solid rgba(16, 42, 67, 0.08);
#         }

#         [data-testid="stTable"] th,
#         [data-testid="stDataFrame"] thead th {
#             background: #eaf2f5 !important;
#             font-weight: 700 !important;
#         }

#         [data-testid="stForm"] {
#             background: transparent;
#             border: none;
#         }

#         [data-testid="stForm"] > div {
#             border: none !important;
#             padding: 0 !important;
#             background: transparent !important;
#             box-shadow: none !important;
#         }

#         </style>
#         """,
#         unsafe_allow_html=True,
#     )


# def initialize_state() -> None:
#     if "prediction_history" not in st.session_state:
#         st.session_state.prediction_history = []


# def build_input_frame() -> tuple[pd.DataFrame, bool]:
#     left_spacer, center, right_spacer = st.columns([0.08, 0.84, 0.08])
#     with center:
#         st.markdown('<div class="panel-title">Underwriting Inputs</div>', unsafe_allow_html=True)
#         with st.form("application_form", clear_on_submit=False):
#             fin1, fin2, fin3 = st.columns(3)
#             with fin1:
#                 income = st.number_input("Annual Income", min_value=1000, value=60000, step=1000)
#                 credit_score = st.slider("Credit Score", 300, 850, 680)
#             with fin2:
#                 loan_amount = st.number_input("Requested Loan Amount", min_value=1000, value=25000, step=1000)
#                 interest_rate = st.slider("Interest Rate", 1.0, 35.0, 10.0)
#             with fin3:
#                 dti_ratio = st.slider("DTI Ratio", 0.0, 1.0, 0.25)
#                 num_credit_lines = st.slider("Number of Credit Lines", 1, 15, 4)

#             borrower1, borrower2, borrower3 = st.columns(3)
#             with borrower1:
#                 age = st.slider("Age", 18, 75, 35)
#                 months_employed = st.slider("Months Employed", 0, 240, 48)
#             with borrower2:
#                 education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
#                 employment_type = st.selectbox(
#                     "Employment Type",
#                     ["Full-time", "Part-time", "Self-employed", "Unemployed"],
#                 )
#             with borrower3:
#                 marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
#                 loan_term = st.selectbox("Loan Term", [12, 24, 36, 48, 60, 72])

#             app1, app2, app3 = st.columns(3)
#             with app1:
#                 has_mortgage = st.selectbox("Has Mortgage", ["No", "Yes"])
#             with app2:
#                 has_dependents = st.selectbox("Has Dependents", ["No", "Yes"])
#             with app3:
#                 has_cosigner = st.selectbox("Has Co-Signer", ["No", "Yes"])

#             loan_purpose = st.selectbox("Loan Purpose", ["Home", "Auto", "Education", "Business", "Other"])
#             submitted = st.form_submit_button("Evaluate Applicant", width=700, type="primary")

#     frame = pd.DataFrame(
#         [
#             {
#                 "Age": age,
#                 "Income": income,
#                 "LoanAmount": loan_amount,
#                 "CreditScore": credit_score,
#                 "MonthsEmployed": months_employed,
#                 "NumCreditLines": num_credit_lines,
#                 "InterestRate": interest_rate,
#                 "LoanTerm": loan_term,
#                 "DTIRatio": dti_ratio,
#                 "Education": education,
#                 "EmploymentType": employment_type,
#                 "MaritalStatus": marital_status,
#                 "HasMortgage": has_mortgage,
#                 "HasDependents": has_dependents,
#                 "LoanPurpose": loan_purpose,
#                 "HasCoSigner": has_cosigner,
#             }
#         ]
#     )

#     frame["Loan_to_Income"] = frame["LoanAmount"] / frame["Income"].clip(lower=1)
#     frame["Credit_per_Line"] = frame["CreditScore"] / frame["NumCreditLines"].clip(lower=1)
#     frame["Income_per_Dependent"] = frame["Income"] / (
#         1 + frame["HasDependents"].map({"Yes": 1, "No": 0}).fillna(0)
#     )
#     frame["Rate_x_DTI"] = frame["InterestRate"] * frame["DTIRatio"]
#     return frame, submitted


# def render_hero(metadata: dict) -> None:
#     model_name = metadata.get("best_model_name") or metadata.get("scoring", {}).get("best_model_name", "Best Model")
#     st.markdown(
#         f"""
#         <section class="hero">
#             <h1>Loan Default Risk Dashboard</h1>
#             <p>
#                 Applicant evaluation workspace powered by the trained machine learning model.
#                 The current production model is <strong>{model_name}</strong>, selected for maximum validation accuracy.
#             </p>
#             <div class="hero-chip-row">
#                 <div class="hero-chip">Accuracy-first model selection</div>
#                 <div class="hero-chip">Decision threshold: {APP_THRESHOLD:.2f}</div>
#                 <div class="hero-chip">Credit risk score range: 300 to 850</div>
#             </div>
#         </section>
#         """,
#         unsafe_allow_html=True,
#     )


# def render_metric_card(label: str, value: str, subtle: str) -> None:
#     st.markdown(
#         f"""
#         <div class="metric-card">
#             <div class="metric-label">{label}</div>
#             <div class="metric-value">{value}</div>
#             <div class="metric-copy">{subtle}</div>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )


# def get_default_label(probability: float, threshold: float) -> tuple[int, str]:
#     predicted_default = 1 if probability >= threshold else 0
#     meaning = "Default" if predicted_default == 1 else "No Default"
#     return predicted_default, meaning


# def get_loan_safety_message(probability: float, threshold: float) -> str:
#     if probability >= threshold:
#         return "Not safe to give the loan under the current model threshold."
#     return "Safer to give the loan under the current model threshold."


# def render_snapshot(applicant: pd.DataFrame) -> None:
#     snapshot = pd.DataFrame(
#         {
#             "Field": [
#                 "Annual Income",
#                 "Requested Loan Amount",
#                 "Credit Score",
#                 "Interest Rate",
#                 "DTI Ratio",
#                 "Employment Length",
#                 "Education",
#                 "Employment Type",
#                 "Marital Status",
#                 "Loan Purpose",
#                 "Loan to Income",
#                 "Credit per Line",
#             ],
#             "Value": [
#                 format_currency(float(applicant.at[0, "Income"])),
#                 format_currency(float(applicant.at[0, "LoanAmount"])),
#                 f"{int(applicant.at[0, 'CreditScore'])}",
#                 f"{float(applicant.at[0, 'InterestRate']):.2f}%",
#                 f"{float(applicant.at[0, 'DTIRatio']):.2f}",
#                 f"{int(applicant.at[0, 'MonthsEmployed'])} months",
#                 applicant.at[0, "Education"],
#                 applicant.at[0, "EmploymentType"],
#                 applicant.at[0, "MaritalStatus"],
#                 applicant.at[0, "LoanPurpose"],
#                 f"{float(applicant.at[0, 'Loan_to_Income']):.2f}",
#                 f"{float(applicant.at[0, 'Credit_per_Line']):.2f}",
#             ],
#         }
#     )
#     st.dataframe(snapshot, width=700, hide_index=True)


# def append_history(applicant: pd.DataFrame, probability: float, score: int, decision: str) -> None:
#     row = {
#         "Run": len(st.session_state.prediction_history) + 1,
#         "Income": float(applicant.at[0, "Income"]),
#         "LoanAmount": float(applicant.at[0, "LoanAmount"]),
#         "CreditScore": int(applicant.at[0, "CreditScore"]),
#         "PD": round(probability, 4),
#         "RiskScore": score,
#         "Decision": decision,
#     }
#     st.session_state.prediction_history.insert(0, row)
#     st.session_state.prediction_history = st.session_state.prediction_history[:12]


# def render_prediction_charts(probability: float, threshold: float) -> None:
#     comparison = pd.DataFrame(
#         {
#             "Metric": ["Default Probability", "Decision Threshold"],
#             "Value": [probability, threshold],
#         }
#     ).set_index("Metric")
#     st.bar_chart(comparison, height=260)
#     st.caption("This chart compares the applicant's predicted probability of default against the model threshold.")


# def render_risk_ranges(threshold: float) -> None:
#     ranges = pd.DataFrame(
#         [
#             {"Risk Band": "Low Risk", "Probability Range": "0% to below 30%", "Typical View": "Generally safer profile"},
#             {"Risk Band": "Intermediate Risk", "Probability Range": "30% to below 60%", "Typical View": "Borderline, review context"},
#             {"Risk Band": "Review Required", "Probability Range": "60% to below 80%", "Typical View": "Needs stricter review"},
#             {"Risk Band": "High Risk", "Probability Range": "80% to 100%", "Typical View": "Very risky lending case"},
#         ]
#     )
#     st.dataframe(ranges, width=700, hide_index=True)
#     st.caption(
#         f"These are business-facing probability bands. The current model decision threshold is {threshold:.2f} ({threshold:.0%})."
#     )


# def render_default_meaning() -> None:
#     default_mapping = pd.DataFrame(
#         [
#             {"Default": 0, "Meaning": "No Default"},
#             {"Default": 1, "Meaning": "Default"},
#         ]
#     )
#     st.dataframe(default_mapping, width=700, hide_index=True)
#     st.caption("Target label used by the model: `0` means the loan is not expected to default, `1` means default.")


# def render_example_profiles() -> None:
#     examples = pd.DataFrame(
#         [
#             {
#                 "Example": "Low Risk",
#                 "Income": "$95,000",
#                 "Loan Amount": "$18,000",
#                 "Credit Score": "760",
#                 "Interest Rate": "7%",
#                 "DTI": "0.18",
#                 "Employment": "84 months, Full-time",
#                 "Typical Signal": "Higher income, lower loan burden, stronger credit history",
#             },
#             {
#                 "Example": "Intermediate Risk",
#                 "Income": "$58,000",
#                 "Loan Amount": "$42,000",
#                 "Credit Score": "660",
#                 "Interest Rate": "13%",
#                 "DTI": "0.36",
#                 "Employment": "30 months, Part-time",
#                 "Typical Signal": "Moderate leverage and average credit quality",
#             },
#             {
#                 "Example": "High Risk",
#                 "Income": "$28,000",
#                 "Loan Amount": "$85,000",
#                 "Credit Score": "520",
#                 "Interest Rate": "24%",
#                 "DTI": "0.72",
#                 "Employment": "6 months, Unemployed/unstable",
#                 "Typical Signal": "High loan-to-income pressure, weaker credit, expensive borrowing",
#             },
#         ]
#     )
#     st.dataframe(examples, width=700, hide_index=True)
#     st.caption("These are illustrative examples to help users understand what input patterns tend to look safer or riskier.")


# def render_history() -> None:
#     history = pd.DataFrame(st.session_state.prediction_history)
#     if history.empty:
#         st.info("Prediction history will appear here after you evaluate an applicant.")
#         return

#     history_display = history.copy()
#     history_display["PD"] = history_display["PD"].map(lambda value: f"{value:.2%}")
#     history_display["Income"] = history_display["Income"].map(format_currency)
#     history_display["LoanAmount"] = history_display["LoanAmount"].map(format_currency)
#     st.dataframe(history_display, width=700, hide_index=True)

#     chart_data = history.iloc[::-1][["Run", "PD", "RiskScore"]].set_index("Run")
#     st.line_chart(chart_data, height=240)
#     st.caption("History chart tracks probability of default and credit risk score across recent evaluations.")


# def main() -> None:
#     st.set_page_config(
#         page_title="Loan Default Risk Dashboard",
#         layout="wide",
#         initial_sidebar_state="collapsed",
#     )
#     apply_styles()
#     initialize_state()

#     if not MODEL_PATH.exists():
#         st.error("Model artifacts are missing. Run `python train.py` first.")
#         return

#     model, metadata = load_artifacts()
#     render_hero(metadata)
#     applicant, submitted = build_input_frame()

#     probability = float(model.predict_proba(applicant)[:, 1][0])
#     score = probability_to_score(probability)
#     risk_profile = get_risk_profile(probability, APP_THRESHOLD)
#     decision = "Review" if probability >= APP_THRESHOLD else "Proceed"
#     default_value, default_meaning = get_default_label(probability, APP_THRESHOLD)
#     safety_message = get_loan_safety_message(probability, APP_THRESHOLD)

#     if submitted:
#         append_history(applicant, probability, score, decision)

#     metric1, metric2, metric3 = st.columns(3, gap="large")
#     with metric1:
#         render_metric_card(
#             "Default Probability",
#             f"{probability:.2%}",
#             f"Predicted class: {default_value} = {default_meaning}",
#         )
#     with metric2:
#         render_metric_card("Decision", decision, safety_message)
#     with metric3:
#         render_metric_card("Credit Risk Score", f"{score}", "Business-facing score derived from the predicted default probability")

#     left, right = st.columns([1.15, 1.0], gap="large")

#     with left:
#         st.markdown(
#             f"""
#             <div class="risk-banner" style="background:{risk_profile['surface']}; color:{risk_profile['color']};">
#                 <div class="risk-title">{risk_profile['label']}</div>
#                 <div>{risk_profile['description']}</div>
#             </div>
#             """,
#             unsafe_allow_html=True,
#         )
#         st.markdown('<div class="panel-title">Prediction Visual</div>', unsafe_allow_html=True)
#         render_prediction_charts(probability, APP_THRESHOLD)

#         st.markdown('<div class="panel-title">Prediction History</div>', unsafe_allow_html=True)
#         render_history()

#     with right:
#         st.markdown('<div class="panel-title">Applicant Input Snapshot</div>', unsafe_allow_html=True)
#         render_snapshot(applicant)

#         st.markdown('<div class="panel-title">Default Meaning</div>', unsafe_allow_html=True)
#         render_default_meaning()

#         st.markdown('<div class="panel-title">Risk Bands</div>', unsafe_allow_html=True)
#         render_risk_ranges(APP_THRESHOLD)

#         st.markdown('<div class="panel-title">Example Profiles</div>', unsafe_allow_html=True)
#         render_example_profiles()

# if __name__ == "__main__":
#     main()
# =======
# if __name__ == "__main__":
#     main()
# >>>>>>> 8cb0757d03465b1673dcb18480a3e68c69b166c7
