from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
DATASET_PATH = Path("loan_data.csv")  # 🔥 FIXED path (portable)
PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
MODELS_DIR = ARTIFACTS_DIR / "models"


@dataclass
class ModelResult:
    name: str
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]
    best_params: dict[str, Any]


def ensure_directories():
    for d in (ARTIFACTS_DIR, PLOTS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.drop(columns=["LoanID"])


def engineer_features(df):
    df = df.copy()
    df["Loan_to_Income"] = df["LoanAmount"] / df["Income"].clip(lower=1)
    df["Credit_per_Line"] = df["CreditScore"] / df["NumCreditLines"].clip(lower=1)
    df["Income_per_Dependent"] = df["Income"] / (
        1 + df["HasDependents"].map({"Yes": 1, "No": 0}).fillna(0)
    )
    df["Rate_x_DTI"] = df["InterestRate"] * df["DTIRatio"]
    return df


def infer_feature_types(df, target):
    cat = df.drop(columns=[target]).select_dtypes(include="object").columns.tolist()
    num = df.drop(columns=[target]).select_dtypes(exclude="object").columns.tolist()
    return cat, num


def make_preprocessor(cat, num):
    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), num),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OneHotEncoder(handle_unknown="ignore"))
        ]), cat)
    ])


# 🔥 REMOVED XGBOOST COMPLETELY
def build_models(preprocessor):
    return {
        "Logistic Regression": Pipeline([
            ("prep", clone(preprocessor)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),
        "Decision Tree": Pipeline([
            ("prep", clone(preprocessor)),
            ("model", DecisionTreeClassifier(class_weight="balanced"))
        ]),
        "Random Forest": Pipeline([
            ("prep", clone(preprocessor)),
            ("model", RandomForestClassifier(n_estimators=200, class_weight="balanced"))
        ])
    }


def evaluate(name, model, X, y):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob > 0.5).astype(int)

    return ModelResult(
        name=name,
        threshold=0.5,
        accuracy=accuracy_score(y, pred),
        precision=precision_score(y, pred),
        recall=recall_score(y, pred),
        f1=f1_score(y, pred),
        roc_auc=roc_auc_score(y, prob),
        confusion_matrix=confusion_matrix(y, pred).tolist(),
        best_params={}
    )


def main():
    ensure_directories()

    df = engineer_features(load_data(DATASET_PATH))
    target = "Default"

    cat, num = infer_feature_types(df, target)
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pre = make_preprocessor(cat, num)
    models = build_models(pre)

    best_model = None
    best_score = -1

    for name, model in models.items():
        model.fit(X_train, y_train)
        result = evaluate(name, model, X_test, y_test)

        print(f"{name}: {result.accuracy:.4f}")

        if result.accuracy > best_score:
            best_score = result.accuracy
            best_model = model

    # 🔥 SAVE CLEAN MODEL
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")

    joblib.dump({
        "threshold": 0.5,
        "categorical_columns": cat,
        "numeric_columns": num
    }, MODELS_DIR / "model_metadata.joblib")

    print("✅ Training complete")


if __name__ == "__main__":
    main()

# from __future__ import annotations

# import json
# from dataclasses import asdict, dataclass
# from pathlib import Path
# from typing import Any

# import joblib
# import matplotlib

# matplotlib.use("Agg")

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import shap
# from sklearn.base import clone
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     ConfusionMatrixDisplay,
#     accuracy_score,
#     classification_report,
#     confusion_matrix,
#     f1_score,
#     precision_score,
#     recall_score,
#     roc_auc_score,
#     roc_curve,
# )
# from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier


# RANDOM_STATE = 42
# DATASET_PATH = Path(r"c:\ml new pro\loan_data.csv")
# PROJECT_ROOT = Path(__file__).resolve().parent
# ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
# PLOTS_DIR = ARTIFACTS_DIR / "plots"
# MODELS_DIR = ARTIFACTS_DIR / "models"


# @dataclass
# class ModelResult:
#     name: str
#     threshold: float
#     accuracy: float
#     precision: float
#     recall: float
#     f1: float
#     roc_auc: float
#     confusion_matrix: list[list[int]]
#     best_params: dict[str, Any]


# def ensure_directories() -> None:
#     for directory in (ARTIFACTS_DIR, PLOTS_DIR, MODELS_DIR):
#         directory.mkdir(parents=True, exist_ok=True)


# def load_data(path: Path) -> pd.DataFrame:
#     df = pd.read_csv(path)
#     df = df.drop(columns=["LoanID"])
#     return df


# def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
#     engineered = df.copy()
#     engineered["Loan_to_Income"] = engineered["LoanAmount"] / engineered["Income"].clip(lower=1)
#     engineered["Credit_per_Line"] = engineered["CreditScore"] / engineered["NumCreditLines"].clip(lower=1)
#     engineered["Income_per_Dependent"] = engineered["Income"] / (
#         1 + engineered["HasDependents"].map({"Yes": 1, "No": 0}).fillna(0)
#     )
#     engineered["Rate_x_DTI"] = engineered["InterestRate"] * engineered["DTIRatio"]
#     return engineered


# def infer_feature_types(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
#     categorical_cols = df.drop(columns=[target]).select_dtypes(include=["object"]).columns.tolist()
#     numeric_cols = df.drop(columns=[target]).select_dtypes(exclude=["object"]).columns.tolist()
#     return categorical_cols, numeric_cols


# def make_preprocessor(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
#     numeric_transformer = Pipeline(
#         steps=[
#             ("imputer", SimpleImputer(strategy="median")),
#             ("scaler", StandardScaler()),
#         ]
#     )
#     categorical_transformer = Pipeline(
#         steps=[
#             ("imputer", SimpleImputer(strategy="most_frequent")),
#             ("encoder", OneHotEncoder(handle_unknown="ignore")),
#         ]
#     )
#     return ColumnTransformer(
#         transformers=[
#             ("num", numeric_transformer, numeric_cols),
#             ("cat", categorical_transformer, categorical_cols),
#         ]
#     )


# def detect_outliers(df: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
#     summary: dict[str, dict[str, float]] = {}
#     for col in columns:
#         q1 = float(df[col].quantile(0.25))
#         q3 = float(df[col].quantile(0.75))
#         iqr = q3 - q1
#         lower = q1 - 1.5 * iqr
#         upper = q3 + 1.5 * iqr
#         mask = (df[col] < lower) | (df[col] > upper)
#         summary[col] = {
#             "q1": q1,
#             "q3": q3,
#             "iqr": iqr,
#             "lower_bound": lower,
#             "upper_bound": upper,
#             "outlier_count": int(mask.sum()),
#             "outlier_pct": float(mask.mean() * 100),
#         }
#     return summary


# def save_eda_plots(df: pd.DataFrame) -> dict[str, str]:
#     sns.set_theme(style="whitegrid")
#     plot_paths: dict[str, str] = {}

#     plt.figure(figsize=(8, 5))
#     sns.countplot(data=df, x="Default")
#     plt.title("Class Distribution of Loan Default")
#     plt.tight_layout()
#     path = PLOTS_DIR / "class_distribution.png"
#     plt.savefig(path, dpi=200)
#     plt.close()
#     plot_paths["class_distribution"] = str(path)

#     plt.figure(figsize=(8, 5))
#     sns.boxplot(data=df, x="Default", y="CreditScore")
#     plt.title("Credit Score vs Default")
#     plt.tight_layout()
#     path = PLOTS_DIR / "credit_score_vs_default.png"
#     plt.savefig(path, dpi=200)
#     plt.close()
#     plot_paths["credit_score_vs_default"] = str(path)

#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#     for ax, col in zip(axes, ["Income", "DTIRatio", "InterestRate"]):
#         sns.violinplot(data=df, x="Default", y=col, ax=ax, inner="quartile")
#         ax.set_title(f"{col} vs Default")
#     fig.tight_layout()
#     path = PLOTS_DIR / "income_dti_interest_vs_default.png"
#     fig.savefig(path, dpi=200)
#     plt.close(fig)
#     plot_paths["income_dti_interest_vs_default"] = str(path)

#     corr = df.select_dtypes(exclude=["object"]).corr(numeric_only=True)
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
#     plt.title("Correlation Heatmap")
#     plt.tight_layout()
#     path = PLOTS_DIR / "correlation_heatmap.png"
#     plt.savefig(path, dpi=200)
#     plt.close()
#     plot_paths["correlation_heatmap"] = str(path)

#     for col in ["Income", "LoanAmount", "DTIRatio"]:
#         plt.figure(figsize=(8, 5))
#         sns.boxplot(x=df[col])
#         plt.title(f"Outlier Inspection for {col}")
#         plt.tight_layout()
#         path = PLOTS_DIR / f"outlier_{col.lower()}.png"
#         plt.savefig(path, dpi=200)
#         plt.close()
#         plot_paths[f"outlier_{col.lower()}"] = str(path)

#     return plot_paths


# def build_model_candidates(preprocessor: ColumnTransformer, pos_weight: float) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
#     candidates: dict[str, tuple[Any, dict[str, list[Any]]]] = {}

#     candidates["Logistic Regression"] = (
#         Pipeline(
#             steps=[
#                 ("preprocessor", clone(preprocessor)),
#                 (
#                     "model",
#                     LogisticRegression(
#                         max_iter=1000,
#                         class_weight="balanced",
#                         solver="liblinear",
#                         random_state=RANDOM_STATE,
#                     ),
#                 ),
#             ]
#         ),
#         {
#             "model__C": [0.1, 1.0, 3.0, 10.0],
#         },
#     )

#     candidates["Decision Tree"] = (
#         Pipeline(
#             steps=[
#                 ("preprocessor", clone(preprocessor)),
#                 (
#                     "model",
#                     DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
#                 ),
#             ]
#         ),
#         {
#             "model__max_depth": [4, 6, 8, 12, None],
#             "model__min_samples_split": [2, 10, 25, 50],
#             "model__min_samples_leaf": [1, 5, 10, 20],
#             "model__criterion": ["gini", "entropy"],
#         },
#     )

#     candidates["Random Forest"] = (
#         Pipeline(
#             steps=[
#                 ("preprocessor", clone(preprocessor)),
#                 (
#                     "model",
#                     RandomForestClassifier(
#                         n_estimators=300,
#                         class_weight="balanced_subsample",
#                         n_jobs=-1,
#                         random_state=RANDOM_STATE,
#                     ),
#                 ),
#             ]
#         ),
#         {
#             "model__n_estimators": [200, 300, 400],
#             "model__max_depth": [8, 12, 16, None],
#             "model__min_samples_split": [2, 10, 25],
#             "model__min_samples_leaf": [1, 5, 10],
#             "model__max_features": ["sqrt", "log2", None],
#         },
#     )

#     candidates["XGBoost"] = (
#         Pipeline(
#             steps=[
#                 ("preprocessor", clone(preprocessor)),
#                 (
#                     "model",
#                     XGBClassifier(
#                         objective="binary:logistic",
#                         eval_metric="auc",
#                         scale_pos_weight=pos_weight,
#                         random_state=RANDOM_STATE,
#                         n_jobs=-1,
#                     ),
#                 ),
#             ]
#         ),
#         {
#             "model__n_estimators": [200, 300, 400],
#             "model__max_depth": [3, 4, 6],
#             "model__learning_rate": [0.03, 0.05, 0.1],
#             "model__subsample": [0.8, 1.0],
#             "model__colsample_bytree": [0.8, 1.0],
#             "model__min_child_weight": [1, 3, 5],
#         },
#     )

#     return candidates


# def choose_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, dict[str, float]]:
#     candidates = np.linspace(0.1, 0.9, 81)
#     best_threshold = 0.5
#     best_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
#     best_score = -1.0

#     for threshold in candidates:
#         y_pred = (y_prob >= threshold).astype(int)
#         accuracy = accuracy_score(y_true, y_pred)
#         precision = precision_score(y_true, y_pred, zero_division=0)
#         recall = recall_score(y_true, y_pred, zero_division=0)
#         f1 = f1_score(y_true, y_pred, zero_division=0)
#         score = accuracy
#         if score > best_score:
#             best_score = score
#             best_threshold = float(threshold)
#             best_metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

#     return best_threshold, best_metrics


# def evaluate_model(
#     name: str,
#     estimator: Any,
#     X_valid: pd.DataFrame,
#     y_valid: pd.Series,
#     best_params: dict[str, Any],
# ) -> ModelResult:
#     y_prob = estimator.predict_proba(X_valid)[:, 1]
#     threshold, threshold_metrics = choose_threshold(y_valid, y_prob)
#     y_pred = (y_prob >= threshold).astype(int)

#     return ModelResult(
#         name=name,
#         threshold=threshold,
#         accuracy=float(threshold_metrics["accuracy"]),
#         precision=float(threshold_metrics["precision"]),
#         recall=float(threshold_metrics["recall"]),
#         f1=float(threshold_metrics["f1"]),
#         roc_auc=float(roc_auc_score(y_valid, y_prob)),
#         confusion_matrix=confusion_matrix(y_valid, y_pred).tolist(),
#         best_params=best_params,
#     )


# def train_and_select_model(
#     X_train: pd.DataFrame,
#     y_train: pd.Series,
#     X_valid: pd.DataFrame,
#     y_valid: pd.Series,
#     preprocessor: ColumnTransformer,
# ) -> tuple[Any, ModelResult, list[ModelResult]]:
#     class_counts = y_train.value_counts()
#     pos_weight = float(class_counts[0] / class_counts[1])
#     candidates = build_model_candidates(preprocessor, pos_weight)
#     cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

#     fitted_models: dict[str, Any] = {}
#     results: list[ModelResult] = []

#     for name, (pipeline, param_grid) in candidates.items():
#         search = RandomizedSearchCV(
#             estimator=pipeline,
#             param_distributions=param_grid,
#             n_iter=min(8, np.prod([len(v) for v in param_grid.values()])),
#             scoring="accuracy",
#             cv=cv,
#             n_jobs=-1,
#             random_state=RANDOM_STATE,
#             verbose=1,
#         )
#         search.fit(X_train, y_train)
#         fitted_models[name] = search.best_estimator_
#         result = evaluate_model(name, search.best_estimator_, X_valid, y_valid, search.best_params_)
#         results.append(result)

#     results.sort(key=lambda item: (item.accuracy, item.roc_auc, item.f1), reverse=True)
#     best_result = results[0]
#     best_model = fitted_models[best_result.name]
#     return best_model, best_result, results


# def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, name: str) -> str:
#     fig, ax = plt.subplots(figsize=(6, 5))
#     ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues", colorbar=False)
#     ax.set_title(f"{name} Confusion Matrix")
#     path = PLOTS_DIR / f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
#     fig.tight_layout()
#     fig.savefig(path, dpi=200)
#     plt.close(fig)
#     return str(path)


# def save_roc_curve(y_true: pd.Series, y_prob: np.ndarray, name: str) -> str:
#     fpr, tpr, _ = roc_curve(y_true, y_prob)
#     auc_value = roc_auc_score(y_true, y_prob)
#     fig, ax = plt.subplots(figsize=(7, 5))
#     ax.plot(fpr, tpr, label=f"{name} ROC-AUC = {auc_value:.3f}")
#     ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
#     ax.set_xlabel("False Positive Rate")
#     ax.set_ylabel("True Positive Rate")
#     ax.set_title(f"{name} ROC Curve")
#     ax.legend(loc="lower right")
#     fig.tight_layout()
#     path = PLOTS_DIR / f"{name.lower().replace(' ', '_')}_roc_curve.png"
#     fig.savefig(path, dpi=200)
#     plt.close(fig)
#     return str(path)


# def extract_feature_importance(estimator: Any, X_sample: pd.DataFrame) -> pd.DataFrame:
#     if "preprocessor" not in estimator.named_steps:
#         raise ValueError("Estimator does not contain a preprocessor step.")

#     preprocessor = estimator.named_steps["preprocessor"]
#     model = estimator.named_steps["model"]
#     feature_names = preprocessor.get_feature_names_out()

#     if hasattr(model, "feature_importances_"):
#         importance = np.asarray(model.feature_importances_)
#     elif hasattr(model, "coef_"):
#         importance = np.abs(np.asarray(model.coef_)).reshape(-1)
#     else:
#         transformed = preprocessor.transform(X_sample)
#         explainer = shap.Explainer(model, transformed, feature_names=feature_names)
#         shap_values = explainer(transformed[: min(2000, transformed.shape[0])])
#         importance = np.abs(shap_values.values).mean(axis=0)

#     feature_importance = pd.DataFrame(
#         {"feature": feature_names, "importance": importance}
#     ).sort_values("importance", ascending=False)
#     return feature_importance


# def save_feature_importance_plot(feature_importance: pd.DataFrame, name: str) -> str:
#     top_features = feature_importance.head(15).iloc[::-1]
#     fig, ax = plt.subplots(figsize=(10, 7))
#     ax.barh(top_features["feature"], top_features["importance"], color="#156082")
#     ax.set_title(f"Top Feature Importance - {name}")
#     ax.set_xlabel("Importance")
#     fig.tight_layout()
#     path = PLOTS_DIR / f"{name.lower().replace(' ', '_')}_feature_importance.png"
#     fig.savefig(path, dpi=200)
#     plt.close(fig)
#     return str(path)


# def probability_to_score(probability: float) -> int:
#     return int(np.clip(round(850 - (probability * 550)), 300, 850))


# def generate_business_insights(df: pd.DataFrame, feature_importance: pd.DataFrame) -> dict[str, Any]:
#     risky = df[df["Default"] == 1]
#     safe = df[df["Default"] == 0]
#     insights = {
#         "default_rate": float(df["Default"].mean()),
#         "avg_credit_score_defaulted": float(risky["CreditScore"].mean()),
#         "avg_credit_score_non_defaulted": float(safe["CreditScore"].mean()),
#         "avg_interest_rate_defaulted": float(risky["InterestRate"].mean()),
#         "avg_interest_rate_non_defaulted": float(safe["InterestRate"].mean()),
#         "avg_dti_defaulted": float(risky["DTIRatio"].mean()),
#         "avg_dti_non_defaulted": float(safe["DTIRatio"].mean()),
#         "avg_loan_to_income_defaulted": float(risky["Loan_to_Income"].mean()),
#         "avg_loan_to_income_non_defaulted": float(safe["Loan_to_Income"].mean()),
#         "top_risk_features": feature_importance.head(10)["feature"].tolist(),
#     }
#     return insights


# def save_json(path: Path, payload: dict[str, Any]) -> None:
#     with path.open("w", encoding="utf-8") as file:
#         json.dump(payload, file, indent=2)


# def save_markdown_report(
#     dataset_shape: tuple[int, int],
#     class_balance: dict[int, int],
#     outliers: dict[str, dict[str, float]],
#     model_results: list[ModelResult],
#     best_validation_result: ModelResult,
#     test_result: ModelResult,
#     business_insights: dict[str, Any],
#     plot_paths: dict[str, str],
# ) -> None:
#     lines = [
#         "# Loan Default Risk Modeling Report",
#         "",
#         "## Dataset Overview",
#         f"- Shape: {dataset_shape[0]} rows x {dataset_shape[1]} columns after dropping `LoanID`.",
#         f"- Default distribution: {class_balance}.",
#         "",
#         "## Outlier Summary",
#     ]
#     for col, stats in outliers.items():
#         lines.append(
#             f"- {col}: {stats['outlier_count']} outliers ({stats['outlier_pct']:.2f}%), "
#             f"IQR bounds [{stats['lower_bound']:.2f}, {stats['upper_bound']:.2f}]"
#         )

#     lines.extend(["", "## Model Comparison"])
#     for result in model_results:
#         lines.append(
#             f"- {result.name}: accuracy={result.accuracy:.3f}, recall={result.recall:.3f}, "
#             f"precision={result.precision:.3f}, f1={result.f1:.3f}, "
#             f"roc_auc={result.roc_auc:.3f}, threshold={result.threshold:.2f}"
#         )

#     lines.extend(
#         [
#             "",
#             "## Best Model",
#             f"- Selected model: {best_validation_result.name}",
#             f"- Validation performance: accuracy={best_validation_result.accuracy:.3f}, recall={best_validation_result.recall:.3f}, "
#             f"precision={best_validation_result.precision:.3f}, "
#             f"roc_auc={best_validation_result.roc_auc:.3f}.",
#             f"- Test performance: accuracy={test_result.accuracy:.3f}, recall={test_result.recall:.3f}, "
#             f"precision={test_result.precision:.3f}, roc_auc={test_result.roc_auc:.3f}.",
#             "- Justification: selected for highest validation accuracy while retaining solid ranking power on ROC-AUC.",
#             "",
#             "## Business Insights",
#             f"- Defaulted borrowers have lower average credit score ({business_insights['avg_credit_score_defaulted']:.1f}) "
#             f"than non-defaulted borrowers ({business_insights['avg_credit_score_non_defaulted']:.1f}).",
#             f"- Defaulted borrowers show higher average interest rate ({business_insights['avg_interest_rate_defaulted']:.2f}) "
#             f"and DTI ratio ({business_insights['avg_dti_defaulted']:.2f}).",
#             f"- Loan-to-income ratio is higher for defaulted customers "
#             f"({business_insights['avg_loan_to_income_defaulted']:.2f} vs "
#             f"{business_insights['avg_loan_to_income_non_defaulted']:.2f}).",
#             f"- Top risk features: {', '.join(business_insights['top_risk_features'][:8])}.",
#             "",
#             "## Key Plots",
#         ]
#     )
#     for label, plot_path in plot_paths.items():
#         lines.append(f"- {label}: `{plot_path}`")

#     (ARTIFACTS_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


# def main() -> None:
#     ensure_directories()

#     df = load_data(DATASET_PATH)
#     raw_dataset_shape = df.shape
#     df = engineer_features(df)

#     target = "Default"
#     categorical_cols, numeric_cols = infer_feature_types(df, target)
#     outliers = detect_outliers(df, ["Income", "LoanAmount", "DTIRatio"])
#     plot_paths = save_eda_plots(df)

#     X = df.drop(columns=[target])
#     y = df[target]

#     X_train_full, X_test, y_train_full, y_test = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
#     )
#     X_train, X_valid, y_train, y_valid = train_test_split(
#         X_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=RANDOM_STATE
#     )

#     preprocessor = make_preprocessor(categorical_cols, numeric_cols)
#     best_model, best_valid_result, model_results = train_and_select_model(
#         X_train, y_train, X_valid, y_valid, preprocessor
#     )

#     X_train_valid = pd.concat([X_train, X_valid], axis=0)
#     y_train_valid = pd.concat([y_train, y_valid], axis=0)
#     best_model.fit(X_train_valid, y_train_valid)

#     test_prob = best_model.predict_proba(X_test)[:, 1]
#     test_pred = (test_prob >= best_valid_result.threshold).astype(int)
#     test_metrics = ModelResult(
#         name=best_valid_result.name,
#         threshold=best_valid_result.threshold,
#         accuracy=float(accuracy_score(y_test, test_pred)),
#         precision=float(precision_score(y_test, test_pred, zero_division=0)),
#         recall=float(recall_score(y_test, test_pred, zero_division=0)),
#         f1=float(f1_score(y_test, test_pred, zero_division=0)),
#         roc_auc=float(roc_auc_score(y_test, test_prob)),
#         confusion_matrix=confusion_matrix(y_test, test_pred).tolist(),
#         best_params=best_valid_result.best_params,
#     )

#     feature_importance = extract_feature_importance(best_model, X_train_valid.sample(n=3000, random_state=RANDOM_STATE))
#     feature_importance_path = save_feature_importance_plot(feature_importance, best_valid_result.name)
#     plot_paths["feature_importance"] = feature_importance_path
#     plot_paths["confusion_matrix"] = save_confusion_matrix(y_test, test_pred, best_valid_result.name)
#     plot_paths["roc_curve"] = save_roc_curve(y_test, test_prob, best_valid_result.name)

#     scoring_metadata = {
#         "best_model_name": best_valid_result.name,
#         "threshold": best_valid_result.threshold,
#         "score_formula": "credit_risk_score = clip(round(850 - probability_of_default * 550), 300, 850)",
#         "example_scores": {
#             "low_risk_pd_0.05": probability_to_score(0.05),
#             "medium_risk_pd_0.30": probability_to_score(0.30),
#             "high_risk_pd_0.70": probability_to_score(0.70),
#         },
#     }

#     business_insights = generate_business_insights(df, feature_importance)

#     payload = {
#         "dataset": {
#             "path": str(DATASET_PATH),
#             "shape_after_drop_loanid": raw_dataset_shape,
#             "default_distribution": y.value_counts().to_dict(),
#             "default_rate": float(y.mean()),
#             "missing_values": df.isna().sum().to_dict(),
#             "outliers": outliers,
#         },
#         "validation_model_results": [asdict(item) for item in model_results],
#         "best_validation_model": asdict(best_valid_result),
#         "test_metrics": asdict(test_metrics),
#         "classification_report": classification_report(y_test, test_pred, output_dict=True),
#         "business_insights": business_insights,
#         "top_features": feature_importance.head(15).to_dict(orient="records"),
#         "scoring": scoring_metadata,
#         "plots": plot_paths,
#     }

#     joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
#     joblib.dump(
#         {
#             "best_model_name": best_valid_result.name,
#             "threshold": best_valid_result.threshold,
#             "top_features": feature_importance.head(15).to_dict(orient="records"),
#             "scoring": scoring_metadata,
#             "categorical_columns": categorical_cols,
#             "numeric_columns": numeric_cols,
#         },
#         MODELS_DIR / "model_metadata.joblib",
#     )
#     save_json(ARTIFACTS_DIR / "metrics.json", payload)
#     feature_importance.head(30).to_csv(ARTIFACTS_DIR / "feature_importance.csv", index=False)
#     if "Logistic Regression" in best_valid_result.name and hasattr(best_model.named_steps["model"], "coef_"):
#         coef_df = pd.DataFrame(
#             {
#                 "feature": best_model.named_steps["preprocessor"].get_feature_names_out(),
#                 "coefficient": best_model.named_steps["model"].coef_.ravel(),
#             }
#         ).sort_values("coefficient", ascending=False)
#         coef_df.to_csv(ARTIFACTS_DIR / "signed_coefficients.csv", index=False)
#     save_markdown_report(
#         dataset_shape=raw_dataset_shape,
#         class_balance=y.value_counts().to_dict(),
#         outliers=outliers,
#         model_results=model_results,
#         best_validation_result=best_valid_result,
#         test_result=test_metrics,
#         business_insights=business_insights,
#         plot_paths=plot_paths,
#     )

#     print(json.dumps({"best_model": best_valid_result.name, "test_metrics": asdict(test_metrics)}, indent=2))


# if __name__ == "__main__":
#     main()
