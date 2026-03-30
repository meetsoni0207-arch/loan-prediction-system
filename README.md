# Loan Default Risk Prediction

Production-style machine learning project for predicting loan default risk from the provided `loan_data.csv` dataset.

## What is included

- End-to-end preprocessing and feature engineering
- EDA plots and outlier inspection
- Model comparison across Logistic Regression, Decision Tree, Random Forest, and XGBoost
- Imbalance handling with class weights and recall-focused thresholding
- Recall-focused tuning and threshold selection
- Saved model artifacts and metrics
- Streamlit scoring app

## Project structure

```text
.
|-- app.py
|-- train.py
|-- requirements.txt
`-- artifacts/
    |-- metrics.json
    |-- report.md
    |-- feature_importance.csv
    |-- models/
    `-- plots/
```

## Run training

```powershell
python train.py
```

## Run the Streamlit app

```powershell
python -m streamlit run app.py
```
"# loan-prediction-system" 
