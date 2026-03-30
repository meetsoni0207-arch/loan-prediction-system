# Loan Default Risk Modeling Report

## Dataset Overview
- Shape: 255347 rows x 17 columns after dropping `LoanID`.
- Default distribution: {0: 225694, 1: 29653}.

## Outlier Summary
- Income: 0 outliers (0.00%), IQR bounds [-52264.75, 217309.25]
- LoanAmount: 0 outliers (0.00%), IQR bounds [-118087.50, 373228.50]
- DTIRatio: 0 outliers (0.00%), IQR bounds [-0.30, 1.30]

## Model Comparison
- Logistic Regression: accuracy=0.886, recall=0.062, precision=0.612, f1=0.112, roc_auc=0.756, threshold=0.89
- XGBoost: accuracy=0.886, recall=0.041, precision=0.648, f1=0.076, roc_auc=0.747, threshold=0.90
- Decision Tree: accuracy=0.885, recall=0.042, precision=0.548, f1=0.078, roc_auc=0.724, threshold=0.85
- Random Forest: accuracy=0.885, recall=0.046, precision=0.537, f1=0.085, roc_auc=0.733, threshold=0.77

## Best Model
- Selected model: Logistic Regression
- Validation performance: accuracy=0.886, recall=0.062, precision=0.612, roc_auc=0.756.
- Test performance: accuracy=0.887, recall=0.063, precision=0.619, roc_auc=0.762.
- Justification: selected for highest validation accuracy while retaining solid ranking power on ROC-AUC.

## Business Insights
- Defaulted borrowers have lower average credit score (559.3) than non-defaulted borrowers (576.2).
- Defaulted borrowers show higher average interest rate (15.90) and DTI ratio (0.51).
- Loan-to-income ratio is higher for defaulted customers (3.25 vs 2.04).
- Top risk features: num__Age, num__Loan_to_Income, num__InterestRate, num__MonthsEmployed, cat__EmploymentType_Full-time, cat__HasDependents_Yes, cat__HasCoSigner_Yes, cat__LoanPurpose_Home.

## Key Plots
- class_distribution: `C:\loan\artifacts\plots\class_distribution.png`
- credit_score_vs_default: `C:\loan\artifacts\plots\credit_score_vs_default.png`
- income_dti_interest_vs_default: `C:\loan\artifacts\plots\income_dti_interest_vs_default.png`
- correlation_heatmap: `C:\loan\artifacts\plots\correlation_heatmap.png`
- outlier_income: `C:\loan\artifacts\plots\outlier_income.png`
- outlier_loanamount: `C:\loan\artifacts\plots\outlier_loanamount.png`
- outlier_dtiratio: `C:\loan\artifacts\plots\outlier_dtiratio.png`
- feature_importance: `C:\loan\artifacts\plots\logistic_regression_feature_importance.png`
- confusion_matrix: `C:\loan\artifacts\plots\logistic_regression_confusion_matrix.png`
- roc_curve: `C:\loan\artifacts\plots\logistic_regression_roc_curve.png`