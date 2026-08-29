# ============================================================
# GERMAN CREDIT SCORING - COMPLETE FINAL CODE
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)
# ============================================================
# 1. LOAD DATASET
# ============================================================
data_path = "data/statlog+german+credit+data/german.data"
df = pd.read_csv(
    data_path,
    sep=r"\s+",
    header=None
)
# ============================================================
# 2. ASSIGN COLUMN NAMES
# ============================================================
columns = [
    "checking_account",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "dependents",
    "telephone",
    "foreign_worker",
    "credit_risk"
]
df.columns = columns
# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================
# Credit amount per month
df["credit_per_month"] = (
    df["credit_amount"] / df["duration"]
)
# ============================================================
# 4. DEFINE FEATURES AND TARGET
# ============================================================
X = df.drop("credit_risk", axis=1)
# Original dataset:
# 1 = Good Credit
# 2 = Bad Credit
#
# We convert:
# 1 -> 0 = Good Credit
# 2 -> 1 = Bad Credit

y = df["credit_risk"].map({
    1: 0,
    2: 1
})
# ============================================================
# 5. IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# ============================================================
categorical_columns = X.select_dtypes(
    include=["object"]
).columns
numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print("\nDataset shape:", df.shape)
print("\nCategorical columns:")
print(list(categorical_columns))
print("\nNumerical columns:")
print(list(numerical_columns))
# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================
# IMPORTANT:
# Split BEFORE fitting the preprocessing object.
# This prevents data leakage from the test set.
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("\nTraining data shape:", X_train_raw.shape)
print("Testing data shape:", X_test_raw.shape)
# ============================================================
# 7. PREPROCESSING
# ============================================================
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)
# Fit ONLY on training data
X_train = preprocessor.fit_transform(X_train_raw)
# Transform test data using the already-fitted preprocessor
X_test = preprocessor.transform(X_test_raw)
print("\nProcessed training shape:", X_train.shape)
print("Processed testing shape:", X_test.shape)
# ============================================================
# 8. GET FEATURE NAMES
# ============================================================
feature_names = preprocessor.get_feature_names_out()
print("\nNumber of processed features:", len(feature_names))
# ============================================================
# 9. LOGISTIC REGRESSION
# ============================================================
logistic_model = LogisticRegression(
    max_iter=5000,
    solver="liblinear",
    random_state=42
)
# Train
logistic_model.fit(X_train, y_train)
# Predictions
lr_pred = logistic_model.predict(X_test)
lr_prob = logistic_model.predict_proba(X_test)[:, 1]
# ============================================================
# 10. LOGISTIC REGRESSION EVALUATION
# ============================================================
lr_precision = precision_score(
    y_test,
    lr_pred,
    zero_division=0
)
lr_recall = recall_score(
    y_test,
    lr_pred,
    zero_division=0
)
lr_f1 = f1_score(
    y_test,
    lr_pred,
    zero_division=0
)
lr_roc_auc = roc_auc_score(
    y_test,
    lr_prob
)
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)
print("Precision:", round(lr_precision, 4))
print("Recall:", round(lr_recall, 4))
print("F1-Score:", round(lr_f1, 4))
print("ROC-AUC:", round(lr_roc_auc, 4))
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        lr_pred,
        target_names=["Good Credit", "Bad Credit"],
        zero_division=0
    )
)
# ============================================================
# 11. DECISION TREE
# ============================================================
decision_tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)
# Train
decision_tree.fit(X_train, y_train)
# Predictions
dt_pred = decision_tree.predict(X_test)
dt_prob = decision_tree.predict_proba(X_test)[:, 1]
# ============================================================
# 12. DECISION TREE EVALUATION
# ============================================================
dt_precision = precision_score(
    y_test,
    dt_pred,
    zero_division=0
)
dt_recall = recall_score(
    y_test,
    dt_pred,
    zero_division=0
)
dt_f1 = f1_score(
    y_test,
    dt_pred,
    zero_division=0
)
dt_roc_auc = roc_auc_score(
    y_test,
    dt_prob
)
print("\n" + "=" * 60)
print("DECISION TREE RESULTS")
print("=" * 60)
print("Precision:", round(dt_precision, 4))
print("Recall:", round(dt_recall, 4))
print("F1-Score:", round(dt_f1, 4))
print("ROC-AUC:", round(dt_roc_auc, 4))
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        dt_pred,
        target_names=["Good Credit", "Bad Credit"],
        zero_division=0
    )
)
# ============================================================
# 13. RANDOM FOREST
# ============================================================
random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
# Train
random_forest.fit(X_train, y_train)
# Predictions
rf_pred = random_forest.predict(X_test)
rf_prob = random_forest.predict_proba(X_test)[:, 1]
# ============================================================
# 14. RANDOM FOREST EVALUATION
# ============================================================
rf_precision = precision_score(
    y_test,
    rf_pred,
    zero_division=0
)
rf_recall = recall_score(
    y_test,
    rf_pred,
    zero_division=0
)
rf_f1 = f1_score(
    y_test,
    rf_pred,
    zero_division=0
)
rf_roc_auc = roc_auc_score(
    y_test,
    rf_prob
)
print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)
print("Precision:", round(rf_precision, 4))
print("Recall:", round(rf_recall, 4))
print("F1-Score:", round(rf_f1, 4))
print("ROC-AUC:", round(rf_roc_auc, 4))
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_pred,
        target_names=["Good Credit", "Bad Credit"],
        zero_division=0
    )
)
# ============================================================
# 15. MODEL COMPARISON
# ============================================================
results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "Precision": [
        lr_precision,
        dt_precision,
        rf_precision
    ],

    "Recall": [
        lr_recall,
        dt_recall,
        rf_recall
    ],

    "F1-Score": [
        lr_f1,
        dt_f1,
        rf_f1
    ],

    "ROC-AUC": [
        lr_roc_auc,
        dt_roc_auc,
        rf_roc_auc
    ]
})
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(
    results.round(4).to_string(index=False)
)
# ============================================================
# 16. FIND BEST MODEL
# ============================================================
best_model_row = results.loc[
    results["ROC-AUC"].idxmax()
]
best_model_name = best_model_row["Model"]
best_roc_auc = best_model_row["ROC-AUC"]
# ============================================================
# 17. LOGISTIC REGRESSION CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(
    y_test,
    lr_pred
)
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION CONFUSION MATRIX")
print("=" * 60)
print(cm)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Good Credit", "Bad Credit"]
)
disp.plot()
plt.title(
    "Logistic Regression - Confusion Matrix"
)
plt.tight_layout()
plt.show()
# ============================================================
# 18. RANDOM FOREST CONFUSION MATRIX
# ============================================================
rf_cm = confusion_matrix(
    y_test,
    rf_pred
)
print("\nRandom Forest Confusion Matrix:")
print(rf_cm)
rf_disp = ConfusionMatrixDisplay(
    confusion_matrix=rf_cm,
    display_labels=["Good Credit", "Bad Credit"]
)
rf_disp.plot()
plt.title(
    "Random Forest - Confusion Matrix"
)
plt.tight_layout()
plt.show()
# ============================================================
# 19. ROC CURVE COMPARISON
# ============================================================
lr_fpr, lr_tpr, _ = roc_curve(
    y_test,
    lr_prob
)
dt_fpr, dt_tpr, _ = roc_curve(
    y_test,
    dt_prob
)
rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_prob
)
plt.figure(figsize=(8, 6))
plt.plot(
    lr_fpr,
    lr_tpr,
    label=f"Logistic Regression (AUC = {lr_roc_auc:.2f})"
)
plt.plot(
    dt_fpr,
    dt_tpr,
    label=f"Decision Tree (AUC = {dt_roc_auc:.2f})"
)
plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC = {rf_roc_auc:.2f})"
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(
    "ROC Curve Comparison"
)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# ============================================================
# 20. LOGISTIC REGRESSION FEATURE COEFFICIENTS
# ============================================================
coefficients = logistic_model.coef_[0]
lr_feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})
lr_feature_importance["Absolute_Coefficient"] = (
    lr_feature_importance["Coefficient"].abs()
)
lr_feature_importance = lr_feature_importance.sort_values(
    "Absolute_Coefficient",
    ascending=False
)
print("\n" + "=" * 60)
print("TOP LOGISTIC REGRESSION FEATURES")
print("=" * 60)
print(
    lr_feature_importance[
        ["Feature", "Coefficient"]
    ].head(10).to_string(index=False)
)
# ============================================================
# 21. LOGISTIC REGRESSION TOP 10 FEATURE PLOT
# ============================================================
top_lr_features = (
    lr_feature_importance
    .head(10)
    .sort_values("Coefficient")
)
plt.figure(figsize=(10, 6))
plt.barh(
    top_lr_features["Feature"],
    top_lr_features["Coefficient"]
)
plt.axvline(
    x=0,
    linestyle="--"
)
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.title(
    "Top 10 Important Features - Logistic Regression"
)
plt.tight_layout()
plt.show()
# ============================================================
# 22. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================
rf_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": random_forest.feature_importances_
})
rf_importance = rf_importance.sort_values(
    "Importance",
    ascending=False
)
print("\n" + "=" * 60)
print("TOP RANDOM FOREST FEATURES")
print("=" * 60)
print(
    rf_importance.head(10).to_string(index=False)
)
# ============================================================
# 23. RANDOM FOREST TOP 10 FEATURE PLOT
# ============================================================
top_rf_features = (
    rf_importance
    .head(10)
    .sort_values("Importance")
)
plt.figure(figsize=(10, 6))
plt.barh(
    top_rf_features["Feature"],
    top_rf_features["Importance"]
)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title(
    "Top 10 Important Features - Random Forest"
)
plt.tight_layout()
plt.show()
# ============================================================
# 24. FINAL RESULT
# ============================================================
print("\n")
print("=" * 60)
print("FINAL RESULT")
print("=" * 60)
print(
    "Best model based on ROC-AUC:",
    best_model_name
)
print(
    "Best ROC-AUC:",
    round(best_roc_auc, 4)
)
print("\nCredit scoring model completed successfully.")
print("=" * 60)
