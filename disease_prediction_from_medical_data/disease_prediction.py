# ============================================================
# TASK 4: DISEASE PREDICTION FROM MEDICAL DATA
# CodeAlpha Machine Learning Internship
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

print("=" * 65)
print("TASK 4: DISEASE PREDICTION FROM MEDICAL DATA")
print("=" * 65)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

print("\n1. Loading medical dataset...")

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("Dataset loaded successfully.")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])

# ------------------------------------------------------------
# 2. DATA PREPROCESSING
# ------------------------------------------------------------

print("\n2. Preparing data...")

# Check missing values
print("Missing values:", X.isnull().sum().sum())

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# 3. CREATE RANDOM FOREST MODEL
# ------------------------------------------------------------

print("\n3. Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed successfully.")

# ------------------------------------------------------------
# 4. MAKE PREDICTIONS
# ------------------------------------------------------------

print("\n4. Making predictions...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Predictions completed.")

# ------------------------------------------------------------
# 5. MODEL EVALUATION
# ------------------------------------------------------------

print("\n5. MODEL EVALUATION")
print("-" * 65)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names
    )
)

# ------------------------------------------------------------
# 6. CONFUSION MATRIX
# ------------------------------------------------------------

print("\n6. Creating confusion matrix...")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=data.target_names,
    yticklabels=data.target_names
)

plt.title("Disease Prediction - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

plt.savefig("disease_confusion_matrix.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 7. ROC CURVE
# ------------------------------------------------------------

print("Creating ROC curve...")

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Disease Prediction - ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("disease_roc_curve.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 8. FEATURE IMPORTANCE
# ------------------------------------------------------------

print("Creating feature importance graph...")

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"].head(10)[::-1],
    feature_importance["Importance"].head(10)[::-1]
)

plt.xlabel("Importance")
plt.ylabel("Medical Feature")
plt.title("Top 10 Important Medical Features")
plt.tight_layout()

plt.savefig("disease_feature_importance.png", dpi=300)
plt.show()

# ------------------------------------------------------------
# 9. SAVE TRAINED MODEL
# ------------------------------------------------------------

print("\nSaving trained model...")

joblib.dump(
    model,
    "disease_prediction_model.pkl"
)

print("Model saved as: disease_prediction_model.pkl")

# ------------------------------------------------------------
# 10. SAMPLE PREDICTIONS
# ------------------------------------------------------------

print("\n7. SAMPLE PREDICTIONS")
print("-" * 65)

sample_X = X_test.iloc[:10]
sample_y = y_test.iloc[:10]
sample_pred = model.predict(sample_X)

for i in range(len(sample_X)):
    actual = data.target_names[sample_y.iloc[i]]
    predicted = data.target_names[sample_pred[i]]

    print(
        f"Sample {i + 1}: "
        f"Actual = {actual}, "
        f"Predicted = {predicted}"
    )

# ------------------------------------------------------------
# 11. FINAL OUTPUT
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("TASK 4 COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nGenerated files:")
print("1. disease_prediction_model.pkl")
print("2. disease_confusion_matrix.png")
print("3. disease_roc_curve.png")
print("4. disease_feature_importance.png")

print("\nDisease prediction task completed successfully!")