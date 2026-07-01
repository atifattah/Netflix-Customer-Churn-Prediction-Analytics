import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, confusion_matrix, classification_report)
from database_connection import load_churn_data
from data_preprocessing import preprocess_data

# Load data
df = load_churn_data()

df = preprocess_data(df)

X = df.drop("churn", axis=1)

y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load model
model = joblib.load("models/random_forest_model.pkl")

# Prediction
prediction = model.predict(X_test)
probability = model.predict_proba(X_test)[:,1]

# Classification Report
report = classification_report(
    y_test,
    prediction,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv("model_classification_report.csv")

print(report_df)

# Confusion Matrix
cm = confusion_matrix(
    y_test,
    prediction
)

plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# ROC Curve
fpr, tpr, threshold = roc_curve(
    y_test,
    probability
)

roc_auc = auc(
    fpr,
    tpr
)

print("\nROC-AUC Score:")
print(round(roc_auc,4))

plt.figure(figsize=(6,4))
plt.plot(
    fpr,
    tpr,
    label=f"AUC={roc_auc:.3f}"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.show()

# Feature Importance

# Get Random Forest model from pipeline
rf_model = model[-1]

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance":
    rf_model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

importance.to_csv(
    "feature_importance.csv",
    index=False
)

print("\nTop Features")
print(importance.head(15))