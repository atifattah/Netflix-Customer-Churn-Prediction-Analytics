from database_connection import load_churn_data
from data_preprocessing import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import os

# Load data
df = load_churn_data()

# Preprocess
df = preprocess_data(df)

print(df.dtypes)

# Split
X = df.drop("churn", axis=1)

y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=42
            )
        )
    ]
)

# Model
pipeline.fit(
    X_train,
    y_train
)

# ==============================
# Feature Importance Analysis
# ==============================

feature_importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Coefficient": pipeline.named_steps["model"].coef_[0]
    }
)

# Absolute impact of each feature
feature_importance["Impact"] = (
    abs(feature_importance["Coefficient"])
)


# Sort highest impact features first
feature_importance = (
    feature_importance
    .sort_values(
        by="Impact",
        ascending=False
    )
)


print("\nTop 15 Important Features:")
print(feature_importance.head(15))


# Save feature importance
feature_importance.to_csv(
    "models/logistic_regression_feature_importance.csv",
    index=False
)

# Prediction
prediction = pipeline.predict(X_test)

print(classification_report(y_test, prediction))

# ROC-AUC Score
probability = pipeline.predict_proba(X_test)[:,1]

roc_auc = roc_auc_score(y_test, probability)

print("ROC-AUC Score:", roc_auc)

# Confusion Matrix
cm = confusion_matrix(y_test, prediction)

print("Confusion Matrix: ")

print(cm)

plt.figure(figsize=(6,4))

sns.heatmap(cm, annot=True, fmt="d")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Logistic Regression Confusion Matrix")

plt.show()

os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(pipeline, "models/logistic_regression_pipeline.pkl")

print("Model saved successfully")