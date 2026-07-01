from database_connection import load_churn_data
from data_preprocessing import preprocess_data

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import os


# ==========================
# Load Data
# ==========================

df = load_churn_data()


# ==========================
# Preprocessing
# ==========================

df = preprocess_data(
    df,
    training=True
)

print("Data after preprocessing:")
print(df.head())


# ==========================
# Split Features
# ==========================

X = df.drop(
    "churn",
    axis=1
)

joblib.dump(
    X.columns.tolist(),
    "models/random_forest_features.pkl"
)

y = df["churn"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# ==========================
# Random Forest Pipeline
# ==========================

pipeline = Pipeline(
    [
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=10,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)



# ==========================
# Train Model
# ==========================

pipeline.fit(
    X_train,
    y_train
)



# ==========================
# Prediction
# ==========================

prediction = pipeline.predict(
    X_test
)



# ==========================
# Evaluation
# ==========================

print(
    classification_report(
        y_test,
        prediction
    )
)



# ROC-AUC

probability = pipeline.predict_proba(
    X_test
)[:,1]


roc_auc = roc_auc_score(
    y_test,
    probability
)


print(
    "ROC-AUC Score:",
    roc_auc
)



# Confusion Matrix

cm = confusion_matrix(
    y_test,
    prediction
)


print(
    "Confusion Matrix:"
)

print(cm)



plt.figure(
    figsize=(6,4)
)


sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)


plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    "Random Forest Confusion Matrix"
)

plt.show()



# ==========================
# Feature Importance
# ==========================


importance = pd.DataFrame(
    {
        "Feature":X.columns,
        "Importance":
        pipeline.named_steps["model"].feature_importances_
    }
)


importance = importance.sort_values(
    by="Importance",
    ascending=False
)


print("\nTop Features")

print(
    importance.head(15)
)



os.makedirs(
    "models",
    exist_ok=True
)


importance.to_csv(
    "models/random_forest_feature_importance.csv",
    index=False
)



# Save model

joblib.dump(
    pipeline,
    "models/random_forest_model.pkl"
)


print(
    "Random Forest model saved successfully"
)