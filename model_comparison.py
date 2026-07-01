from database_connection import load_churn_data
from data_preprocessing import preprocess_data

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ----------------------------
# Load Data
# ----------------------------

print("Loading data...")

df = load_churn_data()


# ----------------------------
# Preprocessing
# ----------------------------

print("Preprocessing data...")

df = preprocess_data(df)


# ----------------------------
# Split Data
# ----------------------------

X = df.drop("churn", axis=1)

y = df["churn"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# ----------------------------
# Load Models
# ----------------------------

print("Loading trained models...")


logistic_model = joblib.load(
    "models/logistic_regression_pipeline.pkl"
)


random_forest_model = joblib.load(
    "models/random_forest_model.pkl"
)



# ----------------------------
# Prediction
# ----------------------------

models = {

    "Logistic Regression": logistic_model,

    "Random Forest": random_forest_model

}


results = []


for name, model in models.items():

    print("\nEvaluating:", name)


    prediction = model.predict(X_test)


    probability = model.predict_proba(X_test)[:,1]


    accuracy = accuracy_score(
        y_test,
        prediction
    )


    precision = precision_score(
        y_test,
        prediction
    )


    recall = recall_score(
        y_test,
        prediction
    )


    f1 = f1_score(
        y_test,
        prediction
    )


    roc_auc = roc_auc_score(
        y_test,
        probability
    )


    cm = confusion_matrix(
        y_test,
        prediction
    )


    results.append(
        {

            "Model": name,

            "Accuracy": round(accuracy,4),

            "Precision": round(precision,4),

            "Recall": round(recall,4),

            "F1 Score": round(f1,4),

            "ROC-AUC": round(roc_auc,4),

            "Confusion Matrix": cm

        }
    )



# ----------------------------
# Comparison Table
# ----------------------------


comparison_df = pd.DataFrame(results)


print("\n==============================")
print("MODEL PERFORMANCE COMPARISON")
print("==============================")

print(
    comparison_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]
    ]
)



# ----------------------------
# Save Results
# ----------------------------


comparison_df.to_csv(
    "model_comparison_results.csv",
    index=False
)


print("\nComparison saved successfully")



# ----------------------------
# Best Model
# ----------------------------


best_model = comparison_df.sort_values(
    by="ROC-AUC",
    ascending=False
).iloc[0]


print("\nBest Model Based on ROC-AUC:")

print(
    best_model["Model"]
)