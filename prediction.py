import pandas as pd
import joblib

from data_preprocessing import preprocess_data

# Load trained model
MODEL_PATH = "models/random_forest_model.pkl"

model = joblib.load(MODEL_PATH)

print("Netflix Churn Prediction System")
print("=" * 40)

# Example customer input
customer = {

    "customer_id": "CUST_TEST_001",
    "age": 45,
    "gender": "Male",
    "country": "USA",
    "signup_date": pd.to_datetime("2024-01-15"),
    "subscription_plan": "Premium",
    "monthly_fee": 649,
    "payment_method": "Credit Card",
    "auto_renewal": 1,
    "payment_failures": 2,
    "watch_hours_last_30_days": 25,
    "movies_watched": 10,
    "series_watched": 5,
    "episodes_completed": 25,
    "login_frequency": 8,
    "days_since_last_login": 75,
    "favorite_genre": "Drama",
    "average_content_rating": 4.2,
    "content_completion_rate": 0.35,
    "number_of_devices": 2,
    "customer_support_tickets": 6,
    "buffering_complaints": 5,
    "app_crashes_reported": 3
}

# Convert input into dataframe
customer_df = pd.DataFrame([customer])

# Add churn column because preprocessing expects it
customer_df["churn"] = 0

print("\nRaw Customer Data")
print(customer_df.T)

# Apply same preprocessing used during training
customer_df = preprocess_data(
    customer_df,
    training=False
)

print("\nAfter preprocessing")
print(customer_df.T)

# Load training feature columns
features = joblib.load(
    "models/random_forest_features.pkl"
)

# Add missing columns
for col in features:
    if col not in customer_df.columns:
        customer_df[col] = 0

# Remove extra columns
customer_df = customer_df[features]

print("\nFinal Model Input")
print(customer_df.T)

# Prediction probability
churn_probability = model.predict_proba(customer_df)[0][1]

# Prediction class
prediction = model.predict(customer_df)[0]

if prediction == 1:
    prediction_label = "Churn"
else:
    prediction_label = "Stay"

print("\nPrediction Result")
print("=" * 40)

print(
    f"Churn Probability: {round(churn_probability*100,2)}%"
)

if churn_probability >= 0.70:
    risk = "HIGH RISK"

    recommendation = (
        "Customer retention action required"
    )

elif churn_probability >= 0.40:
    risk = "MEDIUM RISK"
    recommendation = (
        "Monitor customer engagement"
    )

else:
    risk = "LOW RISK"

    recommendation = (
        "Customer is healthy"
    )

print("Risk Category:", risk)

print("Recommendation:", recommendation)

if prediction == 1:
    print("\nFinal Prediction: Customer will churn")
else:
    print("\nFinal Prediction: Customer will stay")