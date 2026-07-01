import os
from pathlib import Path
from urllib.parse import quote_plus

import joblib
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from data_preprocessing import preprocess_data


# ----------------------------
# Database Connection
# ----------------------------

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT", "3306")


def build_database_url():
    required_values = {
        "DB_HOST": DB_HOST,
        "DB_USER": DB_USER,
        "DB_PASSWORD": DB_PASSWORD,
        "DB_NAME": DB_NAME,
    }

    missing = [name for name, value in required_values.items() if not value]
    if missing:
        raise ValueError(
            "Missing required database environment variables: " + ", ".join(missing)
        )

    return (
        "mysql+pymysql://"
        f"{quote_plus(DB_USER)}:"
        f"{quote_plus(DB_PASSWORD)}@"
        f"{DB_HOST}:"
        f"{DB_PORT}/"
        f"{DB_NAME}"
    )


DATABASE_URL = build_database_url()
engine = create_engine(DATABASE_URL)


# ----------------------------
# Load Model
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "models" / "random_forest_model.pkl")
features = joblib.load(BASE_DIR / "models" / "random_forest_features.pkl")


# ----------------------------
# Load Customer Data
# ----------------------------

query = """
SELECT
    c.customer_id,
    c.age,
    c.gender,
    c.country,
    c.signup_date,

    s.subscription_plan,
    s.monthly_fee,
    s.payment_method,
    s.auto_renewal,
    s.payment_failures,

    u.watch_hours_last_30_days,
    u.movies_watched,
    u.series_watched,
    u.episodes_completed,
    u.login_frequency,
    u.days_since_last_login,
    u.favorite_genre,
    u.average_content_rating,
    u.content_completion_rate,
    u.number_of_devices
FROM customers c
LEFT JOIN subscriptions s
    ON c.customer_id = s.customer_id
LEFT JOIN user_engagement u
    ON c.customer_id = u.customer_id
"""


df = pd.read_sql(query, engine)
print("Customers loaded:", len(df))

if df.empty:
    raise ValueError("No customer data was returned from the database.")


# ----------------------------
# Feature Engineering
# ----------------------------

processed_df = preprocess_data(df)

for col in features:
    if col not in processed_df.columns:
        processed_df[col] = 0

X = processed_df[features].fillna(0)


# ----------------------------
# Prediction
# ----------------------------

probabilities = model.predict_proba(X)[:, 1]
predictions = model.predict(X)


# ----------------------------
# Risk Category
# ----------------------------

risk = []
for p in probabilities:
    if p >= 0.7:
        risk.append("HIGH RISK")
    elif p >= 0.4:
        risk.append("MEDIUM RISK")
    else:
        risk.append("LOW RISK")


# ----------------------------
# Save Results
# ----------------------------

prediction_df = pd.DataFrame(
    {
        "customer_id": df["customer_id"],
        "churn_probability": probabilities,
        "risk_category": risk,
        "prediction": ["Churn" if x == 1 else "Stay" for x in predictions],
    }
)

prediction_df["engagement_score"] = processed_df["engagement_score"]
prediction_df["payment_risk_score"] = processed_df["payment_risk_score"]
prediction_df["inactivity_score"] = processed_df["inactivity_score"]
prediction_df["customer_health_score"] = processed_df["customer_health_score"]

df.to_sql(
    "churn_predictions",
    con=engine,
    if_exists="append",
    index=False
)

engine.dispose()

print("Prediction table created successfully")
print(prediction_df.head())