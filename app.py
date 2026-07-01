import streamlit as st
import pandas as pd
import joblib
import uuid

from data_preprocessing import preprocess_data
from database_insert import get_next_customer_id, save_prediction
from sqlalchemy import text
from database_connection import engine

# -------------------------------
# Load Model
# -------------------------------

model = joblib.load("models/random_forest_model.pkl")

features = joblib.load("models/random_forest_features.pkl")

# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Netflix Churn Prediction",
    page_icon="🎬",
    layout="wide"
)


st.title("🎬 Netflix Customer Churn Prediction")

st.write(
    """
    Machine Learning powered churn prediction system.
    
    Model:
    Random Forest Classifier
    
    Purpose:
    Identify customers who are likely to churn and recommend retention actions.
    """
)

# -------------------------------
# Customer Input
# -------------------------------

st.sidebar.header("Customer Information")

customer_id = None

age = st.sidebar.number_input(
    "Age",
    18,
    80,
    35
)

signup_date = st.sidebar.date_input(
    "Signup Date",
    value=pd.to_datetime("2024-01-15")
)

gender = st.sidebar.selectbox(
    "Gender",
    [
        "Male",
        "Female"
    ]
)

country = st.sidebar.selectbox(
    "Country",
    [
        "USA",
        "India",
        "UK",
        "Canada",
        "Germany",
        "France",
        "Australia"
    ]
)

subscription_plan = st.sidebar.selectbox(
    "Subscription Plan",
    [
        "Basic",
        "Standard",
        "Premium"
    ]
)

monthly_fee = st.sidebar.number_input(
    "Monthly Fee",
    0,
    2000,
    649
)

auto_renewal = st.sidebar.selectbox(
    "Auto Renewal",
    [
        0,
        1
    ]
)

payment_failures = st.sidebar.number_input(
    "Payment Failures",
    0,
    10,
    0
)

watch_hours = st.sidebar.number_input(
    "Watch Hours Last 30 Days",
    0,
    300,
    40
)

movies = st.sidebar.number_input(
    "Movies Watched",
    0,
    100,
    20
)

series = st.sidebar.number_input(
    "Series Watched",
    0,
    100,
    10
)

episodes = st.sidebar.number_input(
    "Episodes Completed",
    0,
    200,
    40
)

login_frequency = st.sidebar.number_input(
    "Login Frequency",
    0,
    100,
    20
)

days_since_login = st.sidebar.number_input(
    "Days Since Last Login",
    0,
    365,
    10
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Credit Card",
        "Debit Card",
        "UPI",
        "PayPal"
    ]
)

genre = st.sidebar.selectbox(
    "Favorite Genre",
    [
        "Action",
        "Comedy",
        "Drama",
        "Sci-Fi",
        "Thriller"
    ]
)

rating = st.sidebar.slider(
    "Average Content Rating",
    1.0,
    5.0,
    4.0
)

completion = st.sidebar.slider(
    "Content Completion Rate",
    0.0,
    1.0,
    0.8
)

devices = st.sidebar.number_input(
    "Number of Devices",
    1,
    10,
    2
)

tickets = st.sidebar.number_input(
    "Customer Support Tickets",
    0,
    20,
    0
)

buffering = st.sidebar.number_input(
    "Buffering Complaints",
    0,
    20,
    0
)

crashes = st.sidebar.number_input(
    "App Crashes",
    0,
    20,
    0
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Churn"):
    next_customer_id = get_next_customer_id()

    customer = {
        "customer_id": next_customer_id,
        "signup_date": pd.to_datetime(signup_date),
        "age": age,
        "gender": gender,
        "country": country,
        "subscription_plan": subscription_plan,
        "monthly_fee": monthly_fee,
        "auto_renewal": auto_renewal,
        "payment_failures": payment_failures,
        "watch_hours_last_30_days": watch_hours,
        "movies_watched": movies,
        "series_watched": series,
        "episodes_completed": episodes,
        "login_frequency": login_frequency,
        "days_since_last_login": days_since_login,
        "payment_method": payment_method,
        "favorite_genre": genre,
        "average_content_rating": rating,
        "content_completion_rate": completion,
        "number_of_devices": devices,
        "customer_support_tickets": tickets,
        "buffering_complaints": buffering,
        "app_crashes_reported": crashes,
        "churn":0
    }

    df = pd.DataFrame([customer])

    df = preprocess_data(df)

    for col in features:
        if col not in df.columns:
            df[col]=0

    df=df[features]

    # Model Prediction
    probability = model.predict_proba(df)[0][1]

    prediction = model.predict(df)[0]

    if prediction == 1:
        prediction_label = "Churn"
    else:
        prediction_label = "Stay"

    st.subheader("Prediction Result")

    st.metric("Churn Probability", f"{probability*100:.2f}%")

    risk_category = ""

    if probability >= 0.7:
        risk_category = "HIGH RISK"
        st.error(
            "HIGH RISK CUSTOMER"
        )
        st.write(
            "Recommendation: Offer retention benefits immediately."
        )

    elif probability >= 0.4:
        risk_category = "MEDIUM RISK"
        st.warning(
            "MEDIUM RISK CUSTOMER"
        )
        st.write(
            "Recommendation: Monitor engagement."
        )

    else:
        risk_category = "LOW RISK"
        st.success(
            "LOW RISK CUSTOMER"
        )
        st.write(
            "Recommendation: Customer is healthy."
        )

    # Save Prediction to Database
    prediction_record = {
        "prediction_date": pd.Timestamp.now(),
        "customer_id": next_customer_id,
        "age": age,
        "subscription_plan": subscription_plan,
        "monthly_fee": monthly_fee,
        "payment_failures": payment_failures,
        "watch_hours_last_30_days": watch_hours,
        "login_frequency": login_frequency,
        "days_since_last_login": days_since_login,
        "engagement_score":
            float(df["engagement_score"].iloc[0]),
        "payment_risk_score":
            float(df["payment_risk_score"].iloc[0]),
        "inactivity_score":
            float(df["inactivity_score"].iloc[0]),
        "customer_health_score":
            float(df["customer_health_score"].iloc[0]),
        "churn_probability":
            float(probability),
        "risk_category":
            risk_category,
        "prediction":
            prediction_label,
        "model_version":
            "RF_v1"
    }

    save_prediction(prediction_record)

    st.info("Prediction saved successfully to database.")

    st.subheader("Final Prediction")

    if prediction == 1:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is likely to stay")