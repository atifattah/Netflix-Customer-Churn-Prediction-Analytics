from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT", "3306")

if not all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME]):
    raise ValueError(
        "Missing one or more required database environment variables: "
        "DB_HOST, DB_USER, DB_PASSWORD, DB_NAME"
    )

DATABASE_URL = (
    f"mysql+pymysql://{quote_plus(DB_USER)}:{quote_plus(DB_PASSWORD)}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(
    DATABASE_URL
)


def get_next_customer_id():
    with engine.connect() as connection:
        result = connection.execute(text("SELECT customer_id FROM churn_predictions"))
        existing_ids = []

        for row in result.fetchall():
            value = row[0]
            if value is None:
                continue
            try:
                existing_ids.append(int(str(value).strip()))
            except ValueError:
                continue

        if existing_ids:
            return max(existing_ids) + 1

        return 1


def get_available_columns():
    with engine.connect() as connection:
        result = connection.execute(text("SHOW COLUMNS FROM churn_predictions"))
        return [row[0] for row in result.fetchall()]


def prepare_prediction_frame(data):
    df = pd.DataFrame([data])
    available_columns = get_available_columns()
    insertable_columns = [col for col in df.columns if col in available_columns]

    if not insertable_columns:
        raise ValueError("No matching columns found in churn_predictions table")

    return df[insertable_columns]


def save_prediction(data):

    df = prepare_prediction_frame(data)

    df.to_sql(
        "churn_predictions",
        con=engine,
        if_exists="append",
        index=False
    )

    print("Prediction saved successfully")