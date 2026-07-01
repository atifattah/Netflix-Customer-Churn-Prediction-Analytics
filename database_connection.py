import os
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()

# Read database credentials
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT", "3306")

from urllib.parse import quote_plus

# Create MySQL connection string
def get_connection_string():
    if not all([DB_HOST, DB_USER, DB_PASSWORD, DB_NAME]):
        raise ValueError(
            "Missing one or more required database environment variables: "
            "DB_HOST, DB_USER, DB_PASSWORD, DB_NAME"
        )

    return (
        "mysql+pymysql://"
        f"{quote_plus(DB_USER)}:"
        f"{quote_plus(DB_PASSWORD)}@"
        f"{DB_HOST}:"
        f"{DB_PORT}/"
        f"{DB_NAME}"
    )

# Create SQLAlchemy engine
engine = create_engine(
    get_connection_string(),
    echo=True
)

def test_connection():
    try:
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT DATABASE();")
            )

            database = result.fetchone()
            print(
                "Connected Successfully!"
            )
            print(
                "Current Database:",
                database[0]
            )

    except Exception as e:
        print(
            "Database connection failed:"
        )
        print(e)

def load_churn_data(query=None):

    if query is None:
        query = """
        SELECT *
        FROM netflix_churn_model;
        """

    df = pd.read_sql(
        query,
        engine
    )

    return df

if __name__ == "__main__":

    # Test SQL connection
    test_connection()

    # Load ML dataset
    df = load_churn_data()

    print("\nDataset Loaded Successfully")
    print("---------------------------")
    print(df.head())
    print("\nDataset Shape:")
    print(df.shape)