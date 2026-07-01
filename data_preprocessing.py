import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib

def preprocess_data(df, training=False):

    df = df.copy()


    # Date feature

    df["signup_date"] = pd.to_datetime(df["signup_date"])

    df["tenure_days"] = (
        pd.Timestamp.today()
        -
        df["signup_date"]
    ).dt.days


    df.drop(
    [
        "customer_id",
        "signup_date"
    ],
    axis=1,
    errors="ignore",
    inplace=True
    )


    # Feature engineering
    df["engagement_score"] = (
        (df["watch_hours_last_30_days"] / 100)
        *
        df["content_completion_rate"]
    )

    df["payment_risk_score"] = (
        df["payment_failures"]
        /
        3
    )


    df["inactivity_score"] = (
        df["days_since_last_login"]
        /
        120
    )


    df["customer_health_score"] = (
        df["engagement_score"]
        -
        df["inactivity_score"]
        -
        df["payment_risk_score"]
    )


    categorical_cols = [
        "gender",
        "country",
        "subscription_plan",
        "payment_method",
        "favorite_genre"
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("\n", " ", regex=False)
                .str.strip()
            )


    if training:

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

        encoded = encoder.fit_transform(
            df[categorical_cols]
        )


        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=df.index
        )


        df.drop(
            categorical_cols,
            axis=1,
            inplace=True
        )


        df = pd.concat(
            [
                df,
                encoded_df
            ],
            axis=1
        )


        joblib.dump(
            encoder,
            "models/encoder.pkl"
        )


    else:

        encoder = joblib.load(
            "models/encoder.pkl"
        )


        encoded = encoder.transform(
            df[categorical_cols]
        )


        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(categorical_cols),
            index=df.index
        )


        df.drop(
            categorical_cols,
            axis=1,
            inplace=True
        )


        df = pd.concat(
            [
                df,
                encoded_df
            ],
            axis=1
        )


    return df