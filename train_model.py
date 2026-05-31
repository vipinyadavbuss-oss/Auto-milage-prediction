import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data(path="auto-mpg.csv"):
    df = pd.read_csv(path)
    df.replace("?", np.nan, inplace=True)
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["brand"] = df["car name"].apply(lambda x: x.split()[0])
    df["power_to_weight"] = df["horsepower"] / df["weight"]
    df["car_age"] = df["model year"].max() - df["model year"]

    brands = sorted(df["brand"].unique().tolist())

    numeric_df = df.select_dtypes(include=np.number)
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]

    df = pd.get_dummies(df, columns=["brand"], drop_first=True)

    X = df.drop(["mpg", "car name"], axis=1)
    y = df["mpg"]
    return X, y, df, brands


def train_and_save_model():
    X, y, df, brands = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(random_state=42, n_estimators=150))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("Test R2:", r2_score(y_test, y_pred))

    model_data = {
        "pipeline": pipeline,
        "feature_columns": X.columns.tolist(),
        "brands": brands,
        "max_model_year": int(df["model year"].max()),
    }

    with open("model.pkl", "wb") as f:
        pickle.dump(model_data, f)

    print("Saved model.pkl with", len(X.columns), "features.")


if __name__ == "__main__":
    train_and_save_model()
