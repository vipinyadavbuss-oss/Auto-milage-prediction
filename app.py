from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)

pipeline = model_data["pipeline"]
feature_columns = model_data["feature_columns"]
brands = model_data["brands"]
max_model_year = model_data["max_model_year"]


def build_input_dataframe(form):
    values = {
        "cylinders": int(form.get("cylinders", 4)),
        "displacement": float(form.get("displacement", 150)),
        "horsepower": float(form.get("horsepower", 100)),
        "weight": float(form.get("weight", 2800)),
        "acceleration": float(form.get("acceleration", 15)),
        "model_year": int(form.get("model_year", max_model_year)),
        "origin": int(form.get("origin", 1)),
        "brand": form.get("brand", brands[0]),
    }

    row = {
        "cylinders": values["cylinders"],
        "displacement": values["displacement"],
        "horsepower": values["horsepower"],
        "weight": values["weight"],
        "acceleration": values["acceleration"],
        "model year": values["model_year"],
        "origin": values["origin"],
        "brand": values["brand"],
    }

    df = pd.DataFrame([row])
    df["power_to_weight"] = df["horsepower"] / df["weight"]
    df["car_age"] = max_model_year - df["model year"]
    df = pd.get_dummies(df, columns=["brand"], prefix="brand", drop_first=True)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]
    return df


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        input_df = build_input_dataframe(request.form)
        prediction = float(pipeline.predict(input_df)[0])

    return render_template(
        "index.html",
        brands=brands,
        prediction=prediction,
    )


if __name__ == "__main__":
    app.run(debug=True)
