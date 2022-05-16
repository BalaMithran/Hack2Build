# -*- coding: utf-8 -*-
"""
Inference script that extends from the base infer interface
"""
from os.path import exists
from joblib import load
import numpy as np
import pandas as pd

from flask import Flask, jsonify, request

app = Flask(__name__)

churnmodel = None


@app.before_first_request
def init():
    """
    Load the model if it is available locally
    """
    global churnmodel
    path = "C:/Users/bmithran/Desktop/hack2build/output"
    model_name = "predict.pkl"

    if exists(f"{path}/{model_name}"):
        print(f"Loading classifier pipeline from {path}")
        with open(f"{path}/{model_name}", "rb") as handle:
            churnmodel = load(handle)
            print("Model loaded successfully")
    else:
        raise FileNotFoundError(f"{path}/{model_name}")

    return None


@app.route("/v1/models/{}:predict".format("sensormodel"), methods=["POST"])
def add_income():
    global churnmodel
    inp = [dict(request.json)]
    #     print(inp)
    df = pd.DataFrame(inp)
    #     print(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["weekend"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    df = df.drop("timestamp", axis=1)
    prediction = churnmodel.predict(df)
    #     print(prediction)
    return {"churn": int(prediction[0])}


if __name__ == "__main__":
    init()
    app.run(host="0.0.0.0", debug=True, port=9001)
