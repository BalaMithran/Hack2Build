from flask import Flask, jsonify, request
app = Flask(__name__)
churnmodel = None


@app.before_first_request
def init():
    """
    Load the model if it is available locally
    """
    global churnmodel
    # path = "/mnt/models"
    # path = "/"
    model_name = "classifier.pkl"

    if exists(f"{model_name}"):
        # print(f"Loading classifier from {path}")
        with open(f"{model_name}", "rb") as handle:
            churnmodel = load(handle)
            print("Model loaded successfully")
    else:
        raise FileNotFoundError(f"{model_name}")

    return None

@app.route('/sensor', methods=['POST'])
def add_income():
    global churnmodel
    inp = [dict(request.json)]
#     print(inp)
    df = pd.DataFrame(inp)
#     print(df)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["weekend"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    df = df.drop('timestamp',axis = 1)
    prediction = churnmodel.predict(df)
    print(prediction)
    return {"churn":int(prediction[0])}
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)