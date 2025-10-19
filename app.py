import joblib
import pandas as pd
from flask import Flask, render_template, request
import os

MODEL_PATH = "models/best_model.joblib"
app = Flask(__name__)

if not os.path.exists(MODEL_PATH):
    model = None
    print("Train model first using train.py")
else:
    model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template("index.html", error="Model not found. Run train.py first.")
    
    input_data = request.form.to_dict()
    for k, v in input_data.items():
        try:
            input_data[k] = float(v) if "." in v else int(v)
        except:
            input_data[k] = v

    X = pd.DataFrame([input_data])
    try:
        pred = model.predict(X)[0]
        return render_template("index.html", prediction=f"{pred:,.2f}")
    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
