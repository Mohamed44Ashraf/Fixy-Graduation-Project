import pandas as pd
from predict import predict_rating
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running. Use POST /predict"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "No JSON data received"}), 400

        input_data = pd.DataFrame([data])
        predicted_rating = predict_rating(input_data)

        return jsonify({"predicted_rating": int(predicted_rating[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)