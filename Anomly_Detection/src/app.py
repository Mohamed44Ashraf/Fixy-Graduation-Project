from flask import Flask, jsonify , request 
import pandas as pd
from predict import predict_anomaly



app = Flask(__name__)



@app.route('/' , methods = ['Get'])
def home():
    return jsonify({"message" : "Welcome to Anomaly Detection API , Use POST /predict to get predictions"})


@app.route('/predict' , methods = ['POST'])
def predict():
    data = request.get_json()

    if data is None:
        return jsonify({"error": "No JSON data received"}), 400
    
    data = pd.DataFrame([data])

    predicted_anomly = predict_anomaly(data)

    return jsonify({"predicted_anomaly": int(predicted_anomly[0])})


if __name__=="__main__":
    app.run(debug=True)