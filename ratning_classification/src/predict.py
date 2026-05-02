import pandas as pd 
from config import  model_path, finalRate_encoder_path
from preprocess import preprocess_data
import joblib
import numpy as np



model = joblib.load(open(model_path, 'rb'))
finalRate_encoder = joblib.load(open(finalRate_encoder_path, 'rb'))


def predict_rating(data):
    # Preprocess the input data
    preprocessed_data = preprocess_data(data)
    
    # Make predictions using the loaded model
    y_pred_prob = model.predict(preprocessed_data)
    y_pred = np.argmax(y_pred_prob, axis=1)    

    # Decode the predicted ratings back to their original form
    decoded_predictions = finalRate_encoder.inverse_transform(y_pred)
    
    return decoded_predictions

