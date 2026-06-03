from config import model_path
import joblib
from validator import clean_data, validate_inputs


model = joblib.load(model_path)


def predict_anomaly(data):
    data = clean_data(data)

    errors = validate_inputs(data)

    if errors:
        return errors
    

    data = data.reindex(columns=model.feature_names_in_, fill_value=0)
    predicted_anomaly = model.predict(data)
    
    return predicted_anomaly 
