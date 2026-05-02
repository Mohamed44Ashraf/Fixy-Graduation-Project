import pandas as pd
from config import FEATURES , scaler_path
from validator import clean_input, validate_input
import joblib

scaler = joblib.load(open(scaler_path, 'rb'))   

def preprocess_data(data):

    data = clean_input(data)

    errors = validate_input(data)

    if errors:
        return  errors

    # standardization
    data["service_type"] = data["service_type"].astype(str).str.strip().str.lower()
    data["weather_condition"] = data["weather_condition"].astype(str).str.strip().str.lower()
    data["time_of_day"] = data["time_of_day"].astype(str).str.strip().str.lower()
    data["day_of_week"] = data["day_of_week"].astype(str).str.strip().str.capitalize()

    # encoding
    categorical = ['service_type','day_of_week','weather_condition','time_of_day']
    data = pd.get_dummies(data, columns=categorical)

    # align features
    data = data.reindex(columns=FEATURES, fill_value=0)

    # scaling
    scaled = scaler.transform(data)

    return pd.DataFrame(scaled, columns=FEATURES)