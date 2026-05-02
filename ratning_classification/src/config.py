import os

REQUIRED_FIELDS = [
    'service_type','day_of_week','time_of_day','weather_condition',
    'job_duration_minutes','response_time_minutes','punctuality_rating',
    'professionalism_rating','communication_rating','handling_quality',
    'cost_satisfaction','customer_behavior','clarity_of_issue',
    'safety_and_environment','customer_punctuality'
]


FEATURES = [
    'job_duration_minutes', 'response_time_minutes', 'punctuality_rating',
    'professionalism_rating', 'communication_rating', 'handling_quality',
    'cost_satisfaction', 'customer_behavior', 'clarity_of_issue',
    'safety_and_environment', 'customer_punctuality',

    'service_type_ac_repair', 'service_type_appliance_repair',
    'service_type_carpentry', 'service_type_cleaning',
    'service_type_delivery', 'service_type_electrical',
    'service_type_furniture_assembly', 'service_type_furniture_moving',
    'service_type_it_support', 'service_type_locksmith',
    'service_type_painting', 'service_type_pest_control',
    'service_type_plumbing', 'service_type_roof_leak_repair',
    'service_type_satellite_installation',

    'day_of_week_Friday', 'day_of_week_Monday', 'day_of_week_Saturday',
    'day_of_week_Sunday', 'day_of_week_Thursday', 'day_of_week_Tuesday',
    'day_of_week_Wednesday',

    'weather_condition_cloudy', 'weather_condition_hot',
    'weather_condition_rainy', 'weather_condition_stormy',
    'weather_condition_sunny',

    'time_of_day_afternoon', 'time_of_day_evening',
    'time_of_day_morning', 'time_of_day_night'
]


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


model_path = os.path.join(BASE_DIR, "models", "model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
finalRate_encoder_path = os.path.join(BASE_DIR, "models", "finalRate_encoder.pkl")


