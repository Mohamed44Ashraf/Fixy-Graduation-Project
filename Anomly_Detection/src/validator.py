from config import  original_features




def clean_data(data):
    data = data.copy()
    data =data.replace(r'^\s*$', None, regex=True)

    return data

def validate_inputs(data):
    errors = {}

    # missing columns
    missing = [col for col in original_features if col not in data.columns]

    if missing:
        errors["missing_fields"] = missing


    # Empty fields 
    empty = [col for col in original_features if col in data.columns and data[col].isnull().any()]

    if empty:
        errors["empty_fields"] = empty


    return errors if errors else None