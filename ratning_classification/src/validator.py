import numpy as np
from config import REQUIRED_FIELDS


def clean_input(data):

    data = data.copy()
    data = data.replace(r'^\s*$', np.nan, regex=True)

    return data


def validate_input(data):

    errors = {}

    # missing columns
    missing = [col for col in REQUIRED_FIELDS if col not in data.columns]
    if missing:
        errors["missing_fields"] = missing

    # empty values (null + "")
    empty = [
        col for col in REQUIRED_FIELDS
        if col in data.columns and data[col].isnull().any()
    ]
    if empty:
        errors["empty_fields"] = empty

    return errors if errors else None