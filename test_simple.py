import requests
import json

# Test data for loan prediction
test_data = {
    "no_of_dependents": 2,
    "income_annum": 6700000,
    "loan_amount": 22700000,
    "loan_term": 18,
    "cibil_score": 538,
    "residential_assets_value": 15300000,
    "commercial_assets_value": 5800000,
    "luxury_assets_value": 20400000,
    "bank_asset_value": 6400000,
    "education_graduate": 0,
    "education_not_graduate": 1,
    "self_employed_no": 0,
    "self_employed_yes": 1
}

# Test the API
url = "http://127.0.0.1:8000/predict"
response = requests.post(url, json=test_data)

print("Response:", response.json())