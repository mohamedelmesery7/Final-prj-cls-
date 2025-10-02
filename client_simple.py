'''
Simple script for testing the loan classification model (similar to iris example)
'''

import json
import requests

# Sample loan applications (simplified format like iris)
loan_applications = [
    [2, 6700000, 22700000, 18, 538, 15300000, 5800000, 20400000, 6400000, 0, 1, 0, 1],
    [0, 4100000, 12200000, 8, 417, 2700000, 2200000, 8800000, 3300000, 0, 1, 0, 1],
    [3, 9100000, 29700000, 20, 506, 7100000, 4500000, 33300000, 12800000, 1, 0, 1, 0],
    [5, 9800000, 24200000, 20, 382, 12400000, 8200000, 29400000, 5000000, 0, 1, 0, 1],
]

# API endpoint
url = 'http://127.0.0.1:8000/predict'

predictions = []

for i, application in enumerate(loan_applications):
    # Create proper payload structure
    payload = {
        "no_of_dependents": application[0],
        "income_annum": application[1],
        "loan_amount": application[2], 
        "loan_term": application[3],
        "cibil_score": application[4],
        "residential_assets_value": application[5],
        "commercial_assets_value": application[6],
        "luxury_assets_value": application[7],
        "bank_asset_value": application[8],
        "education_graduate": application[9],
        "education_not_graduate": application[10],
        "self_employed_no": application[11],
        "self_employed_yes": application[12]
    }
    
    # Convert to JSON
    payload_json = json.dumps(payload)
    
    # Set headers
    headers = {'Content-Type': 'application/json'}
    
    # Make request
    response = requests.post(url, data=payload_json, headers=headers)
    
    # Extract prediction
    if response.status_code == 200:
        result = response.json()
        predictions.append(result['prediction'])
        print(f"Application {i+1}: {result['prediction']}")
    else:
        predictions.append("Error")
        print(f"Application {i+1}: Error {response.status_code}")

print("\nAll Predictions:", predictions)