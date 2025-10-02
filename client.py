'''
Script for testing the deployed loan classification model
'''

import json
import requests

# Sample loan data for testing
loan_data = [
    # Sample 1 - High income, good credit score
    {
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
    },
    # Sample 2 - Lower income, moderate credit
    {
        "no_of_dependents": 0,
        "income_annum": 4100000,
        "loan_amount": 12200000,
        "loan_term": 8,
        "cibil_score": 417,
        "residential_assets_value": 2700000,
        "commercial_assets_value": 2200000,
        "luxury_assets_value": 8800000,
        "bank_asset_value": 3300000,
        "education_graduate": 0,
        "education_not_graduate": 1,
        "self_employed_no": 0,
        "self_employed_yes": 1
    },
    # Sample 3 - High assets, good profile
    {
        "no_of_dependents": 3,
        "income_annum": 9100000,
        "loan_amount": 29700000,
        "loan_term": 20,
        "cibil_score": 506,
        "residential_assets_value": 7100000,
        "commercial_assets_value": 4500000,
        "luxury_assets_value": 33300000,
        "bank_asset_value": 12800000,
        "education_graduate": 1,
        "education_not_graduate": 0,
        "self_employed_no": 1,
        "self_employed_yes": 0
    },
    # Sample 4 - Low credit score
    {
        "no_of_dependents": 5,
        "income_annum": 9800000,
        "loan_amount": 24200000,
        "loan_term": 20,
        "cibil_score": 382,
        "residential_assets_value": 12400000,
        "commercial_assets_value": 8200000,
        "luxury_assets_value": 29400000,
        "bank_asset_value": 5000000,
        "education_graduate": 0,
        "education_not_graduate": 1,
        "self_employed_no": 0,
        "self_employed_yes": 1
    }
]

# API endpoint URL
url = 'http://127.0.0.1:8000/predict'

print("Testing Loan Classification API...")
print("=" * 50)

predictions = []
for i, loan in enumerate(loan_data):
    try:
        # Convert to JSON
        payload = json.dumps(loan)
        
        # Set headers for JSON content
        headers = {'Content-Type': 'application/json'}
        
        # Make prediction request
        response = requests.post(url, data=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            predictions.append(result)
            
            print(f"Sample {i+1}:")
            print(f"  Income: {loan['income_annum']:,}")
            print(f"  Loan Amount: {loan['loan_amount']:,}")
            print(f"  CIBIL Score: {loan['cibil_score']}")
            print(f"  Prediction: {result['prediction']}")
            if 'probability' in result:
                print(f"  Probability: {result['probability']:.3f}")
            print()
        else:
            print(f"Sample {i+1}: Error - {response.status_code}")
            print(f"Response: {response.text}")
            print()
            
    except Exception as e:
        print(f"Sample {i+1}: Exception - {e}")
        print()

print("Summary:")
print("-" * 30)
approved = sum(1 for p in predictions if 'Approved' in str(p.get('prediction', '')))
rejected = len(predictions) - approved
print(f"Total Samples: {len(predictions)}")
print(f"Approved: {approved}")
print(f"Rejected: {rejected}")

if predictions:
    print(f"Approval Rate: {approved/len(predictions)*100:.1f}%")

# Test batch prediction if available
print("\n" + "=" * 50)
print("Testing Batch Prediction...")

batch_url = 'http://127.0.0.1:8000/predict/batch'
batch_payload = {
    "applications": loan_data[:2]  # Test with first 2 samples
}

try:
    headers = {'Content-Type': 'application/json'}
    response = requests.post(batch_url, json=batch_payload, headers=headers)
    
    if response.status_code == 200:
        batch_result = response.json()
        print("Batch Prediction Success!")
        if 'summary' in batch_result:
            summary = batch_result['summary']
            print(f"Total Applications: {summary.get('total_applications', 'N/A')}")
            print(f"Approved: {summary.get('approved', 'N/A')}")
            print(f"Rejected: {summary.get('rejected', 'N/A')}")
            print(f"Approval Rate: {summary.get('approval_rate', 'N/A'):.2%}")
    else:
        print(f"Batch prediction failed: {response.status_code}")
        
except Exception as e:
    print(f"Batch prediction not available or error: {e}")

print("\nTesting completed!")