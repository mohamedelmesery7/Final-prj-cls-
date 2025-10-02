# 1. Library imports
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import pandas as pd

# 2. Pydantic model for loan data
class LoanData(BaseModel):
    no_of_dependents: int
    income_annum: float
    loan_amount: float
    loan_term: int
    cibil_score: int
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float
    education_graduate: int
    education_not_graduate: int
    self_employed_no: int
    self_employed_yes: int

# 3. Create the app object
app = FastAPI()

# 4. Load the pickle model and scaler (if available)
try:
    # Load model from pickle file
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully from model.pkl!")
    
    # Try to load scaler if available
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully!")
    except:
        scaler = None
        print("No scaler found, proceeding without scaling")
        
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None

# 5. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Loan Classification API'}

# 6. Route with a single parameter, returns the parameter within a message
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome ': f'{name}'}

# 7. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted loan status
@app.post('/predict')
def predict_loan(data: LoanData):
    if model is None:
        return {'error': 'Model not loaded'}
    
    try:
        # Convert pydantic model to dict
        data_dict = data.dict()
        
        # Create DataFrame with correct column names (matching training data)
        df = pd.DataFrame([{
            'no_of_dependents': data_dict['no_of_dependents'],
            'income_annum': data_dict['income_annum'],
            'loan_amount': data_dict['loan_amount'],
            'loan_term': data_dict['loan_term'],
            'cibil_score': data_dict['cibil_score'],
            'residential_assets_value': data_dict['residential_assets_value'],
            'commercial_assets_value': data_dict['commercial_assets_value'],
            'luxury_assets_value': data_dict['luxury_assets_value'],
            'bank_asset_value': data_dict['bank_asset_value'],
            'education_ Graduate': data_dict['education_graduate'],
            'education_ Not Graduate': data_dict['education_not_graduate'],
            'self_employed_ No': data_dict['self_employed_no'],
            'self_employed_ Yes': data_dict['self_employed_yes']
        }])
        
        # Apply scaling if scaler is available
        if scaler is not None:
            # Define continuous features for scaling
            continuous_features = [
                'income_annum', 'loan_amount', 'cibil_score',
                'residential_assets_value', 'commercial_assets_value', 
                'luxury_assets_value', 'bank_asset_value'
            ]
            
            # Scale only continuous features
            existing_continuous = [col for col in continuous_features if col in df.columns]
            if existing_continuous:
                df[existing_continuous] = scaler.transform(df[existing_continuous])
        
        # Make prediction
        prediction = model.predict(df)
        
        # Get probability if model supports it
        try:
            probability = model.predict_proba(df)[0][1]  # Probability of approval
        except:
            probability = float(prediction[0])  # Use prediction as probability
        
        if prediction[0] == 1:
            result = "Loan Approved"
        else:
            result = "Loan Rejected"
        
        return {
            'prediction': result,
            'prediction_value': int(prediction[0]),
            'probability': float(probability)
        }
        
    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}'}

# 8. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload