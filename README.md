## MLFLOW PROJECT
``` bash
# Get conda channels
conda config --show channels

# Build a MLFlow project, if you use one entry point with name (main)
mlflow run . --experiment-name <exp-name> # here it is {chrun-detection}

# If you have multiple entry points
mlflow run -e forest . --experiment-name churn-detection
mlflow run -e logistic . --experiment-name churn-detection
m
lflow run -e xgboost . --experiment-name churn-detection

# If you want some params instead of default values
mlflow run -e logistic . --experiment-name churn-detection -P c=3.5 -P p="l2"
mlflow run -e xgboost . --experiment-name churn-detection -P n=250 -P lr=0.15 -P d=22

```

```
## MLFLOW Models
``` bash
# serve the model via REST
mlflow models serve -m "path" --port 8000 --env-manager=local
mlflow models serve -m "file:///D:/Task%20day%205/mlruns/485748980600203179/ca5c7b67926647319f011cefc5c76374/artifacts/RandomForest/RF_no_balance" --port 8000 --env-manager=local

# it will open in this link
http://localhost:8000/invocations
```

``` python
# Example of loan classification data to be sent
{
    "dataframe_split": {
        "columns": [
            "no_of_dependents",
            "income_annum", 
            "loan_amount",
            "loan_term",
            "cibil_score",
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value",
            "education_ Graduate",
            "education_ Not Graduate", 
            "self_employed_ No",
            "self_employed_ Yes"
        ],
        "data": [
            [2, 6700000, 22700000, 18, 538, 15300000, 5800000, 20400000, 6400000, 0, 1, 0, 1]
        ]
    }
}


## Multiple loan samples
{
    "dataframe_split": {
        "columns": [
            "no_of_dependents",
            "income_annum", 
            "loan_amount",
            "loan_term",
            "cibil_score",
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value",
            "education_ Graduate",
            "education_ Not Graduate", 
            "self_employed_ No",
            "self_employed_ Yes"
        ],
        "data": [
            [2, 6700000, 22700000, 18, 538, 15300000, 5800000, 20400000, 6400000, 0, 1, 0, 1],
            [0, 4100000, 12200000, 8, 417, 2700000, 2200000, 8800000, 3300000, 0, 1, 0, 1],
            [3, 9100000, 29700000, 20, 506, 7100000, 4500000, 33300000, 12800000, 1, 0, 1, 0],
            [5, 9800000, 24200000, 20, 382, 12400000, 8200000, 29400000, 5000000, 0, 1, 0, 1]
        ]
    }
}
```

``` bash 
# if you want to use curl for loan classification

curl -X POST \
  http://localhost:8000/invocations \
  -H 'Content-Type: application/json' \
  -d '{
    "dataframe_split": {
        "columns": [
            "no_of_dependents",
            "income_annum", 
            "loan_amount",
            "loan_term",
            "cibil_score",
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value",
            "education_ Graduate",
            "education_ Not Graduate", 
            "self_employed_ No",
            "self_employed_ Yes"
        ],
        "data": [
            [2, 6700000, 22700000, 18, 538, 15300000, 5800000, 20400000, 6400000, 0, 1, 0, 1],
            [0, 4100000, 12200000, 8, 417, 2700000, 2200000, 8800000, 3300000, 0, 1, 0, 1],
            [3, 9100000, 29700000, 20, 506, 7100000, 4500000, 33300000, 12800000, 1, 0, 1, 0]
        ]
    }
}'


# if you want to use Powershell for loan classification
Invoke-RestMethod -Uri "http://localhost:8000/invocations" -Method Post -Headers @{"Content-Type" = "application/json"} -Body '{
    "dataframe_split": {
        "columns": [
            "no_of_dependents",
            "income_annum", 
            "loan_amount",
            "loan_term",
            "cibil_score",
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value",
            "education_ Graduate",
            "education_ Not Graduate", 
            "self_employed_ No",
            "self_employed_ Yes"
        ],
        "data": [
            [2, 6700000, 22700000, 18, 538, 15300000, 5800000, 20400000, 6400000, 0, 1, 0, 1],
            [0, 4100000, 12200000, 8, 417, 2700000, 2200000, 8800000, 3300000, 0, 1, 0, 1],
            [3, 9100000, 29700000, 20, 506, 7100000, 4500000, 33300000, 12800000, 1, 0, 1, 0]
        ]
    }
}'

```

```