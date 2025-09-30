## Loan Classification MLflow Experiment Tracking
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os, argparse
from imblearn.over_sampling import SMOTE
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, roc_curve, auc, classification_report

import warnings
warnings.filterwarnings('ignore')

## --------------------- Data Preparation ---------------------------- ##

## Read the processed dataset
DATASET_PATH = r'D:\Downloads\new_class_df.csv'
df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

## Separate features and target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

## Split to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

## --------------------- Data Processing ---------------------------- ##

## Define continuous features for scaling (adapt based on your dataset)
continuous_features = [
    'income_annum', 'loan_amount', 'cibil_score',
    'residential_assets_value', 'commercial_assets_value', 
    'luxury_assets_value', 'bank_asset_value'
]

## Apply RobustScaler to continuous features
scaler = RobustScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Only scale if continuous features exist in the dataset
existing_continuous = [col for col in continuous_features if col in X_train.columns]
if existing_continuous:
    X_train_scaled[existing_continuous] = scaler.fit_transform(X_train[existing_continuous])
    X_test_scaled[existing_continuous] = scaler.transform(X_test[existing_continuous])

## --------------------- Imbalancing Handling ---------------------------- ##

## 1. Calculate class weights for imbalanced data
class_counts = np.bincount(y_train)
vals_count = 1 - (class_counts / len(y_train))
vals_count = vals_count / np.sum(vals_count)

dict_weights = {}
for i in range(len(class_counts)):
    dict_weights[i] = vals_count[i]

print(f"Class weights: {dict_weights}")

## 2. Using SMOTE for oversampling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"Original training distribution: {np.bincount(y_train)}")
print(f"SMOTE resampled distribution: {np.bincount(y_train_resampled)}")

## --------------------- Modeling ---------------------------- ##

def train_model(X_train, y_train, X_test, y_test, model_type, plot_name, **model_params):
    
    # Create experiment if it doesn't exist
    try:
        mlflow.set_experiment('loan-classification-new')
    except Exception:
        mlflow.create_experiment('loan-classification-new')
        mlflow.set_experiment('loan-classification-new')
    
    with mlflow.start_run() as run:
        mlflow.set_tag('model_type', model_type)
        mlflow.set_tag('experiment_type', plot_name)

        # Initialize model based on type
        if model_type == 'RandomForest':
            model = RandomForestClassifier(random_state=42, **model_params)
        elif model_type == 'LogisticRegression':
            model = LogisticRegression(random_state=42, max_iter=1000, **model_params)
        elif model_type == 'SVM':
            model = SVC(random_state=42, probability=True, **model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Train the model
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred_test

        # Calculate metrics
        f1_test = f1_score(y_test, y_pred_test)
        acc_test = accuracy_score(y_test, y_pred_test)

        # Log parameters and metrics
        mlflow.log_params(model_params)
        mlflow.log_params({'model_type': model_type})
        mlflow.log_metrics({
            'accuracy': acc_test, 
            'f1_score': f1_test,
            'train_size': len(X_train),
            'test_size': len(X_test)
        })

        # Log the model
        mlflow.sklearn.log_model(model, f'{model_type}/{plot_name}')

        # Plot and log confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred_test)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Approved', 'Approved'], 
                    yticklabels=['Not Approved', 'Approved'])
        plt.title(f'Confusion Matrix - {model_type} ({plot_name})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        conf_matrix_fig = plt.gcf()
        mlflow.log_figure(figure=conf_matrix_fig, artifact_file=f'{plot_name}_confusion_matrix.png')
        plt.close()

        # Plot and log ROC curve
        if hasattr(model, "predict_proba") or model_type == 'SVM':
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_type} ({plot_name})')
            plt.legend(loc="lower right")
            
            roc_fig = plt.gcf()
            mlflow.log_figure(figure=roc_fig, artifact_file=f'{plot_name}_roc_curve.png')
            mlflow.log_metric('roc_auc', roc_auc)
            plt.close()

        # Log classification report as text artifact
        class_report = classification_report(y_test, y_pred_test, target_names=['Not Approved', 'Approved'])
        mlflow.log_text(class_report, f'{plot_name}_classification_report.txt')

        print(f"\n{model_type} - {plot_name}")
        print(f"Accuracy: {acc_test:.4f}")
        print(f"F1-Score: {f1_test:.4f}")
        if hasattr(model, "predict_proba") or model_type == 'SVM':
            print(f"ROC-AUC: {roc_auc:.4f}")

        return model, acc_test, f1_test


def main(n_estimators: int, max_depth: int, C: float):
    
    print("Starting MLflow experiment tracking for Loan Classification...")
    
    # Define model parameters for different experiments
    rf_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
    lr_params = {'C': C}
    svm_params = {'C': C}

    ## Experiment 1: Random Forest without handling imbalance
    print("\n" + "="*60)
    print("Experiment 1: Random Forest - No Imbalance Handling")
    print("="*60)
    train_model(X_train_scaled, y_train, X_test_scaled, y_test, 
                'RandomForest', 'RF_no_balance', **rf_params)

    ## Experiment 2: Random Forest with class weights
    print("\n" + "="*60)
    print("Experiment 2: Random Forest - Class Weights")
    print("="*60)
    rf_params_weighted = {**rf_params, 'class_weight': dict_weights}
    train_model(X_train_scaled, y_train, X_test_scaled, y_test, 
                'RandomForest', 'RF_class_weights', **rf_params_weighted)

    ## Experiment 3: Random Forest with SMOTE
    print("\n" + "="*60)
    print("Experiment 3: Random Forest - SMOTE Oversampling")
    print("="*60)
    train_model(X_train_resampled, y_train_resampled, X_test_scaled, y_test, 
                'RandomForest', 'RF_SMOTE', **rf_params)

    ## Experiment 4: Logistic Regression with SMOTE
    print("\n" + "="*60)
    print("Experiment 4: Logistic Regression - SMOTE Oversampling")
    print("="*60)
    train_model(X_train_resampled, y_train_resampled, X_test_scaled, y_test, 
                'LogisticRegression', 'LR_SMOTE', **lr_params)

    ## Experiment 5: SVM with SMOTE
    print("\n" + "="*60)
    print("Experiment 5: SVM - SMOTE Oversampling")
    print("="*60)
    train_model(X_train_resampled, y_train_resampled, X_test_scaled, y_test, 
                'SVM', 'SVM_SMOTE', **svm_params)

    print("\n" + "="*60)
    print("All experiments completed! Check MLflow UI for detailed results.")
    print("="*60)


if __name__ == '__main__':
    ## Take input from user via CLI using argparser library
    parser = argparse.ArgumentParser(description='Loan Classification MLflow Experiment')
    parser.add_argument('--n_estimators', '-n', type=int, default=100, 
                        help='Number of trees for Random Forest')
    parser.add_argument('--max_depth', '-d', type=int, default=10, 
                        help='Maximum depth for Random Forest')
    parser.add_argument('--C', '-c', type=float, default=1.0, 
                        help='Regularization parameter for Logistic Regression and SVM')
    
    args = parser.parse_args()

    ## Call the main function
    main(n_estimators=args.n_estimators, max_depth=args.max_depth, C=args.C)