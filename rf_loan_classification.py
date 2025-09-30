## Random Forest Loan Classification MLflow Experiment
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, roc_curve, auc, classification_report

import warnings
warnings.filterwarnings('ignore')

## --------------------- Data Preparation ---------------------------- ##

def load_and_prepare_data(dataset_path):
    """Load and prepare the dataset for training"""
    df = pd.read_csv(dataset_path)
    
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Separate features and target
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    # Split to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def preprocess_data(X_train, X_test):
    """Scale continuous features"""
    # Define continuous features for scaling
    continuous_features = [
        'income_annum', 'loan_amount', 'cibil_score',
        'residential_assets_value', 'commercial_assets_value', 
        'luxury_assets_value', 'bank_asset_value'
    ]
    
    # Apply RobustScaler to continuous features
    scaler = RobustScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Only scale if continuous features exist in the dataset
    existing_continuous = [col for col in continuous_features if col in X_train.columns]
    if existing_continuous:
        X_train_scaled[existing_continuous] = scaler.fit_transform(X_train[existing_continuous])
        X_test_scaled[existing_continuous] = scaler.transform(X_test[existing_continuous])
    
    return X_train_scaled, X_test_scaled

def handle_imbalance(X_train, y_train, method='smote'):
    """Handle class imbalance using different methods"""
    
    if method == 'none':
        return X_train, y_train, None
    
    elif method == 'weights':
        # Calculate class weights for imbalanced data
        class_counts = np.bincount(y_train)
        vals_count = 1 - (class_counts / len(y_train))
        vals_count = vals_count / np.sum(vals_count)
        
        dict_weights = {}
        for i in range(len(class_counts)):
            dict_weights[i] = vals_count[i]
        
        print(f"Class weights: {dict_weights}")
        return X_train, y_train, dict_weights
    
    elif method == 'smote':
        # Using SMOTE for oversampling
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"Original training distribution: {np.bincount(y_train)}")
        print(f"SMOTE resampled distribution: {np.bincount(y_train_resampled)}")
        
        return X_train_resampled, y_train_resampled, None

def train_random_forest(X_train, y_train, X_test, y_test, experiment_name, 
                       n_estimators, max_depth, min_samples_split, min_samples_leaf, 
                       max_features, bootstrap, class_weight=None):
    """Train Random Forest model with MLflow tracking"""
    
    # Create experiment if it doesn't exist
    try:
        mlflow.set_experiment(experiment_name)
    except Exception:
        mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        # Set tags
        mlflow.set_tag('model_type', 'RandomForest')
        mlflow.set_tag('algorithm', 'Random Forest Classifier')
        
        # Model parameters
        rf_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'bootstrap': bootstrap,
            'random_state': 42
        }
        
        # Add class weights if provided
        if class_weight is not None:
            rf_params['class_weight'] = class_weight
            mlflow.set_tag('imbalance_handling', 'class_weights')
        elif len(X_train) != len(y_train):  # SMOTE was used
            mlflow.set_tag('imbalance_handling', 'smote')
        else:
            mlflow.set_tag('imbalance_handling', 'none')
        
        # Initialize and train model
        model = RandomForestClassifier(**rf_params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        train_f1 = f1_score(y_train, y_pred_train)
        
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test)
        
        # ROC AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Feature importance
        feature_names = X_train.columns.tolist()
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        
        # Log parameters
        mlflow.log_params(rf_params)
        
        # Log metrics
        mlflow.log_metrics({
            'train_accuracy': train_accuracy,
            'train_f1_score': train_f1,
            'test_accuracy': test_accuracy,
            'test_f1_score': test_f1,
            'roc_auc': roc_auc,
            'train_size': len(X_train),
            'test_size': len(X_test)
        })
        
        # Log feature importance
        for feature, importance in feature_importance.items():
            mlflow.log_metric(f'feature_importance_{feature}', importance)
        
        # Log the model
        mlflow.sklearn.log_model(
            model, 
            "RandomForest_Model",
            input_example=X_test.head(1),
            signature=mlflow.models.infer_signature(X_train, y_train)
        )
        
        # Plot and log confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred_test)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Approved', 'Approved'], 
                    yticklabels=['Not Approved', 'Approved'])
        plt.title(f'Random Forest Confusion Matrix\\nAccuracy: {test_accuracy:.4f}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
        plt.close()
        
        # Plot and log ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Random Forest ROC Curve')
        plt.legend(loc="lower right")
        
        mlflow.log_figure(plt.gcf(), "roc_curve.png")
        plt.close()
        
        # Plot and log feature importance
        plt.figure(figsize=(12, 8))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importance_values = zip(*sorted_features[:15])  # Top 15 features
        
        plt.barh(range(len(features)), importance_values)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importance - Random Forest')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        mlflow.log_figure(plt.gcf(), "feature_importance.png")
        plt.close()
        
        # Log classification report
        class_report = classification_report(y_test, y_pred_test, 
                                           target_names=['Not Approved', 'Approved'])
        mlflow.log_text(class_report, "classification_report.txt")
        
        # Print results
        print(f"\\n{'='*60}")
        print(f"Random Forest Results:")
        print(f"{'='*60}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Train F1-Score: {train_f1:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test F1-Score: {test_f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"{'='*60}")
        
        return model, test_accuracy, test_f1, roc_auc

def main():
    parser = argparse.ArgumentParser(description='Random Forest Loan Classification with MLflow')
    
    # Data parameters
    parser.add_argument('--dataset_path', type=str, default='new_class_df.csv',
                       help='Path to the dataset CSV file')
    parser.add_argument('--experiment_name', type=str, default='RF_Loan_Classification',
                       help='MLflow experiment name')
    
    # Random Forest parameters
    parser.add_argument('--n_estimators', type=int, default=100,
                       help='Number of trees in the forest')
    parser.add_argument('--max_depth', type=int, default=None,
                       help='Maximum depth of the tree (None for unlimited)')
    parser.add_argument('--min_samples_split', type=int, default=2,
                       help='Minimum samples required to split a node')
    parser.add_argument('--min_samples_leaf', type=int, default=1,
                       help='Minimum samples required at each leaf node')
    parser.add_argument('--max_features', type=str, default='sqrt',
                       help='Number of features to consider for best split (sqrt, log2, None, int, float)')
    parser.add_argument('--bootstrap', type=bool, default=True,
                       help='Whether bootstrap samples are used when building trees')
    
    # Imbalance handling
    parser.add_argument('--imbalance_method', type=str, default='smote',
                       choices=['none', 'weights', 'smote'],
                       help='Method to handle class imbalance')
    
    args = parser.parse_args()
    
    # Convert max_features parameter
    if args.max_features == 'None':
        args.max_features = None
    elif args.max_features.isdigit():
        args.max_features = int(args.max_features)
    elif args.max_features.replace('.', '').isdigit():
        args.max_features = float(args.max_features)
    
    # Convert max_depth parameter
    if args.max_depth == 0:
        args.max_depth = None
    
    print("="*80)
    print("RANDOM FOREST LOAN CLASSIFICATION WITH MLFLOW")
    print("="*80)
    print(f"Dataset: {args.dataset_path}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Imbalance Method: {args.imbalance_method}")
    print(f"RF Parameters:")
    print(f"  - n_estimators: {args.n_estimators}")
    print(f"  - max_depth: {args.max_depth}")
    print(f"  - min_samples_split: {args.min_samples_split}")
    print(f"  - min_samples_leaf: {args.min_samples_leaf}")
    print(f"  - max_features: {args.max_features}")
    print(f"  - bootstrap: {args.bootstrap}")
    print("="*80)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data(args.dataset_path)
    
    # Preprocess data
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)
    
    # Handle imbalance
    X_train_final, y_train_final, class_weights = handle_imbalance(
        X_train_scaled, y_train, method=args.imbalance_method
    )
    
    # Train model
    model, accuracy, f1, roc_auc = train_random_forest(
        X_train_final, y_train_final, X_test_scaled, y_test,
        args.experiment_name,
        args.n_estimators, args.max_depth, args.min_samples_split,
        args.min_samples_leaf, args.max_features, args.bootstrap,
        class_weight=class_weights
    )
    
    print(f"\\nTraining completed! Check MLflow UI for detailed results.")
    print(f"Run: mlflow ui")

if __name__ == '__main__':
    main()