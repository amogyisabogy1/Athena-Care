"""
XGBoost Model for Hospital Insurance Claims Denial Prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
import os

import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

DATA_PATH = Path(__file__).parent.parent / "data" / "processed"
MODELS_PATH = Path(__file__).parent.parent / "models"
RESULTS_PATH = Path(__file__).parent.parent / "results"

def load_features():
    """Load engineered features"""
    features_file = DATA_PATH / "hospitals_features.csv"
    
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}. Run feature_engineering.py first.")
    
    print(f"Loading features from {features_file}...")
    df = pd.read_csv(features_file, low_memory=False)
    print(f"Loaded {len(df):,} records with {len(df.columns)} features")
    
    return df

def prepare_data(df):
    """
    Prepare data for modeling
    """
    print("\nPreparing data for modeling...")
    
    # Separate features and target
    target_col = 'likely_denied'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    # Exclude non-feature columns (including target and potential leakage)
    exclude_cols = ['NPI', 'likely_denied', 'claim_denial_risk', 
                   'has_deactivation_history', 'is_deactivated', 'has_deactivation_date',
                   'is_reactivated']  # Perfect predictor of deactivation - data leakage
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    print(f"Features: {len(feature_cols)}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Target rate: {y.mean():.2%}")
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"\nCategorical columns: {categorical_cols}")
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Handle missing values
    print(f"\nMissing values per column:")
    missing = X.isnull().sum()
    print(missing[missing > 0])
    
    # Fill missing values with median for numeric, mode for categorical
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0)
    
    return X, y, feature_cols, label_encoders

class WandbCallback:
    """Custom XGBoost callback to log metrics to wandb"""
    def __init__(self, period=1):
        self.period = period
        self.iteration = 0
    
    def __call__(self, env):
        """Callback function called by XGBoost"""
        if not WANDB_AVAILABLE:
            return
        self.iteration = env.iteration
        if self.iteration % self.period == 0 and env.evaluation_result_list:
            metrics = {}
            for item in env.evaluation_result_list:
                # Format: (dataset, metric, value)
                dataset, metric, value = item
                metrics[f"{dataset}-{metric}"] = value
            if metrics:
                wandb.log(metrics, step=self.iteration)

def train_xgboost_model(X_train, y_train, X_val, y_val, use_wandb=True, project_name="hospital-claims-denial", 
                        use_smote=True, use_class_weights=True):
    """
    Train XGBoost model with hyperparameter tuning, checkpoint saving, and wandb logging
    Includes class imbalance handling via SMOTE and weighted loss
    """
    print("\n" + "=" * 60)
    print("Training XGBoost Model")
    print("=" * 60)
    
    # Calculate class imbalance ratio
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    imbalance_ratio = neg_count / pos_count if pos_count > 0 else 1
    print(f"\nClass imbalance: {pos_count} positive vs {neg_count} negative (ratio: 1:{imbalance_ratio:.1f})")
    
    # Apply SMOTE if requested
    if use_smote and pos_count > 0:
        print(f"\nApplying SMOTE to balance classes...")
        # Target: 5% positive class (similar to the paper's approach)
        target_ratio = 0.05
        smote = SMOTE(sampling_strategy=target_ratio, random_state=42, k_neighbors=min(5, pos_count-1))
        try:
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"  Before: {len(X_train)} samples ({y_train.sum()} positive)")
            print(f"  After SMOTE: {len(X_train_resampled)} samples ({y_train_resampled.sum()} positive)")
            X_train = X_train_resampled
            y_train = y_train_resampled
        except Exception as e:
            print(f"  SMOTE failed: {e}. Continuing without SMOTE...")
            use_smote = False
    
    # Calculate scale_pos_weight for XGBoost (equivalent to weighted loss)
    scale_pos_weight = imbalance_ratio if use_class_weights else 1.0
    if use_smote:
        # If we used SMOTE, reduce the weight since classes are more balanced
        scale_pos_weight = max(1.0, scale_pos_weight * 0.2)  # Reduce weight after SMOTE
    
    print(f"\nUsing scale_pos_weight: {scale_pos_weight:.2f} (penalizes false negatives {scale_pos_weight:.1f}x more)")
    
    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,  # Increased for better training
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': scale_pos_weight,  # Handle class imbalance
        'random_state': 42,
        'n_jobs': -1
    }
    
    print("Model parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Initialize wandb
    if use_wandb:
        if not WANDB_AVAILABLE:
            print("Warning: wandb not available. Install with: pip install wandb")
            use_wandb = False
        else:
            wandb.init(
                project=project_name,
                config=params,
                name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            wandb.config.update({
                'train_size': len(X_train),
                'val_size': len(X_val),
                'n_features': X_train.shape[1]
            })
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Setup checkpoint directory
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = MODELS_PATH / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.json"
    
    # Train model with early stopping and checkpoint
    evals = [(dtrain, 'train'), (dval, 'eval')]
    
    # Custom callback for wandb logging
    callbacks = []
    if use_wandb:
        callbacks.append(WandbCallback(period=1))
    
    # Store evaluation results for logging
    evals_result = {}
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=evals,
        early_stopping_rounds=20,
        verbose_eval=10,
        callbacks=callbacks if callbacks else None,
        evals_result=evals_result
    )
    
    # Log all training history to wandb
    if use_wandb and WANDB_AVAILABLE and evals_result:
        for dataset, metrics_dict in evals_result.items():
            for metric, values in metrics_dict.items():
                for i, value in enumerate(values):
                    wandb.log({f"{dataset}-{metric}": value}, step=i)
    
    # Save best checkpoint
    best_iteration = model.best_iteration
    best_score = model.best_score
    print(f"\nBest iteration: {best_iteration}, Best score: {best_score:.4f}")
    
    # Save the best model
    model.save_model(str(checkpoint_path))
    print(f"Saved best checkpoint to {checkpoint_path}")
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            'best_iteration': best_iteration,
            'best_eval_auc': best_score
        })
        # Save model artifact to wandb
        artifact = wandb.Artifact('xgboost-model', type='model')
        artifact.add_file(str(checkpoint_path))
        wandb.log_artifact(artifact)
    
    # Create sklearn wrapper with best iteration
    sklearn_params = params.copy()
    sklearn_params['n_estimators'] = best_iteration + 1  # +1 because iteration is 0-indexed
    
    sklearn_model = xgb.XGBClassifier(**sklearn_params)
    sklearn_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    return sklearn_model, model, best_iteration, best_score

def evaluate_model(model, X_train, y_train, X_test, y_test, use_wandb=True):
    """
    Evaluate model performance and log to wandb
    """
    print("\n" + "=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'train': {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, zero_division=0),
            'recall': recall_score(y_train, y_train_pred, zero_division=0),
            'f1': f1_score(y_train, y_train_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_train, y_train_proba),
            'pr_auc': average_precision_score(y_train, y_train_proba)
        },
        'test': {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_test_proba),
            'pr_auc': average_precision_score(y_test, y_test_proba)
        }
    }
    
    print("\nTraining Set Metrics:")
    for metric, value in metrics['train'].items():
        print(f"  {metric:12s}: {value:.4f}")
    
    print("\nTest Set Metrics:")
    for metric, value in metrics['test'].items():
        print(f"  {metric:12s}: {value:.4f}")
    
    # Log metrics to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb_metrics = {}
        for split in ['train', 'test']:
            for metric, value in metrics[split].items():
                wandb_metrics[f'{split}/{metric}'] = value
        wandb.log(wandb_metrics)
    
    # Classification report
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Confusion matrix
    print("\nTest Set Confusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    # Log confusion matrix to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_test,
                preds=y_test_pred,
                class_names=["Not Denied", "Denied"]
            )
        })
    
    return metrics, y_test_pred, y_test_proba

def plot_results(model, X_test, y_test, y_test_pred, y_test_proba, feature_cols, use_wandb=True):
    """
    Create visualization plots and log to wandb
    """
    print("\nGenerating visualizations...")
    
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Feature importance
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Log feature importance to wandb
    if use_wandb and WANDB_AVAILABLE:
        # Log top features as a table
        top_features = importance_df.head(20)
        wandb.log({
            "feature_importance": wandb.Table(
                dataframe=top_features[['feature', 'importance']]
            )
        })
        
        # Log feature importance bar chart
        wandb.log({
            "feature_importance_chart": wandb.plot.bar(
                wandb.Table(
                    dataframe=top_features.head(15),
                    columns=["feature", "importance"]
                ),
                "feature",
                "importance",
                title="Top 15 Feature Importance"
            )
        })
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Feature importance plot
    top_20 = importance_df.head(20)
    axes[0, 0].barh(range(len(top_20)), top_20['importance'])
    axes[0, 0].set_yticks(range(len(top_20)))
    axes[0, 0].set_yticklabels(top_20['feature'])
    axes[0, 0].set_xlabel('Importance')
    axes[0, 0].set_title('Top 20 Feature Importance')
    axes[0, 0].invert_yaxis()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_auc = roc_auc_score(y_test, y_test_proba)
    axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Log ROC curve to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "roc_curve": wandb.plot.roc_curve(
                y_test, y_test_proba, 
                labels=["Not Denied", "Denied"]
            )
        })
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    pr_auc = average_precision_score(y_test, y_test_proba)
    axes[1, 0].plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
    axes[1, 0].set_xlabel('Recall')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision-Recall Curve')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Log PR curve to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "pr_curve": wandb.plot.pr_curve(
                y_test, y_test_proba,
                labels=["Not Denied", "Denied"]
            )
        })
    
    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    axes[1, 1].set_title('Confusion Matrix')
    
    plt.tight_layout()
    plot_file = RESULTS_PATH / f"model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_file}")
    
    # Log plot to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({"evaluation_plots": wandb.Image(str(plot_file))})
    
    plt.close()
    
    return importance_df

def save_model(model, label_encoders, feature_cols, metrics, best_iteration=None, best_score=None, use_wandb=True):
    """
    Save model and metadata
    """
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_file = MODELS_PATH / f"xgboost_model_{timestamp}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved model to {model_file}")
    
    # Save label encoders
    encoders_file = MODELS_PATH / f"label_encoders_{timestamp}.pkl"
    with open(encoders_file, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"Saved label encoders to {encoders_file}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'feature_cols': feature_cols,
        'metrics': metrics,
        'model_type': 'XGBoost',
        'n_features': len(feature_cols),
        'best_iteration': best_iteration,
        'best_score': best_score
    }
    
    metadata_file = MODELS_PATH / f"model_metadata_{timestamp}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved metadata to {metadata_file}")
    
    # Log model files to wandb
    if use_wandb and WANDB_AVAILABLE:
        artifact = wandb.Artifact('final-model', type='model')
        artifact.add_file(str(model_file))
        artifact.add_file(str(encoders_file))
        artifact.add_file(str(metadata_file))
        wandb.log_artifact(artifact)
    
    return model_file, encoders_file, metadata_file

def main(use_wandb=True, wandb_project="hospital-claims-denial", use_smote=True, use_class_weights=True):
    """
    Main modeling pipeline with wandb logging and checkpoint saving
    
    Args:
        use_wandb: Whether to use wandb for logging (default: True)
        wandb_project: Wandb project name (default: "hospital-claims-denial")
        use_smote: Whether to use SMOTE for oversampling (default: True)
        use_class_weights: Whether to use class weights in loss (default: True)
    """
    print("=" * 60)
    print("XGBoost Model Training Pipeline")
    print("=" * 60)
    
    # Initialize wandb (or skip if disabled)
    if use_wandb:
        if not WANDB_AVAILABLE:
            print("\nWarning: wandb not installed. Install with: pip install wandb")
            print("Continuing without wandb logging...")
            use_wandb = False
        else:
            # Check if wandb is configured
            wandb_api_key = os.getenv('WANDB_API_KEY')
            if not wandb_api_key:
                print("\nWarning: WANDB_API_KEY not found in environment.")
                print("You can either:")
                print("  1. Set WANDB_API_KEY environment variable")
                print("  2. Run 'wandb login' in terminal")
                print("  3. Set use_wandb=False to skip wandb logging")
                response = input("\nContinue without wandb? (y/n): ").lower()
                if response != 'y':
                    print("Exiting. Please configure wandb or set use_wandb=False")
                    return None, None
                use_wandb = False
    
    # Load features
    df = load_features()
    
    # Prepare data
    X, y, feature_cols, label_encoders = prepare_data(df)
    
    # Train-test split
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split training set for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Validation set: {len(X_val):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # Train model with checkpoint saving and class imbalance handling
    sklearn_model, xgb_model, best_iteration, best_score = train_xgboost_model(
        X_train, y_train, X_val, y_val, 
        use_wandb=use_wandb, 
        project_name=wandb_project,
        use_smote=use_smote,
        use_class_weights=use_class_weights
    )
    
    # Evaluate model
    metrics, y_test_pred, y_test_proba = evaluate_model(
        sklearn_model, X_train, y_train, X_test, y_test, use_wandb=use_wandb
    )
    
    # Plot results
    importance_df = plot_results(
        sklearn_model, X_test, y_test, y_test_pred, y_test_proba, 
        feature_cols, use_wandb=use_wandb
    )
    
    # Save model
    model_file, encoders_file, metadata_file = save_model(
        sklearn_model, label_encoders, feature_cols, metrics,
        best_iteration=best_iteration, best_score=best_score,
        use_wandb=use_wandb
    )
    
    print("\n" + "=" * 60)
    print("Model training complete!")
    print("=" * 60)
    print(f"\nModel saved to: {model_file}")
    print(f"Best checkpoint saved to: {MODELS_PATH / 'checkpoints' / 'best_model.json'}")
    print(f"Results saved to: {RESULTS_PATH}")
    
    if use_wandb and WANDB_AVAILABLE:
        print(f"\nWandb run: {wandb.run.url}")
        wandb.finish()
    
    return sklearn_model, metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train XGBoost model for hospital claims denial prediction')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='hospital-claims-denial', 
                       help='Wandb project name')
    parser.add_argument('--no-smote', action='store_true', help='Disable SMOTE oversampling')
    parser.add_argument('--no-class-weights', action='store_true', help='Disable class weights')
    args = parser.parse_args()
    
    model, metrics = main(
        use_wandb=not args.no_wandb, 
        wandb_project=args.wandb_project,
        use_smote=not args.no_smote,
        use_class_weights=not args.no_class_weights
    )
