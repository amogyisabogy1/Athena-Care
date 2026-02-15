"""
Utility functions for the hospital claims denial prediction project
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

def load_model(model_path=None):
    """
    Load a trained XGBoost model
    
    Args:
        model_path: Path to model file. If None, loads most recent model.
    
    Returns:
        model, label_encoders, metadata
    """
    MODELS_PATH = Path(__file__).parent.parent / "models"
    
    if model_path is None:
        # Find most recent model
        model_files = list(MODELS_PATH.glob("xgboost_model_*.pkl"))
        if not model_files:
            raise FileNotFoundError("No model files found. Train the model first: python src/model.py")
        model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    
    model_path = Path(model_path)
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load encoders
    timestamp = model_path.stem.split("_")[-1]
    encoder_path = MODELS_PATH / f"label_encoders_{timestamp}.pkl"
    if encoder_path.exists():
        with open(encoder_path, 'rb') as f:
            label_encoders = pickle.load(f)
    else:
        label_encoders = {}
    
    # Load metadata
    metadata_path = MODELS_PATH / f"model_metadata_{timestamp}.json"
    import json
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    return model, label_encoders, metadata

def predict_denial_risk(npi_data, model=None, label_encoders=None, metadata=None):
    """
    Predict claim denial risk for hospital data
    
    Args:
        npi_data: DataFrame with NPI data (single or multiple rows)
        model: Trained XGBoost model (if None, loads most recent)
        label_encoders: Label encoders for categorical features
        metadata: Model metadata
    
    Returns:
        DataFrame with predictions and probabilities
    """
    if model is None:
        model, label_encoders, metadata = load_model()
    
    # This is a placeholder - would need to apply same feature engineering
    # as in feature_engineering.py
    raise NotImplementedError("This function requires feature engineering to be applied first")

def get_feature_importance(model, feature_names, top_n=20):
    """
    Get feature importance from trained model
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        top_n: Number of top features to return
    
    Returns:
        DataFrame with feature importance
    """
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return importance_df.head(top_n)
