"""
Utility functions for model loading and prediction
"""

import pickle
import json
from pathlib import Path

MODELS_PATH = Path(__file__).parent.parent / "models"

def load_model():
    """
    Load the most recent trained model and associated files
    
    Returns:
        tuple: (model, label_encoders, metadata)
    """
    # Find most recent model
    model_files = list(MODELS_PATH.glob("xgboost_model_*.pkl"))
    if not model_files:
        raise FileNotFoundError("No trained model found. Run model.py first.")
    
    model_file = max(model_files, key=lambda p: p.stat().st_mtime)
    timestamp = model_file.stem.split("_")[-1]
    
    # Load model
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    # Load encoders
    encoder_file = MODELS_PATH / f"label_encoders_{timestamp}.pkl"
    if encoder_file.exists():
        with open(encoder_file, 'rb') as f:
            label_encoders = pickle.load(f)
    else:
        label_encoders = {}
    
    # Load metadata
    metadata_file = MODELS_PATH / f"model_metadata_{timestamp}.json"
    metadata = None
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    print(f"Loaded model from {model_file}")
    return model, label_encoders, metadata
