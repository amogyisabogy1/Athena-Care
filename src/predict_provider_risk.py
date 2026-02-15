"""
Predict provider risk using trained model
Example script showing how to use the model for predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import sys

MODELS_PATH = Path(__file__).parent.parent / "models"
DATA_PATH = Path(__file__).parent.parent / "data" / "processed"

def load_model_and_encoders():
    """Load the most recent trained model"""
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
    
    print(f"Loaded model from {model_file}")
    return model, label_encoders

def predict_provider_risk(npi_list, model=None, label_encoders=None):
    """
    Predict risk for a list of provider NPIs
    
    Args:
        npi_list: List of NPI numbers (strings or integers)
        model: Trained model (if None, loads most recent)
        label_encoders: Label encoders (if None, loads from model)
    
    Returns:
        DataFrame with predictions
    """
    if model is None:
        model, label_encoders = load_model_and_encoders()
    
    # Load features
    features_file = DATA_PATH / "hospitals_features.csv"
    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_file}. Run feature_engineering.py first.")
    
    df = pd.read_csv(features_file, low_memory=False)
    
    # Filter to requested NPIs
    npi_list = [str(npi) for npi in npi_list]
    provider_df = df[df['NPI'].astype(str).isin(npi_list)].copy()
    
    if len(provider_df) == 0:
        print(f"Warning: No providers found for NPIs: {npi_list}")
        return pd.DataFrame()
    
    # Prepare features (same as in model.py)
    exclude_cols = ['NPI', 'likely_denied', 'claim_denial_risk']
    feature_cols = [col for col in provider_df.columns if col not in exclude_cols]
    
    X = provider_df[feature_cols].copy()
    
    # Apply label encoders
    for col in X.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            le = label_encoders[col]
            # Handle unseen categories
            X[col] = X[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )
        else:
            # Simple encoding if no encoder found
            X[col] = pd.Categorical(X[col]).codes
    
    # Fill missing values
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            X[col] = X[col].fillna(X[col].median() if len(X) > 0 else 0)
        else:
            X[col] = X[col].fillna(0)
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Create results dataframe
    results = pd.DataFrame({
        'NPI': provider_df['NPI'].values,
        'predicted_risk': probabilities,
        'predicted_class': predictions,
        'risk_level': pd.cut(probabilities, 
                           bins=[0, 0.3, 0.6, 1.0],
                           labels=['Low', 'Medium', 'High'])
    })
    
    # Add interpretation
    results['interpretation'] = results.apply(
        lambda row: f"{'High' if row['predicted_class'] == 1 else 'Low'} risk provider. "
                   f"Probability of issues: {row['predicted_risk']:.1%}",
        axis=1
    )
    
    return results

def main():
    """Example usage"""
    if len(sys.argv) < 2:
        print("Usage: python predict_provider_risk.py <NPI1> [NPI2] [NPI3] ...")
        print("\nExample:")
        print("  python predict_provider_risk.py 1234567890")
        print("  python predict_provider_risk.py 1234567890 9876543210 5555555555")
        sys.exit(1)
    
    npi_list = sys.argv[1:]
    
    print("=" * 60)
    print("Provider Risk Prediction")
    print("=" * 60)
    print(f"\nPredicting risk for {len(npi_list)} provider(s)...")
    
    results = predict_provider_risk(npi_list)
    
    if len(results) > 0:
        print("\n" + "=" * 60)
        print("Prediction Results")
        print("=" * 60)
        print(results.to_string(index=False))
        
        # Save results
        output_file = DATA_PATH / "predictions.csv"
        results.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
    else:
        print("\nNo predictions generated. Check that NPIs exist in the dataset.")

if __name__ == "__main__":
    main()
