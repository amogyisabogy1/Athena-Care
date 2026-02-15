"""
API Server for Hospital Risk Prediction Model
Provides REST API endpoints for the UI to use
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from model import load_features, prepare_data
from utils import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for UI

MODELS_PATH = Path(__file__).parent.parent / "models"
DATA_PATH = Path(__file__).parent.parent / "data" / "processed"

# Global variables for model and encoders
model = None
label_encoders = None
feature_cols = None

def load_model_for_api():
    """Load the most recent trained model"""
    global model, label_encoders, feature_cols
    
    try:
        model, label_encoders, metadata = load_model()
        
        if metadata and 'feature_cols' in metadata:
            feature_cols = metadata['feature_cols']
        else:
            # Load features to get column names
            df = load_features()
            _, _, feature_cols, _ = prepare_data(df)
        
        print("✓ Model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict risk for a provider by NPI
    
    Request body:
    {
        "npi": "1234567890"  # or array of NPIs
    }
    
    Returns:
    {
        "predictions": [
            {
                "npi": "1234567890",
                "predicted_risk": 0.75,
                "predicted_class": 1,
                "risk_level": "High",
                "interpretation": "..."
            }
        ]
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'npi' not in data:
            return jsonify({'error': 'Missing "npi" in request body'}), 400
        
        npi_list = data['npi']
        if isinstance(npi_list, str):
            npi_list = [npi_list]
        
        # Load features
        df = load_features()
        
        # Filter to requested NPIs
        npi_list = [str(npi) for npi in npi_list]
        provider_df = df[df['NPI'].astype(str).isin(npi_list)].copy()
        
        if len(provider_df) == 0:
            return jsonify({
                'error': f'No providers found for NPIs: {npi_list}',
                'predictions': []
            }), 404
        
        # Prepare features
        exclude_cols = ['NPI', 'likely_denied', 'claim_denial_risk', 
                       'has_deactivation_history', 'is_deactivated', 'has_deactivation_date',
                       'is_reactivated']
        available_feature_cols = [col for col in feature_cols if col not in exclude_cols and col in provider_df.columns]
        
        X = provider_df[available_feature_cols].copy()
        
        # Apply label encoders
        for col in X.select_dtypes(include=['object']).columns:
            if col in label_encoders:
                le = label_encoders[col]
                X[col] = X[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )
            else:
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
        
        # Create results
        results = []
        for idx, npi in enumerate(provider_df['NPI'].values):
            prob = float(probabilities[idx])
            pred = int(predictions[idx])
            
            # Determine risk level
            if prob > 0.6:
                risk_level = "High"
            elif prob > 0.3:
                risk_level = "Medium"
            else:
                risk_level = "Low"
            
            results.append({
                'npi': str(npi),
                'predicted_risk': round(prob, 4),
                'predicted_class': pred,
                'risk_level': risk_level,
                'interpretation': f"{risk_level} risk provider. Probability of issues: {prob:.1%}"
            })
        
        return jsonify({
            'predictions': results,
            'count': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict risk for multiple providers
    
    Request body:
    {
        "npis": ["1234567890", "9876543210", ...]
    }
    """
    data = request.get_json()
    
    if not data or 'npis' not in data:
        return jsonify({'error': 'Missing "npis" array in request body'}), 400
    
    # Use the same predict endpoint logic
    return predict()

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        _, _, metadata = load_model()
        
        return jsonify({
            'model_type': 'XGBoost',
            'features': len(feature_cols) if feature_cols else 0,
            'metadata': metadata
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/providers/search', methods=['GET'])
def search_providers():
    """
    Search for providers by name or NPI
    
    Query params:
    - q: search query (name or NPI)
    - limit: max results (default: 10)
    """
    query = request.args.get('q', '')
    limit = int(request.args.get('limit', 10))
    
    if not query:
        return jsonify({'error': 'Missing "q" query parameter'}), 400
    
    try:
        df = load_features()
        
        # Search by NPI or name
        if query.isdigit():
            # Search by NPI
            results = df[df['NPI'].astype(str).str.contains(query)].head(limit)
        else:
            # Search by name (if available)
            name_cols = [col for col in df.columns if 'name' in col.lower() or 'organization' in col.lower()]
            if name_cols:
                mask = df[name_cols[0]].astype(str).str.contains(query, case=False, na=False)
                results = df[mask].head(limit)
            else:
                results = df.head(0)
        
        providers = []
        for _, row in results.iterrows():
            providers.append({
                'npi': str(row['NPI']),
                'name': row.get('Provider Organization Name (Legal Business Name)', 'N/A') if 'Provider Organization Name (Legal Business Name)' in row else 'N/A'
            })
        
        return jsonify({
            'providers': providers,
            'count': len(providers)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Starting Hospital Risk Prediction API")
    print("=" * 60)
    
    # Load model on startup
    if load_model_for_api():
        print("\nStarting Flask server...")
        print("API endpoints:")
        print("  GET  /health - Health check")
        print("  POST /predict - Predict risk for provider(s)")
        print("  GET  /model/info - Model information")
        print("  GET  /providers/search - Search providers")
        
        # Get port from environment or default to 5000
        port = int(os.environ.get('PORT', 5000))
        host = os.environ.get('HOST', '0.0.0.0')
        
        print(f"\nServer running on: http://{host}:{port}")
        print("=" * 60)
        
        app.run(host=host, port=port, debug=False)
    else:
        print("Failed to load model. Please train the model first:")
        print("  python src/model.py --no-wandb")
        sys.exit(1)
