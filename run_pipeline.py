#!/usr/bin/env python3
"""
Main pipeline script to run the complete hospital claims denial prediction pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Run the complete pipeline"""
    print("=" * 60)
    print("Hospital Claims Denial Prediction Pipeline")
    print("=" * 60)
    
    # Step 1: Data Processing
    print("\n[Step 1/3] Processing NPPES data...")
    from data_processing import main as process_data
    hospitals_df = process_data()
    
    # Step 2: Feature Engineering
    print("\n[Step 2/3] Engineering features...")
    from feature_engineering import main as engineer_features
    features_df = engineer_features()
    
    # Step 3: Model Training
    print("\n[Step 3/3] Training XGBoost model...")
    from model import main as train_model
    model, metrics = train_model()
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review model results in the 'results/' directory")
    print("2. Check saved model in the 'models/' directory")
    print("3. Explore data in the 'notebooks/' directory")
    print("\nTo use with real claims data:")
    print("- Replace synthetic target in feature_engineering.py")
    print("- Join NPPES data with actual claims denial data using NPI")
    print("- Retrain the model")

if __name__ == "__main__":
    main()
