"""
Load and prepare actual claims denial data for model training
This script shows how to integrate real claims data with NPPES data
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "processed"

def load_claims_data(claims_file_path):
    """
    Load claims data with denial information
    
    Expected columns in claims data:
    - NPI: Provider National Provider Identifier
    - claim_id: Unique claim identifier
    - claim_date: Date of claim submission
    - service_code: CPT/HCPCS code
    - claim_amount: Billed amount
    - denial_status: 0 = approved, 1 = denied
    - denial_reason: Reason code for denial (optional)
    - patient_id: Patient identifier (optional)
    
    Args:
        claims_file_path: Path to claims data CSV file
    
    Returns:
        DataFrame with claims data
    """
    print(f"Loading claims data from {claims_file_path}...")
    df = pd.read_csv(claims_file_path, low_memory=False)
    print(f"Loaded {len(df):,} claims")
    
    # Validate required columns
    required_cols = ['NPI', 'denial_status']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Ensure denial_status is binary
    if df['denial_status'].dtype != 'int64':
        df['denial_status'] = df['denial_status'].map({'approved': 0, 'denied': 1, 0: 0, 1: 1}).fillna(0).astype(int)
    
    return df

def aggregate_claims_by_provider(claims_df):
    """
    Aggregate claims data to provider level for training
    
    Creates features like:
    - Total claims count
    - Denial rate
    - Average claim amount
    - Most common denial reasons
    
    Args:
        claims_df: DataFrame with individual claims
    
    Returns:
        DataFrame aggregated by NPI
    """
    print("\nAggregating claims by provider...")
    
    # Group by NPI
    provider_stats = claims_df.groupby('NPI').agg({
        'denial_status': [
            'count',  # Total claims
            'sum',    # Total denials
            'mean'    # Denial rate
        ],
        'claim_amount': ['mean', 'std', 'sum'] if 'claim_amount' in claims_df.columns else [],
        'claim_date': ['min', 'max'] if 'claim_date' in claims_df.columns else []
    }).reset_index()
    
    # Flatten column names
    provider_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                             for col in provider_stats.columns.values]
    
    # Rename columns
    rename_dict = {
        'denial_status_count': 'total_claims',
        'denial_status_sum': 'total_denials',
        'denial_status_mean': 'denial_rate'
    }
    
    if 'claim_amount_mean' in provider_stats.columns:
        rename_dict.update({
            'claim_amount_mean': 'avg_claim_amount',
            'claim_amount_std': 'std_claim_amount',
            'claim_amount_sum': 'total_claim_amount'
        })
    
    provider_stats = provider_stats.rename(columns=rename_dict)
    
    # Calculate denial rate if not already calculated
    if 'denial_rate' not in provider_stats.columns:
        provider_stats['denial_rate'] = provider_stats['total_denials'] / provider_stats['total_claims']
    
    # Most common denial reason (if available)
    if 'denial_reason' in claims_df.columns:
        denial_reasons = claims_df[claims_df['denial_status'] == 1].groupby('NPI')['denial_reason'].apply(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else None
        ).reset_index()
        denial_reasons.columns = ['NPI', 'most_common_denial_reason']
        provider_stats = provider_stats.merge(denial_reasons, on='NPI', how='left')
    
    print(f"Aggregated to {len(provider_stats):,} providers")
    print(f"Average denial rate: {provider_stats['denial_rate'].mean():.2%}")
    
    return provider_stats

def merge_with_nppes_data(provider_claims_df, nppes_features_df):
    """
    Merge claims data with NPPES features
    
    Args:
        provider_claims_df: Aggregated claims data by provider
        nppes_features_df: NPPES features from feature_engineering.py
    
    Returns:
        Merged DataFrame ready for modeling
    """
    print("\nMerging claims data with NPPES features...")
    
    # Merge on NPI
    merged_df = nppes_features_df.merge(
        provider_claims_df,
        on='NPI',
        how='inner'  # Only keep providers with claims data
    )
    
    print(f"Merged dataset: {len(merged_df):,} providers")
    print(f"Providers with claims: {len(provider_claims_df):,}")
    print(f"Providers in NPPES: {len(nppes_features_df):,}")
    
    return merged_df

def create_target_variable(merged_df, threshold=0.1):
    """
    Create binary target variable from denial rate
    
    Args:
        merged_df: Merged DataFrame with denial_rate
        threshold: Denial rate threshold for classification (default: 10%)
    
    Returns:
        DataFrame with binary target variable
    """
    print(f"\nCreating target variable (threshold: {threshold:.1%})...")
    
    # Binary target: 1 if denial rate > threshold, 0 otherwise
    merged_df['likely_denied'] = (merged_df['denial_rate'] > threshold).astype(int)
    
    # Also keep continuous denial rate
    merged_df['claim_denial_risk'] = merged_df['denial_rate']
    
    print(f"Target distribution:")
    print(merged_df['likely_denied'].value_counts())
    print(f"Denial rate: {merged_df['likely_denied'].mean():.2%}")
    
    return merged_df

def main(claims_file_path, nppes_features_file=None, output_file='hospitals_with_claims.csv'):
    """
    Main pipeline to load and prepare claims data
    
    Args:
        claims_file_path: Path to claims data CSV
        nppes_features_file: Path to NPPES features (if None, uses default)
        output_file: Output filename
    """
    print("=" * 60)
    print("Claims Data Processing Pipeline")
    print("=" * 60)
    
    # Load claims data
    claims_df = load_claims_data(claims_file_path)
    
    # Aggregate by provider
    provider_claims_df = aggregate_claims_by_provider(claims_df)
    
    # Load NPPES features if provided
    if nppes_features_file:
        nppes_features_df = pd.read_csv(nppes_features_file, low_memory=False)
    else:
        nppes_features_file = DATA_PATH / "hospitals_features.csv"
        if nppes_features_file.exists():
            nppes_features_df = pd.read_csv(nppes_features_file, low_memory=False)
        else:
            print(f"\nWarning: NPPES features file not found at {nppes_features_file}")
            print("Saving provider claims data only...")
            OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
            output_path = OUTPUT_PATH / output_file
            provider_claims_df.to_csv(output_path, index=False)
            print(f"Saved to {output_path}")
            return provider_claims_df
    
    # Merge with NPPES data
    merged_df = merge_with_nppes_data(provider_claims_df, nppes_features_df)
    
    # Create target variable
    merged_df = create_target_variable(merged_df)
    
    # Save merged data
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_PATH / output_file
    merged_df.to_csv(output_path, index=False)
    print(f"\nSaved merged data to {output_path}")
    
    return merged_df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_claims_data.py <claims_file_path> [nppes_features_file]")
        print("\nExample:")
        print("  python load_claims_data.py data/claims.csv")
        print("  python load_claims_data.py data/claims.csv data/processed/hospitals_features.csv")
        sys.exit(1)
    
    claims_file = sys.argv[1]
    nppes_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    merged_data = main(claims_file, nppes_file)
