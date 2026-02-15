"""
Feature Engineering Module
Creates features from NPPES data that may correlate with insurance claim denials
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

DATA_PATH = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "processed"

def calculate_data_completeness(df):
    """
    Calculate data completeness scores - incomplete data may correlate with claim denials
    
    Returns:
        DataFrame with completeness features
    """
    print("Calculating data completeness features...")
    
    # Key fields that should be present for claims processing
    key_fields = [
        'Provider Organization Name (Legal Business Name)',
        'Employer Identification Number (EIN)',
        'Provider First Line Business Practice Location Address',
        'Provider Business Practice Location Address City Name',
        'Provider Business Practice Location Address State Name',
        'Provider Business Practice Location Address Postal Code',
        'Provider Business Practice Location Address Telephone Number',
        'Healthcare Provider Taxonomy Code_1',
        'Provider License Number_1',
        'Provider License Number State Code_1'
    ]
    
    # Calculate completeness for each key field
    for field in key_fields:
        if field in df.columns:
            df[f'{field}_complete'] = df[field].notna().astype(int)
    
    # Overall completeness score
    available_key_fields = [f'{f}_complete' for f in key_fields if f'{f}_complete' in df.columns]
    if available_key_fields:
        df['data_completeness_score'] = df[available_key_fields].mean(axis=1)
        df['missing_critical_fields'] = (df['data_completeness_score'] < 0.8).astype(int)
    
    return df

def extract_taxonomy_features(df):
    """
    Extract features from taxonomy codes
    Different hospital types may have different denial rates
    """
    print("Extracting taxonomy features...")
    
    taxonomy_cols = [col for col in df.columns if 'Healthcare Provider Taxonomy Code' in col and '_' in col]
    
    # Count number of taxonomy codes
    df['num_taxonomy_codes'] = df[taxonomy_cols].notna().sum(axis=1)
    
    # Extract primary taxonomy category
    if 'Healthcare Provider Taxonomy Code_1' in df.columns:
        df['primary_taxonomy_prefix'] = df['Healthcare Provider Taxonomy Code_1'].astype(str).str[:5]
        
        # Hospital type categories
        hospital_types = {
            '282N': 'General_Acute_Care',
            '282E': 'Long_Term_Care',
            '283Q': 'Psychiatric',
            '2843': 'Rehabilitation',
            '282Y': 'Specialty',
            '282V': 'Long_Term_Care_Hospital'
        }
        
        df['hospital_type'] = df['primary_taxonomy_prefix'].map(hospital_types)
        df['hospital_type'] = df['hospital_type'].fillna('Other')
    
    return df

def extract_license_features(df):
    """
    Extract license-related features
    Missing or incomplete license info may lead to denials
    """
    print("Extracting license features...")
    
    license_cols = [col for col in df.columns if 'Provider License Number' in col and '_' in col]
    license_state_cols = [col for col in df.columns if 'Provider License Number State Code' in col and '_' in col]
    
    # Count licenses
    df['num_licenses'] = df[license_cols].notna().sum(axis=1)
    
    # Check if primary license is present
    if 'Provider License Number_1' in df.columns:
        df['has_primary_license'] = df['Provider License Number_1'].notna().astype(int)
    
    # Check license state consistency
    if 'Provider License Number State Code_1' in df.columns and 'Provider Business Practice Location Address State Name' in df.columns:
        df['license_state_match'] = (
            df['Provider License Number State Code_1'] == 
            df['Provider Business Practice Location Address State Name']
        ).astype(int)
        df['license_state_match'] = df['license_state_match'].fillna(0)
    
    return df

def extract_status_features(df):
    """
    Extract status-related features
    Deactivated or recently updated providers may have issues
    """
    print("Extracting status features...")
    
    # Deactivation status
    if 'NPI Deactivation Reason Code' in df.columns:
        df['is_deactivated'] = df['NPI Deactivation Reason Code'].notna().astype(int)
    
    if 'NPI Deactivation Date' in df.columns:
        df['has_deactivation_date'] = df['NPI Deactivation Date'].notna().astype(int)
    
    # Reactivation status
    if 'NPI Reactivation Date' in df.columns:
        df['is_reactivated'] = df['NPI Reactivation Date'].notna().astype(int)
    
    # Days since enumeration
    if 'Provider Enumeration Date' in df.columns:
        # Convert to datetime if not already
        enum_date = pd.to_datetime(df['Provider Enumeration Date'], errors='coerce')
        df['days_since_enumeration'] = (pd.Timestamp.now() - enum_date).dt.days
        df['days_since_enumeration'] = df['days_since_enumeration'].fillna(0)
    
    # Days since last update
    if 'Last Update Date' in df.columns:
        # Convert to datetime if not already
        update_date = pd.to_datetime(df['Last Update Date'], errors='coerce')
        df['days_since_update'] = (pd.Timestamp.now() - update_date).dt.days
        df['days_since_update'] = df['days_since_update'].fillna(0)
        df['recently_updated'] = (df['days_since_update'] < 365).astype(int)
    
    return df

def extract_organization_features(df):
    """
    Extract organization structure features
    """
    print("Extracting organization features...")
    
    # Subpart status
    if 'Is Organization Subpart' in df.columns:
        df['is_subpart'] = df['Is Organization Subpart'].map({'Y': 1, 'N': 0, '': 0}).fillna(0)
    
    # Has parent organization
    if 'Parent Organization LBN' in df.columns:
        df['has_parent_org'] = df['Parent Organization LBN'].notna().astype(int)
    
    return df

def extract_geographic_features(df):
    """
    Extract geographic features
    Some states/regions may have different denial patterns
    """
    print("Extracting geographic features...")
    
    if 'Provider Business Practice Location Address State Name' in df.columns:
        df['state'] = df['Provider Business Practice Location Address State Name']
        
        # Region mapping (simplified)
        region_map = {
            'CA': 'West', 'OR': 'West', 'WA': 'West', 'NV': 'West', 'AZ': 'West',
            'NY': 'Northeast', 'MA': 'Northeast', 'PA': 'Northeast', 'NJ': 'Northeast',
            'TX': 'South', 'FL': 'South', 'GA': 'South', 'NC': 'South',
            'IL': 'Midwest', 'OH': 'Midwest', 'MI': 'Midwest', 'WI': 'Midwest'
        }
        
        df['region'] = df['state'].map(region_map).fillna('Other')
    
    return df

def create_target_from_nppes_data(df):
    """
    Create target variable from actual NPPES data
    Priority order:
    1. Claims denial data (if available)
    2. NPI deactivation history (has been deactivated at some point)
    3. Data completeness (missing critical fields)
    
    This predicts REAL outcomes from the data!
    """
    print("Creating target variable from NPPES data...")
    
    # Check if real claims data exists (takes priority)
    if 'denial_rate' in df.columns or 'likely_denied' in df.columns:
        print("Real claims data detected! Using claims denial as target.")
        if 'likely_denied' not in df.columns and 'denial_rate' in df.columns:
            df['likely_denied'] = (df['denial_rate'] > 0.1).astype(int)
        return df
    
    # Try NPI deactivation first (use deactivation DATE, not reason code)
    # Predict: Has this provider been deactivated at some point?
    has_deactivation_data = False
    deactivation_rate = 0
    
    if 'NPI Deactivation Date' in df.columns:
        # Use deactivation date as indicator (more reliable than reason code)
        df['has_deactivation_history'] = df['NPI Deactivation Date'].notna().astype(int)
        deactivation_count = df['has_deactivation_history'].sum()
        deactivation_rate = deactivation_count / len(df) if len(df) > 0 else 0
        
        if deactivation_count > 50:  # Need at least 50 for meaningful training
            has_deactivation_data = True
            df['likely_denied'] = df['has_deactivation_history'].copy()
            print(f"✅ Using NPI Deactivation History as target variable")
            print(f"   This predicts which providers have been deactivated at some point")
            print(f"   Providers with deactivation history: {deactivation_count:,} ({deactivation_rate:.2%})")
            print(f"   Providers never deactivated: {(~df['has_deactivation_history'].astype(bool)).sum():,} ({1-deactivation_rate:.2%})")
    
    # Fallback to data completeness if not enough deactivation data
    if not has_deactivation_data:
        if 'missing_critical_fields' in df.columns:
            # Target: Providers with missing critical fields (incomplete data)
            df['likely_denied'] = df['missing_critical_fields'].copy()
            missing_rate = df['likely_denied'].mean()
            print(f"Using 'Missing Critical Fields' as target variable")
            print(f"This predicts which providers have incomplete data")
            print(f"Providers with missing critical fields: {df['likely_denied'].sum():,} ({missing_rate:.2%})")
            print(f"Providers with complete data: {(~df['likely_denied'].astype(bool)).sum():,} ({1-missing_rate:.2%})")
            
        elif 'data_completeness_score' in df.columns:
            # Alternative: Predict providers with poor data quality
            df['likely_denied'] = (df['data_completeness_score'] < 0.8).astype(int)
            poor_quality_rate = df['likely_denied'].mean()
            print(f"Using 'Poor Data Quality' as target variable (<80% complete)")
            print(f"Providers with poor data quality: {df['likely_denied'].sum():,} ({poor_quality_rate:.2%})")
        else:
            raise ValueError("No suitable target variable found in data!")
    
    return df

def select_features(df):
    """
    Select final features for modeling
    """
    print("Selecting features for modeling...")
    
    # Feature categories
    feature_cols = []
    
    # Completeness features
    # EXCLUDE individual field completeness scores - they're too directly related to the target
    # Only use aggregate scores if they're not the target
    
    # Check if missing_critical_fields is the target
    is_missing_fields_target = False
    if 'missing_critical_fields' in df.columns and 'likely_denied' in df.columns:
        is_missing_fields_target = df['likely_denied'].equals(df['missing_critical_fields'])
    
    # Don't use individual field completeness if predicting missing fields
    if not is_missing_fields_target:
        completeness_cols = [col for col in df.columns if '_complete' in col]
        feature_cols.extend(completeness_cols)
    
    # Only add aggregate completeness score if it's NOT directly the target
    if 'data_completeness_score' in df.columns:
        if not is_missing_fields_target:
            feature_cols.append('data_completeness_score')
    
    # Don't add missing_critical_fields as feature if it's the target
    if 'missing_critical_fields' in df.columns:
        if not is_missing_fields_target:
            feature_cols.append('missing_critical_fields')
    
    # Taxonomy features
    if 'num_taxonomy_codes' in df.columns:
        feature_cols.append('num_taxonomy_codes')
    if 'hospital_type' in df.columns:
        feature_cols.append('hospital_type')
    
    # License features
    if 'num_licenses' in df.columns:
        feature_cols.append('num_licenses')
    if 'has_primary_license' in df.columns:
        feature_cols.append('has_primary_license')
    if 'license_state_match' in df.columns:
        feature_cols.append('license_state_match')
    
    # Status features (exclude is_deactivated if it's the target)
    # Note: is_deactivated will be excluded if it's used as target
    if 'is_reactivated' in df.columns:
        feature_cols.append('is_reactivated')
    if 'days_since_enumeration' in df.columns:
        feature_cols.append('days_since_enumeration')
    if 'days_since_update' in df.columns:
        feature_cols.append('days_since_update')
    if 'recently_updated' in df.columns:
        feature_cols.append('recently_updated')
    
    # Organization features
    if 'is_subpart' in df.columns:
        feature_cols.append('is_subpart')
    if 'has_parent_org' in df.columns:
        feature_cols.append('has_parent_org')
    
    # Geographic features
    if 'state' in df.columns:
        feature_cols.append('state')
    if 'region' in df.columns:
        feature_cols.append('region')
    
    # UHC Transparency in Coverage features (if available)
    uhc_features = [col for col in df.columns if col.startswith('uhc_')]
    feature_cols.extend(uhc_features)
    if uhc_features:
        print(f"  Added {len(uhc_features)} UHC TiC features")
    
    # Keep only existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Exclude target variables from features to prevent data leakage
    # Check if deactivation-related features are the target
    if 'likely_denied' in df.columns and 'has_deactivation_history' in df.columns:
        # If target is based on deactivation, exclude all deactivation-related features
        if df['likely_denied'].equals(df['has_deactivation_history']):
            # Remove all deactivation-related features (they're the target!)
            # Also remove is_reactivated - it perfectly predicts deactivation (all deactivated were reactivated)
            feature_cols = [col for col in feature_cols if col not in [
                'is_deactivated', 
                'has_deactivation_date',
                'has_deactivation_history',
                'is_reactivated'  # Perfect predictor - all deactivated were reactivated
            ]]
            print(f"  Excluded deactivation features (is_deactivated, has_deactivation_date, is_reactivated) to prevent data leakage")
    
    # Also check if is_deactivated is the target
    if 'is_deactivated' in feature_cols and 'likely_denied' in df.columns:
        if df['likely_denied'].equals(df['is_deactivated']):
            feature_cols = [col for col in feature_cols if col != 'is_deactivated']
            print(f"  Excluded is_deactivated to prevent data leakage")
    
    # Add target variables (for reference, not as features)
    if 'likely_denied' in df.columns:
        feature_cols.append('likely_denied')
    if 'has_deactivation_history' in df.columns:
        feature_cols.append('has_deactivation_history')
    if 'is_deactivated' in df.columns and 'is_deactivated' not in feature_cols:
        # Keep is_deactivated if it's not the target
        feature_cols.append('is_deactivated')
    
    # Keep NPI for reference
    if 'NPI' in df.columns:
        feature_cols.append('NPI')
    
    print(f"Selected {len(feature_cols)} features")
    
    return df[feature_cols]

def main():
    """Main feature engineering pipeline"""
    print("=" * 60)
    print("Feature Engineering Pipeline")
    print("=" * 60)
    
    # Load processed data
    input_file = DATA_PATH / "hospitals_processed.csv"
    if not input_file.exists():
        print(f"Error: {input_file} not found. Run data_processing.py first.")
        return
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"Loaded {len(df):,} records")
    
    # Apply feature engineering
    df = calculate_data_completeness(df)
    df = extract_taxonomy_features(df)
    df = extract_license_features(df)
    df = extract_status_features(df)
    df = extract_organization_features(df)
    df = extract_geographic_features(df)
    df = create_target_from_nppes_data(df)
    df = select_features(df)
    
    # Save features
    output_file = OUTPUT_PATH / "hospitals_features.csv"
    print(f"\nSaving features to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df):,} records with {len(df.columns)} features")
    
    print("\n" + "=" * 60)
    print("Feature engineering complete!")
    print("=" * 60)
    
    return df

if __name__ == "__main__":
    features_df = main()
