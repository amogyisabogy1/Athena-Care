"""
Data Processing Module for NPPES Data
Loads and preprocesses NPPES data for hospital claim denial prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Configuration
DATA_PATH = "/Users/diyasreedhar/Downloads/NPPES_Data_Dissemination_February_2026"
NPI_FILE = "npidata_pfile_20050523-20260208.csv"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "processed"

def load_nppes_data(sample_size=None, chunk_size=100000):
    """
    Load NPPES data, optionally sampling or using chunks for large files
    
    Args:
        sample_size: Number of rows to sample (None for full dataset)
        chunk_size: Size of chunks if processing in batches
    
    Returns:
        DataFrame with NPPES data
    """
    file_path = os.path.join(DATA_PATH, NPI_FILE)
    
    print(f"Loading NPPES data from {file_path}...")
    
    if sample_size:
        print(f"Sampling {sample_size} rows...")
        df = pd.read_csv(file_path, nrows=sample_size, low_memory=False)
    else:
        # For large files, we'll process in chunks
        print("Loading full dataset (this may take a while)...")
        df = pd.read_csv(file_path, low_memory=False)
    
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    return df

def filter_hospitals(df):
    """
    Filter data to only include hospitals (organizations)
    
    Args:
        df: Raw NPPES dataframe
    
    Returns:
        DataFrame with only hospital/organization records
    """
    print("\nFiltering for hospitals (Entity Type Code = 2)...")
    
    # Entity Type Code: 1 = Individual, 2 = Organization
    # Handle both string and numeric formats
    if df['Entity Type Code'].dtype == 'object':
        hospitals = df[df['Entity Type Code'] == '2'].copy()
    else:
        hospitals = df[df['Entity Type Code'] == 2.0].copy()
    
    print(f"Found {len(hospitals):,} hospital/organization records")
    print(f"Removed {len(df) - len(hospitals):,} individual provider records")
    
    return hospitals

def identify_hospital_taxonomy(df):
    """
    Identify hospitals using taxonomy codes
    Common hospital taxonomy codes include:
    - 282N00000X: General Acute Care Hospital
    - 282NC2000X: Children's Hospital
    - 282NR1301X: Rural Acute Care Hospital
    - 282E00000X: Long Term Care Hospital
    - 283Q00000X: Psychiatric Hospital
    - 284300000X: Rehabilitation Hospital
    """
    print("\nIdentifying hospitals by taxonomy codes...")
    
    # Get all taxonomy code columns
    taxonomy_cols = [col for col in df.columns if 'Healthcare Provider Taxonomy Code' in col]
    
    # Hospital taxonomy codes (first 5 digits)
    hospital_taxonomy_prefixes = ['282N', '282E', '283Q', '2843', '282Y', '282V']
    
    def is_hospital(row):
        for col in taxonomy_cols:
            if pd.notna(row[col]):
                code = str(row[col])
                if any(code.startswith(prefix) for prefix in hospital_taxonomy_prefixes):
                    return True
        return False
    
    df['is_hospital_by_taxonomy'] = df.apply(is_hospital, axis=1)
    
    hospital_count = df['is_hospital_by_taxonomy'].sum()
    print(f"Identified {hospital_count:,} records as hospitals by taxonomy codes")
    
    return df

def basic_cleaning(df):
    """
    Perform basic data cleaning
    
    Args:
        df: DataFrame to clean
    
    Returns:
        Cleaned DataFrame
    """
    print("\nPerforming basic data cleaning...")
    
    # Replace '<UNAVAIL>' and empty strings with NaN
    df = df.replace(['<UNAVAIL>', '', ' '], np.nan)
    
    # Convert date columns
    date_cols = ['Provider Enumeration Date', 'Last Update Date', 
                 'NPI Deactivation Date', 'NPI Reactivation Date', 'Certification Date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    print("Basic cleaning completed")
    
    return df

def save_processed_data(df, filename='hospitals_processed.csv'):
    """
    Save processed data to disk
    
    Args:
        df: DataFrame to save
        filename: Output filename
    """
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_PATH / filename
    
    print(f"\nSaving processed data to {filepath}...")
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df):,} rows to {filepath}")

def main():
    """Main processing pipeline"""
    print("=" * 60)
    print("NPPES Data Processing Pipeline")
    print("=" * 60)
    
    # Load data - use larger sample to get deactivated providers
    # Remove sample_size parameter to process full dataset
    df = load_nppes_data(sample_size=1000000)  # Use 1M rows to get deactivation data
    
    # Filter for hospitals
    hospitals = filter_hospitals(df)
    
    # Identify hospitals by taxonomy
    hospitals = identify_hospital_taxonomy(hospitals)
    
    # Basic cleaning
    hospitals = basic_cleaning(hospitals)
    
    # Save processed data
    save_processed_data(hospitals)
    
    print("\n" + "=" * 60)
    print("Data processing complete!")
    print("=" * 60)
    
    return hospitals

if __name__ == "__main__":
    hospitals_df = main()
