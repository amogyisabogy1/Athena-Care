"""
Load and integrate UHC Transparency in Coverage (TiC) data with NPPES features
TiC data contains negotiated rates, provider information, and pricing data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import gzip
import requests
from urllib.parse import urlparse

DATA_PATH = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "processed"

def download_mrf_file_from_url(url, output_dir, timeout=60, max_size_mb=100):
    """
    Download MRF file from URL (handles .gz compressed files)
    With timeout and size limits to avoid long downloads
    
    Args:
        url: URL to MRF file
        output_dir: Directory to save downloaded file
        timeout: Maximum time to wait (seconds)
        max_size_mb: Maximum file size to download (MB)
    
    Returns:
        Path to downloaded file or None if failed
    """
    import gzip
    import requests
    from urllib.parse import urlparse
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = urlparse(url).path.split('/')[-1]
    output_path = output_dir / filename
    
    # If already downloaded, skip
    if output_path.exists():
        print(f"  File already exists: {output_path.name}")
        return output_path
    
    print(f"  Downloading: {filename} (timeout: {timeout}s, max: {max_size_mb}MB)")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': '*/*'
    }
    
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=timeout)
        response.raise_for_status()
        
        max_size_bytes = max_size_mb * 1024 * 1024
        downloaded = 0
        
        # Handle gzip files
        if url.endswith('.gz'):
            json_path = output_path.with_suffix('')
            with gzip.GzipFile(fileobj=response.raw) as f_in:
                with open(json_path, 'wb') as f_out:
                    while True:
                        chunk = f_in.read(8192)
                        if not chunk:
                            break
                        downloaded += len(chunk)
                        if downloaded > max_size_bytes:
                            print(f"  ⚠ File too large ({downloaded/(1024*1024):.1f}MB), skipping...")
                            if json_path.exists():
                                json_path.unlink()
                            return None
                        f_out.write(chunk)
            print(f"  ✓ Downloaded and decompressed ({downloaded/(1024*1024):.1f}MB)")
            return json_path
        else:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded > max_size_bytes:
                            print(f"  ⚠ File too large ({downloaded/(1024*1024):.1f}MB), stopping...")
                            output_path.unlink()
                            return None
            print(f"  ✓ Downloaded ({downloaded/(1024*1024):.1f}MB)")
            return output_path
    except requests.Timeout:
        print(f"  ✗ Timeout after {timeout}s - file may be too large")
        return None
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None

def load_uhc_index_file(index_file_path):
    """
    Load UHC MRF index file and extract file URLs
    
    Args:
        index_file_path: Path to UHC index JSON file
    
    Returns:
        List of MRF file URLs to download
    """
    print(f"Loading UHC index file: {index_file_path}")
    
    with open(index_file_path, 'r') as f:
        data = json.load(f)
    
    file_urls = []
    
    # Extract file URLs from reporting structure
    if 'reporting_structure' in data:
        for entity in data['reporting_structure']:
            # In-network files
            if 'in_network_files' in entity:
                for file_info in entity['in_network_files']:
                    if 'location' in file_info:
                        file_urls.append(file_info['location'])
            
            # Allowed amount files
            if 'allowed_amount_file' in entity:
                if isinstance(entity['allowed_amount_file'], dict) and 'location' in entity['allowed_amount_file']:
                    file_urls.append(entity['allowed_amount_file']['location'])
    
    print(f"Found {len(file_urls)} MRF file(s) in index")
    return file_urls, data

def load_uhc_mrf_data(mrf_file_path_or_url, download_dir=None):
    """
    Load UHC Machine Readable File (MRF) data
    
    Can handle:
    - Index files (extracts URLs and downloads actual MRF files)
    - Direct MRF data files (JSON or JSON.gz)
    - URLs to MRF files
    
    Args:
        mrf_file_path_or_url: Path to UHC MRF file or index file, or URL
        download_dir: Directory to download files (if URL provided)
    
    Returns:
        DataFrame with UHC rate data
    """
    print(f"Loading UHC MRF data from {mrf_file_path_or_url}...")
    
    file_path = Path(mrf_file_path_or_url)
    
    # Check if it's an index file
    if file_path.exists() and 'index.json' in file_path.name:
        print("Detected index file - extracting MRF file URLs...")
        file_urls, index_data = load_uhc_index_file(file_path)
        
        if not file_urls:
            raise ValueError("No MRF file URLs found in index file")
        
        # Download and process each file
        if download_dir is None:
            download_dir = DATA_PATH / "uhc_mrf_files"
        else:
            download_dir = Path(download_dir)
        
        all_rates = []
        
        # Only process first file to avoid long downloads
        print(f"\n⚠️  Processing only FIRST file to avoid long downloads")
        print(f"   (Total files available: {len(file_urls)})")
        
        if file_urls:
            url = file_urls[0]
            print(f"\nProcessing: {url[:80]}...")
            downloaded_file = download_mrf_file_from_url(url, download_dir, timeout=60, max_size_mb=50)
            
            if downloaded_file and downloaded_file.exists():
                # Load the actual MRF data
                rates_df = load_actual_mrf_file(downloaded_file)
                if rates_df is not None and len(rates_df) > 0:
                    all_rates.append(rates_df)
                    print(f"✓ Successfully processed 1 file with {len(rates_df):,} rate records")
            else:
                print("⚠️  Could not download file (may be too large or timeout)")
                print("   You can download files manually and process them directly")
        
        if all_rates:
            df = pd.concat(all_rates, ignore_index=True)
            print(f"\nTotal rate records loaded: {len(df):,}")
            return df
        else:
            raise ValueError("No rate data could be loaded from MRF files")
    
    # Direct file or URL
    elif file_path.exists() or mrf_file_path_or_url.startswith('http'):
        return load_actual_mrf_file(mrf_file_path_or_url, download_dir)
    else:
        raise FileNotFoundError(f"File not found: {mrf_file_path_or_url}")

def load_actual_mrf_file(file_path_or_url, download_dir=None):
    """
    Load actual MRF data file (not index)
    """
    import gzip
    import requests
    
    # Handle URLs
    if file_path_or_url.startswith('http'):
        if download_dir is None:
            download_dir = DATA_PATH / "uhc_mrf_files"
        else:
            download_dir = Path(download_dir)
        
        file_path = download_mrf_file_from_url(file_path_or_url, download_dir)
        if file_path is None:
            return None
    else:
        file_path = Path(file_path_or_url)
    
    print(f"Loading MRF data from: {file_path.name}")
    
    # Handle gzip files
    if file_path.suffix == '.gz' or str(file_path).endswith('.gz'):
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(file_path, 'r') as f:
            data = json.load(f)
    
    # Extract provider rate information
    rates = []
    
    if 'in_network' in data:
        for item in data['in_network']:
            billing_code = item.get('billing_code')
            billing_code_type = item.get('billing_code_type')
            name = item.get('name', '')
            
            for negotiated_rate in item.get('negotiated_rates', []):
                # Get provider references
                for provider_group in negotiated_rate.get('provider_groups', []):
                    for provider in provider_group.get('npi', []):
                        for price_info in negotiated_rate.get('negotiated_prices', []):
                            rates.append({
                                'NPI': provider,
                                'TIN': provider_group.get('tin', {}).get('value') if isinstance(provider_group.get('tin'), dict) else provider_group.get('tin'),
                                'service_code': billing_code,
                                'service_code_type': billing_code_type,
                                'service_description': name,
                                'negotiated_rate': price_info.get('negotiated_rate'),
                                'billing_code_modifier': price_info.get('billing_code_modifier', []),
                                'expiration_date': price_info.get('expiration_date'),
                                'billing_class': price_info.get('billing_class')
                            })
    
    if not rates:
        print("  Warning: No rate data found in file")
        return None
    
    df = pd.DataFrame(rates)
    print(f"  Loaded {len(df):,} rate records")
    
    # Validate NPI column
    if 'NPI' not in df.columns or df['NPI'].isna().all():
        print("  Warning: No valid NPI data found")
        return None
    
    return df

def aggregate_uhc_data_by_provider(uhc_df):
    """
    Aggregate UHC rate data by provider to create features
    
    Creates features like:
    - Number of negotiated rates
    - Average negotiated rate
    - Rate variability
    - Service code coverage
    - Network participation indicators
    """
    print("\nAggregating UHC data by provider...")
    
    # Group by NPI
    provider_stats = uhc_df.groupby('NPI').agg({
        'service_code': ['count', 'nunique'],  # Total rates, unique services
        'negotiated_rate': ['mean', 'std', 'min', 'max', 'count'] if 'negotiated_rate' in uhc_df.columns else []
    }).reset_index()
    
    # Flatten column names
    provider_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                             for col in provider_stats.columns.values]
    
    # Rename columns
    rename_dict = {
        'service_code_count': 'uhc_total_rates',
        'service_code_nunique': 'uhc_unique_services',
    }
    
    if 'negotiated_rate_mean' in provider_stats.columns:
        rename_dict.update({
            'negotiated_rate_mean': 'uhc_avg_rate',
            'negotiated_rate_std': 'uhc_rate_std',
            'negotiated_rate_min': 'uhc_min_rate',
            'negotiated_rate_max': 'uhc_max_rate',
            'negotiated_rate_count': 'uhc_rate_count'
        })
    
    provider_stats = provider_stats.rename(columns=rename_dict)
    
    # Calculate rate variability (coefficient of variation)
    if 'uhc_avg_rate' in provider_stats.columns and 'uhc_rate_std' in provider_stats.columns:
        provider_stats['uhc_rate_cv'] = provider_stats['uhc_rate_std'] / provider_stats['uhc_avg_rate']
        provider_stats['uhc_rate_cv'] = provider_stats['uhc_rate_cv'].fillna(0)
    
    # Network participation indicator
    provider_stats['uhc_in_network'] = 1
    
    print(f"Aggregated to {len(provider_stats):,} providers")
    
    return provider_stats

def merge_with_nppes_data(uhc_provider_df, nppes_features_df):
    """
    Merge UHC TiC data with NPPES features
    
    Args:
        uhc_provider_df: Aggregated UHC data by provider
        nppes_features_df: NPPES features from feature_engineering.py
    
    Returns:
        Merged DataFrame ready for modeling
    """
    print("\nMerging UHC TiC data with NPPES features...")
    
    # Merge on NPI
    merged_df = nppes_features_df.merge(
        uhc_provider_df,
        on='NPI',
        how='left'  # Keep all NPPES providers, mark UHC network status
    )
    
    # Fill missing UHC features with 0 (provider not in UHC network)
    uhc_cols = [col for col in merged_df.columns if col.startswith('uhc_')]
    for col in uhc_cols:
        if col != 'uhc_in_network':
            merged_df[col] = merged_df[col].fillna(0)
        else:
            merged_df[col] = merged_df[col].fillna(0).astype(int)
    
    print(f"Merged dataset: {len(merged_df):,} providers")
    print(f"Providers in UHC network: {merged_df['uhc_in_network'].sum():,} ({merged_df['uhc_in_network'].mean():.2%})")
    print(f"Providers not in UHC network: {(~merged_df['uhc_in_network'].astype(bool)).sum():,}")
    
    return merged_df

def create_uhc_based_targets(merged_df):
    """
    Create prediction targets based on UHC data
    
    Potential targets:
    1. Providers with missing/incomplete rate data
    2. Providers with unusual pricing patterns
    3. Providers at risk of network termination
    4. Data quality issues in rate files
    """
    print("\nCreating UHC-based target variables...")
    
    # Target 1: Providers with incomplete rate data (few rates relative to expected)
    if 'uhc_total_rates' in merged_df.columns:
        # Providers with very few rates (potential data quality issue)
        median_rates = merged_df[merged_df['uhc_in_network'] == 1]['uhc_total_rates'].median()
        merged_df['uhc_incomplete_data'] = (
            (merged_df['uhc_in_network'] == 1) & 
            (merged_df['uhc_total_rates'] < median_rates * 0.5)
        ).astype(int)
        print(f"Providers with incomplete UHC rate data: {merged_df['uhc_incomplete_data'].sum():,}")
    
    # Target 2: Providers with unusual pricing (high variability)
    if 'uhc_rate_cv' in merged_df.columns:
        high_cv_threshold = merged_df[merged_df['uhc_in_network'] == 1]['uhc_rate_cv'].quantile(0.9)
        merged_df['uhc_unusual_pricing'] = (
            (merged_df['uhc_in_network'] == 1) & 
            (merged_df['uhc_rate_cv'] > high_cv_threshold)
        ).astype(int)
        print(f"Providers with unusual pricing patterns: {merged_df['uhc_unusual_pricing'].sum():,}")
    
    return merged_df

def main(uhc_file_path, nppes_features_file=None, output_file='hospitals_with_uhc.csv'):
    """
    Main pipeline to load and integrate UHC TiC data
    
    Args:
        uhc_file_path: Path to UHC MRF file (JSON or CSV)
        nppes_features_file: Path to NPPES features (if None, uses default)
        output_file: Output filename
    """
    print("=" * 60)
    print("UHC Transparency in Coverage Data Integration")
    print("=" * 60)
    
    # Load UHC data
    uhc_df = load_uhc_mrf_data(uhc_file_path)
    
    # Aggregate by provider
    uhc_provider_df = aggregate_uhc_data_by_provider(uhc_df)
    
    # Load NPPES features
    if nppes_features_file:
        nppes_features_df = pd.read_csv(nppes_features_file, low_memory=False)
    else:
        nppes_features_file = DATA_PATH / "hospitals_features.csv"
        if nppes_features_file.exists():
            nppes_features_df = pd.read_csv(nppes_features_file, low_memory=False)
        else:
            print(f"\nWarning: NPPES features file not found at {nppes_features_file}")
            print("Saving UHC provider data only...")
            OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
            output_path = OUTPUT_PATH / output_file
            uhc_provider_df.to_csv(output_path, index=False)
            print(f"Saved to {output_path}")
            return uhc_provider_df
    
    # Merge with NPPES data
    merged_df = merge_with_nppes_data(uhc_provider_df, nppes_features_df)
    
    # Create UHC-based targets
    merged_df = create_uhc_based_targets(merged_df)
    
    # Save merged data
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_PATH / output_file
    merged_df.to_csv(output_path, index=False)
    print(f"\nSaved merged data to {output_path}")
    
    return merged_df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_uhc_tic_data.py <uhc_mrf_file> [nppes_features_file]")
        print("\nExample:")
        print("  python load_uhc_tic_data.py data/uhc_mrf.json")
        print("  python load_uhc_tic_data.py data/uhc_rates.csv data/processed/hospitals_features.csv")
        print("\nNote: UHC MRF files can be JSON (machine-readable format) or CSV")
        sys.exit(1)
    
    uhc_file = sys.argv[1]
    nppes_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    merged_data = main(uhc_file, nppes_file)
