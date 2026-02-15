# UHC Transparency in Coverage Data Integration

## Overview

The [UHC Transparency in Coverage website](https://transparency-in-coverage.uhc.com/) provides Machine Readable Files (MRFs) containing:
- **Negotiated rates** between UHC and providers
- **Provider information** (NPI, TIN, etc.)
- **Service codes** (CPT/HCPCS codes)
- **Rate information** and pricing data
- **Network participation** status

## What You Can Predict with UHC + NPPES Data

### 1. **Data Quality Issues in Rate Files** ⭐
**Target**: Providers with incomplete or missing rate data in UHC files

**Why it matters**: 
- Incomplete rate data can cause billing issues
- Missing rates may indicate data quality problems
- Can predict which providers need rate file updates

**Features**:
- Number of negotiated rates per provider
- Service code coverage
- Rate data completeness

### 2. **Unusual Pricing Patterns**
**Target**: Providers with high rate variability or outlier pricing

**Why it matters**:
- Unusual pricing may indicate errors
- High variability could signal data quality issues
- Can identify providers needing rate verification

**Features**:
- Rate coefficient of variation
- Rate min/max/mean
- Pricing outliers

### 3. **Network Participation Prediction**
**Target**: Predict which providers are in UHC network vs not

**Why it matters**:
- Understand network composition
- Identify providers that should be in network but aren't
- Network expansion opportunities

**Features**:
- NPPES provider characteristics
- Geographic location
- Provider type and taxonomy

### 4. **Rate Data Completeness**
**Target**: Predict which providers will have incomplete rate data

**Why it matters**:
- Proactive data quality management
- Identify providers needing attention
- Prevent billing issues

## How to Use UHC Data

### Step 1: Download UHC MRF Files

1. Visit [UHC Transparency in Coverage](https://transparency-in-coverage.uhc.com/)
2. Navigate to Machine Readable Files section
3. Download MRF files (typically JSON format)
4. Files may be large - download in chunks if needed

### Step 2: Load and Integrate UHC Data

```bash
# Basic usage
python src/load_uhc_tic_data.py data/uhc_mrf.json

# With custom NPPES features
python src/load_uhc_tic_data.py data/uhc_mrf.json data/processed/hospitals_features.csv
```

### Step 3: Train Model with UHC Features

The integrated dataset will include:
- All NPPES features (from `feature_engineering.py`)
- UHC rate features:
  - `uhc_total_rates`: Total number of negotiated rates
  - `uhc_unique_services`: Number of unique service codes
  - `uhc_avg_rate`: Average negotiated rate
  - `uhc_rate_std`: Rate standard deviation
  - `uhc_rate_cv`: Rate coefficient of variation
  - `uhc_in_network`: Binary indicator (1 = in UHC network)
- UHC-based targets:
  - `uhc_incomplete_data`: Providers with incomplete rate data
  - `uhc_unusual_pricing`: Providers with unusual pricing patterns

### Step 4: Train Model

```bash
# Train with UHC-enriched data
python src/model.py --no-wandb
```

The model will automatically use the new features and can predict:
- Providers with data quality issues in rate files
- Providers with unusual pricing patterns
- Network participation patterns

## UHC MRF File Structure

UHC MRF files typically follow this structure:

```json
{
  "in_network": [
    {
      "negotiated_rates": [
        {
          "provider_references": [
            {
              "npi": ["1234567890"],
              "tin": {"type": "ein", "value": "12-3456789"}
            }
          ],
          "negotiated_prices": [
            {
              "negotiated_rate": 150.00,
              "service_code": "99213",
              "billing_code_type": "CPT"
            }
          ]
        }
      ],
      "billing_code": "99213",
      "billing_code_type": "CPT",
      "name": "Office or other outpatient visit"
    }
  ]
}
```

## Example Use Cases

### Use Case 1: Predict Data Quality Issues

**Goal**: Identify providers likely to have incomplete rate data

**Target**: `uhc_incomplete_data` (providers with < 50% of median rates)

**Benefits**:
- Proactive data quality management
- Focus resources on providers needing attention
- Prevent billing and claims issues

### Use Case 2: Identify Pricing Anomalies

**Goal**: Find providers with unusual or potentially erroneous pricing

**Target**: `uhc_unusual_pricing` (high rate variability)

**Benefits**:
- Flag potential data errors
- Identify providers needing rate verification
- Improve data quality

### Use Case 3: Network Analysis

**Goal**: Understand which providers are in UHC network and why

**Features**: NPPES characteristics + UHC network status

**Benefits**:
- Network composition analysis
- Identify expansion opportunities
- Understand provider participation patterns

## Integration with Existing Model

The UHC data enhances your existing NPPES-based deactivation prediction model by adding:

1. **Financial/Pricing Context**: Rate information provides financial health indicators
2. **Network Status**: In-network vs out-of-network status
3. **Data Quality Signals**: Rate data completeness as a quality indicator
4. **Operational Indicators**: Pricing patterns may correlate with operational issues

## Next Steps

1. **Download UHC MRF files** from the transparency website
2. **Run integration script** to merge with NPPES data
3. **Choose a target** (data quality, pricing patterns, etc.)
4. **Train model** with enriched features
5. **Evaluate** which UHC features are most predictive

## Notes

- UHC MRF files can be very large (GBs)
- May need to process in chunks
- File structure may vary - adjust parser as needed
- Rate data is sensitive - ensure proper data handling
