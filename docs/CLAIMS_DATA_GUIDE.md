# Claims Data Guide

## What Data You Need

To replace the synthetic target with real claims denial data, you need a dataset with the following structure:

### Required Columns

1. **NPI** (string/int): National Provider Identifier - used to join with NPPES data
2. **denial_status** (int): Binary indicator
   - `0` = Claim approved/paid
   - `1` = Claim denied/rejected

### Optional but Recommended Columns

3. **claim_id** (string): Unique identifier for each claim
4. **claim_date** (date): Date of claim submission
5. **service_code** (string): CPT/HCPCS code for the service
6. **claim_amount** (float): Billed amount
7. **denial_reason** (string): Reason code or description for denial
8. **patient_id** (string): Patient identifier (anonymized)

### Example Claims Data Format

```csv
NPI,claim_id,claim_date,service_code,claim_amount,denial_status,denial_reason
1234567890,CLM001,2024-01-15,99213,150.00,0,
1234567890,CLM002,2024-01-16,99214,200.00,1,Prior Authorization Required
1234567890,CLM003,2024-01-17,36415,50.00,0,
9876543210,CLM004,2024-01-18,99213,150.00,1,Incomplete Documentation
```

## Where to Get Claims Data

### 1. Internal Systems (Best Option)
If you work for a healthcare organization, payer, or provider:
- Claims processing systems
- EMR/EHR systems
- Revenue cycle management systems
- Data warehouses

### 2. CMS Data (Public)
- **Medicare Claims**: Available through CMS data releases
- **Medicaid Claims**: State-specific, may require data use agreements
- **CMS Synthetic Claims Data**: Publicly available synthetic datasets for research

### 3. Commercial Claims Databases
- Healthcare data vendors (requires licensing)
- Research databases (requires IRB approval and data use agreements)

### 4. Synthetic Data Generators
For testing/development:
- Generate synthetic claims data with realistic patterns
- Use libraries like `synthetic_data` or custom generators

## Data Requirements

### Minimum Data Size
- **Providers**: At least 1,000 unique providers (NPIs)
- **Claims per Provider**: Ideally 50+ claims per provider for reliable statistics
- **Total Claims**: Minimum 50,000 claims for meaningful patterns

### Data Quality
- **NPI Validation**: Ensure NPIs match NPPES data
- **Denial Rate Distribution**: Should have reasonable denial rates (typically 5-20%)
- **Temporal Coverage**: At least 6-12 months of data
- **Missing Data**: Handle missing values appropriately

## Using the Claims Data Loader

### Step 1: Prepare Your Claims Data

Save your claims data as a CSV file with the required columns:

```python
# Example: Create sample claims data
import pandas as pd

claims_data = pd.DataFrame({
    'NPI': ['1234567890', '1234567890', '9876543210'],
    'claim_id': ['CLM001', 'CLM002', 'CLM003'],
    'claim_date': ['2024-01-15', '2024-01-16', '2024-01-17'],
    'service_code': ['99213', '99214', '99213'],
    'claim_amount': [150.00, 200.00, 150.00],
    'denial_status': [0, 1, 1],
    'denial_reason': ['', 'Prior Auth Required', 'Incomplete Docs']
})

claims_data.to_csv('data/claims.csv', index=False)
```

### Step 2: Run the Claims Data Loader

```bash
# Basic usage
python src/load_claims_data.py data/claims.csv

# With custom NPPES features file
python src/load_claims_data.py data/claims.csv data/processed/hospitals_features.csv
```

### Step 3: Update Feature Engineering

The merged data will include:
- All NPPES features (from `feature_engineering.py`)
- Claims-based features:
  - `total_claims`: Total number of claims
  - `total_denials`: Total number of denied claims
  - `denial_rate`: Proportion of claims denied (0-1)
  - `avg_claim_amount`: Average claim amount
  - `likely_denied`: Binary target (1 if denial_rate > 10%)

### Step 4: Train Model

```bash
# Train with real claims data
python src/model.py
```

The model will automatically use the new target variable from the merged dataset.

## Alternative: Using Aetna Price Transparency Data

**Note**: The Aetna MRF data from Payerset contains **pricing/rate information**, not claims denials. However, you could potentially use it for:

1. **Rate Analysis**: Identify providers with unusual pricing patterns
2. **Provider Profiling**: Combine with other data sources
3. **Feature Engineering**: Use negotiated rates as features

But you would still need actual claims denial data to train a denial prediction model.

## Next Steps

1. **Obtain Claims Data**: Get access to claims data from your organization or public sources
2. **Prepare Data**: Format according to the requirements above
3. **Run Loader**: Use `load_claims_data.py` to merge with NPPES features
4. **Train Model**: Run `model.py` with the merged dataset
5. **Evaluate**: Compare performance with synthetic vs. real data

## Questions?

If you have claims data but it's in a different format, we can modify the loader script to accommodate your specific structure.
