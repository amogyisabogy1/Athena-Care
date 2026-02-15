# Prediction Use Cases with NPPES + External Data

## Current Model: NPI Deactivation Prediction

**What it predicts**: Which providers are likely to have their NPI deactivated

**Performance**: 
- Recall: 46.09% (catches 46% of deactivated providers)
- Precision: 0.88%
- ROC-AUC: 0.71

## Additional Predictions You Can Make

### 1. **Data Quality Issues** (Using NPPES Only)

**Target**: Providers with missing critical fields

**Use Case**: Identify providers needing data verification before network enrollment

**How to use**:
```bash
# Already implemented - just change target in feature_engineering.py
# Set target to 'missing_critical_fields' instead of deactivation
```

### 2. **UHC Network Data Quality** (Using UHC TiC Data)

**Target**: Providers with incomplete rate data in UHC files

**Why it matters**:
- Incomplete rate data causes billing issues
- Predicts which providers need rate file updates
- Proactive data quality management

**How to use**:
```bash
# Download UHC MRF files from https://transparency-in-coverage.uhc.com/
# Integrate with NPPES data
python src/load_uhc_tic_data.py data/uhc_mrf.json

# Train model
python src/model.py --no-wandb
```

**Features added**:
- `uhc_total_rates`: Number of negotiated rates
- `uhc_avg_rate`: Average negotiated rate
- `uhc_rate_cv`: Rate variability
- `uhc_in_network`: Network participation

### 3. **Unusual Pricing Patterns** (Using UHC TiC Data)

**Target**: Providers with high rate variability or outlier pricing

**Why it matters**:
- Flags potential data errors
- Identifies providers needing rate verification
- Improves data quality

**How to use**: Same as above, target `uhc_unusual_pricing`

### 4. **Provider Network Participation** (Using UHC TiC Data)

**Target**: Predict which providers are in UHC network

**Why it matters**:
- Network composition analysis
- Identify expansion opportunities
- Understand participation patterns

### 5. **Claims Denial Prediction** (Using Claims Data)

**Target**: Which providers have high claim denial rates

**Why it matters**:
- Original goal of the project
- Predicts operational issues
- Identifies providers needing support

**How to use**:
```bash
# Load claims data
python src/load_claims_data.py data/claims.csv

# Train model
python src/model.py --no-wandb
```

## Combining Multiple Data Sources

You can combine NPPES + UHC + Claims data for even richer predictions:

```python
# 1. Start with NPPES features
python src/data_processing.py
python src/feature_engineering.py

# 2. Add UHC rate data
python src/load_uhc_tic_data.py data/uhc_mrf.json

# 3. Add claims data (if available)
python src/load_claims_data.py data/claims.csv data/processed/hospitals_with_uhc.csv

# 4. Train with all features
python src/model.py --no-wandb
```

## Recommended Prediction Targets

### For Healthcare Payers:
1. **Provider Data Quality** - Identify providers needing data updates
2. **Network Participation** - Understand which providers are in network
3. **Rate Data Completeness** - Ensure complete rate files
4. **Claims Denial Risk** - Predict providers likely to have denials

### For Healthcare Providers:
1. **NPI Deactivation Risk** - Predict administrative issues
2. **Data Completeness** - Identify missing information
3. **Network Status** - Understand payer network participation

### For Data Quality Teams:
1. **Incomplete Rate Files** - Predict which providers need updates
2. **Unusual Pricing** - Flag potential data errors
3. **Missing Provider Data** - Identify data gaps

## Model Flexibility

The model automatically:
- Detects available data sources
- Selects appropriate target variable
- Includes relevant features
- Handles class imbalance

Just provide the data and the model adapts!
