# Hospital Insurance Claims Denial Prediction

This project uses XGBoost to predict which hospitals are likely to have insurance claims denied based on NPPES (National Plan and Provider Enumeration System) data.

## Overview

The National Provider Identifier (NPI) data from NPPES contains information about healthcare providers including:
- Organization details (hospitals, clinics, etc.)
- Provider taxonomy codes (specialty classifications)
- License information
- Address and contact information
- Deactivation/reactivation status
- Data completeness indicators

## Project Structure

```
Athena-Care/
├── data/                    # Data directory (link to NPPES data)
├── notebooks/              # Jupyter notebooks for exploration
├── src/
│   ├── data_processing.py  # Data loading and preprocessing
│   ├── feature_engineering.py  # Feature creation
│   ├── model.py            # XGBoost model training and evaluation
│   └── utils.py            # Utility functions
├── models/                 # Saved model files
├── results/                 # Model evaluation results
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure NPPES data is accessible (update path in `src/data_processing.py`)

3. Run the pipeline:
```bash
python src/data_processing.py
python src/feature_engineering.py
python src/model.py
```

## Features

The model uses features derived from NPPES data that may correlate with claim denials:
- Data completeness scores
- Provider taxonomy codes (specialty types)
- Deactivation/reactivation history
- License information completeness
- Geographic features
- Organization structure (subparts, parent organizations)

## Note on Target Variable

Since actual claims denial data is not available in NPPES, the model uses a synthetic target variable based on features that correlate with higher risk of claim denials (e.g., incomplete data, deactivation history, missing licenses). In production, this should be replaced with actual claims denial data.

## Usage

### Quick Start

1. **Process the data:**
   ```bash
   python src/data_processing.py
   ```
   This will filter for hospitals and perform basic cleaning. By default, it processes 100,000 rows for testing. Remove the `sample_size` parameter in the script to process the full dataset.

2. **Engineer features:**
   ```bash
   python src/feature_engineering.py
   ```
   This creates features that may correlate with claim denials and generates a synthetic target variable.

3. **Train the model:**
   ```bash
   python src/model.py
   ```
   This trains an XGBoost model, evaluates it, and saves the results.
   
   **With Weights & Biases (wandb) logging:**
   ```bash
   # First, login to wandb (one-time setup)
   wandb login
   
   # Train with wandb logging (default)
   python src/model.py
   
   # Or specify a custom project name
   python src/model.py --wandb-project my-project-name
   
   # To disable wandb logging
   python src/model.py --no-wandb
   ```
   
   The model will:
   - Save the best checkpoint during training (based on validation AUC)
   - Log all metrics, hyperparameters, and visualizations to wandb
   - Save model artifacts to wandb for easy model versioning
   
   **With Weights & Biases (wandb) logging:**
   ```bash
   # First, login to wandb (one-time setup)
   wandb login
   
   # Train with wandb logging (default)
   python src/model.py
   
   # Or specify a custom project name
   python src/model.py --wandb-project my-project-name
   
   # To disable wandb logging
   python src/model.py --no-wandb
   ```
   
   The model will:
   - Save the best checkpoint during training (based on validation AUC)
   - Log all metrics, hyperparameters, and visualizations to wandb
   - Save model artifacts to wandb for easy model versioning

## Weights & Biases Integration

This project includes full integration with [Weights & Biases (wandb)](https://wandb.ai) for experiment tracking:

### Setup

1. **Install wandb** (already in requirements.txt):
   ```bash
   pip install -r requirements.txt
   ```

2. **Login to wandb** (one-time setup):
   ```bash
   wandb login
   ```
   You'll need to create a free account at https://wandb.ai if you don't have one.

### What Gets Logged

- **Hyperparameters**: All XGBoost parameters
- **Training Metrics**: Real-time logging of train/eval AUC during training
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- **Visualizations**: 
  - ROC curves
  - Precision-Recall curves
  - Confusion matrices
  - Feature importance charts
- **Model Artifacts**: Saved models and checkpoints
- **Best Model**: Automatically tracks the best iteration based on validation AUC

### Checkpoint Saving

The model automatically saves the best checkpoint during training:
- Location: `models/checkpoints/best_model.json`
- Saved based on: Best validation AUC score
- Early stopping: Stops training if no improvement for 20 rounds

### Understanding the Features

The model uses several feature categories:

- **Data Completeness**: Scores based on presence of critical fields (EIN, address, license, etc.)
- **Taxonomy Features**: Hospital type classification based on taxonomy codes
- **License Features**: License presence, count, and state consistency
- **Status Features**: Deactivation/reactivation history, enumeration dates
- **Organization Features**: Subpart status, parent organization relationships
- **Geographic Features**: State and region information

### Model Output

The trained model outputs:
- Binary prediction: `likely_denied` (0 or 1)
- Risk score: `claim_denial_risk` (0 to 1, continuous)

### Using Real Claims Data

The model currently uses a synthetic target variable. To use real claims denial data:

1. **Prepare your claims data** (see `docs/CLAIMS_DATA_GUIDE.md` for format requirements)
   - Required: `NPI`, `denial_status` (0=approved, 1=denied)
   - Optional: `claim_date`, `claim_amount`, `denial_reason`, etc.

2. **Load and merge claims data:**
   ```bash
   python src/load_claims_data.py data/your_claims.csv
   ```

3. **Train with real data:**
   ```bash
   python src/model.py
   ```

The script will automatically:
- Aggregate claims by provider
- Merge with NPPES features
- Create denial rate-based target variable
- Train the model with real targets

**Note**: The Aetna price transparency data from Payerset contains pricing/rate information, not claims denials. You'll need actual claims submission data with denial status to train a denial prediction model.

### Next Steps

To use this with real claims data:
1. Obtain claims data with denial status (see `docs/CLAIMS_DATA_GUIDE.md`)
2. Use `load_claims_data.py` to merge with NPPES features
3. Retrain the model with the real target variable
4. Validate model performance on held-out test data
