# UI Integration Guide

## For Your Friend: How to Use the Model in Your UI

Your friend doesn't need the model files directly! They just need to **call the API** that serves the model.

## Option 1: Use Deployed API (Recommended - Easiest)

If you deploy the API to Railway/Heroku/etc., your friend just needs the **API URL**.

### Example API Call:

```javascript
// Replace with your deployed API URL
const API_URL = 'https://your-app-name.up.railway.app';

// Make a prediction
async function predictRisk(npi) {
  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ npi: npi })
  });
  
  const data = await response.json();
  return data.predictions[0];
}

// Usage
const result = await predictRisk('1234567890');
console.log(result.risk_level); // "High", "Medium", or "Low"
console.log(result.predicted_risk); // 0.75 (probability)
```

### API Endpoints:

- `GET /health` - Check if API is running
- `POST /predict` - Get prediction for one or more NPIs
- `GET /providers/search?q=NAME_OR_NPI` - Search for providers
- `GET /model/info` - Get model information

### Example Response:

```json
{
  "predictions": [
    {
      "npi": "1234567890",
      "predicted_risk": 0.75,
      "predicted_class": 1,
      "risk_level": "High",
      "interpretation": "High risk provider. Probability of issues: 75.0%"
    }
  ],
  "count": 1
}
```

## Option 2: Run API Locally

If you want to run the API on your own machine:

### Step 1: Clone the Repository

```bash
git clone https://github.com/amogyisabogy1/Athena-Care.git
cd Athena-Care
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Train the Model (First Time Only)

The model files are not in GitHub (they're too large). You need to train it first:

```bash
# Process the data
python run_pipeline.py

# Or step by step:
python src/data_processing.py
python src/feature_engineering.py
python src/model.py --no-wandb
```

**Note:** You'll need the NPPES data files. Update the path in `src/data_processing.py` to point to your data location.

### Step 4: Start the API Server

```bash
python src/api.py
```

The API will run on `http://localhost:5000`

### Step 5: Use in Your UI

```javascript
const API_URL = 'http://localhost:5000';

// Same API calls as above
```

## Option 3: Direct Model Integration (Advanced)

If your friend wants to use the model directly in their UI (without API), they would need:

1. **Model files** from `models/` directory:
   - `xgb_model_*.pkl` - The trained XGBoost model
   - `label_encoders_*.pkl` - Label encoders for categorical features
   - `model_metadata_*.json` - Model metadata

2. **Feature data** from `data/processed/hospitals_features.csv`

3. **Python dependencies** to load and run the model

This is more complex and not recommended. Using the API is much easier!

## Quick Test

Test the API is working:

```bash
# Health check
curl http://localhost:5000/health

# Or if deployed:
curl https://your-app-name.up.railway.app/health
```

Should return:
```json
{"status": "healthy", "model_loaded": true}
```

## Summary

**Easiest for your friend:**
1. You deploy the API to Railway/cloud
2. Share the API URL with them
3. They use `fetch()` calls in their UI to get predictions
4. No model files needed on their end!
