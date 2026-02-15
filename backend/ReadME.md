## HealthScore AI - Model API

### 1) Put your model file here:
`backend/models/xgb_model.json` (or .bst)

### 2) Configure env
Copy `.env.example` to `.env` and set MODEL_PATH if needed.

### 3) Install + run
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
