from typing import Dict, List, Tuple, Optional

import numpy as np
import xgboost as xgb

# IMPORTANT: this must match training feature order/names
EXPECTED_FEATURES: List[str] = [
    "Provider Organization Name (Legal Business Name)_complete",
    "Employer Identification Number (EIN)_complete",
    "Provider First Line Business Practice Location Address_complete",
    "Provider Business Practice Location Address City Name_complete",
    "Provider Business Practice Location Address State Name_complete",
    "Provider Business Practice Location Address Postal Code_complete",
    "Provider Business Practice Location Address Telephone Number_complete",
    "Healthcare Provider Taxonomy Code_1_complete",
    "Provider License Number_1_complete",
    "Provider License Number State Code_1_complete",
    "data_completeness_score",
    "data_completeness_score.1",
    "missing_critical_fields",
    "num_taxonomy_codes",
    "hospital_type",
    "num_licenses",
    "has_primary_license",
    "license_state_match",
    "days_since_enumeration",
    "days_since_update",
    "recently_updated",
    "is_subpart",
    "has_parent_org",
    "state",
    "region",
]

_BOOSTER: Optional[xgb.Booster] = None


def load_model(model_path: str) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster


def init_model(model_path: str):
    global _BOOSTER
    if _BOOSTER is None:
        _BOOSTER = load_model(model_path)


def _vectorize(features: Dict[str, float]) -> Tuple[np.ndarray, List[str]]:
    cols = EXPECTED_FEATURES
    x = np.array([[float(features.get(c, 0.0)) for c in cols]], dtype=np.float32)
    return x, cols


def predict_denial_probability(features: Dict[str, float], topk: int = 5):
    if _BOOSTER is None:
        raise RuntimeError("Model not initialized")

    X, cols = _vectorize(features)
    dmat = xgb.DMatrix(X, feature_names=cols)

    proba = float(_BOOSTER.predict(dmat)[0])

    contrib = _BOOSTER.predict(dmat, pred_contribs=True)[0]
    contrib = np.array(contrib, dtype=np.float32)

    contrib_no_bias = contrib[:-1]
    pairs = list(zip(cols, contrib_no_bias.tolist()))
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    top = pairs[:topk]

    return proba, top
