"""Validated model inference."""
from pathlib import Path
import json,joblib,pandas as pd
from data_preprocessing import add_features
ROOT=Path(__file__).resolve().parents[1]; MODEL_PATH=ROOT/"models/house_price_model_v1.0.0.joblib"; META_PATH=ROOT/"models/metadata_v1.0.0.json"
_model=None; _metadata=None
def load_artifacts():
    global _model,_metadata
    if _model is None: _model=joblib.load(MODEL_PATH); _metadata=json.loads(META_PATH.read_text(encoding="utf-8"))
    return _model,_metadata
def validate_input(data:dict):
    required=["Area","Bedrooms","Bathrooms","Age","Location","Property_Type"]; missing=[k for k in required if k not in data]
    if missing: raise ValueError(f"Missing required fields: {', '.join(missing)}")
    rules={"Area":(100,20000),"Bedrooms":(1,20),"Bathrooms":(1,20),"Age":(0,200)}
    for key,(low,high) in rules.items():
        try: value=float(data[key])
        except (TypeError,ValueError): raise ValueError(f"{key} must be numeric.")
        if not low<=value<=high: raise ValueError(f"{key} must be between {low} and {high}.")
    _,meta=load_artifacts()
    if data["Location"] not in meta["allowed_locations"]: raise ValueError(f"Location must be one of: {', '.join(meta['allowed_locations'])}")
    if data["Property_Type"] not in meta["allowed_property_types"]: raise ValueError(f"Property_Type must be one of: {', '.join(meta['allowed_property_types'])}")
def predict(data:dict)->float:
    validate_input(data); model,_=load_artifacts(); return float(model.predict(add_features(pd.DataFrame([data])))[0])
