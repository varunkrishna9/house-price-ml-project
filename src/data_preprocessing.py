"""Data loading and feature engineering."""
from pathlib import Path
import numpy as np
import pandas as pd
REQUIRED_COLUMNS={"Property_ID","Area","Bedrooms","Bathrooms","Age","Location","Property_Type","Price"}
def load_data(path: str|Path)->pd.DataFrame:
    df=pd.read_csv(path)
    missing=REQUIRED_COLUMNS-set(df.columns)
    if missing: raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if df.empty: raise ValueError("Dataset is empty.")
    return df
def add_features(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy().drop(columns=["Property_ID"],errors="ignore")
    out["Total_Rooms"]=out["Bedrooms"]+out["Bathrooms"]
    out["Area_per_Bedroom"]=out["Area"]/out["Bedrooms"].clip(lower=1)
    out["Bathrooms_per_Bedroom"]=out["Bathrooms"]/out["Bedrooms"].clip(lower=1)
    out["Age_Squared"]=out["Age"]**2
    out["Area_Log"]=np.log1p(out["Area"])
    return out
