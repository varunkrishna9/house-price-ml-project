import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from data_preprocessing import add_features
def test_feature_engineering():
    df=pd.DataFrame({"Property_ID":["P1"],"Area":[1200],"Bedrooms":[3],"Bathrooms":[2],"Age":[5],"Location":["City Center"],"Property_Type":["Apartment"]})
    out=add_features(df)
    assert "Total_Rooms" in out.columns and "Area_per_Bedroom" in out.columns and "Property_ID" not in out.columns
