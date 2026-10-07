from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from model_inference import predict
def test_prediction_is_positive():
    assert predict({"Area":1200,"Bedrooms":3,"Bathrooms":2,"Age":5,"Location":"City Center","Property_Type":"Apartment"})>0
def test_invalid_area():
    try:
        predict({"Area":20,"Bedrooms":3,"Bathrooms":2,"Age":5,"Location":"City Center","Property_Type":"Apartment"})
        assert False
    except ValueError as exc: assert "Area" in str(exc)
