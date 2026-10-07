"""Flask UI and REST API."""
from pathlib import Path
import sys
from flask import Flask,jsonify,render_template,request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from model_inference import load_artifacts,predict
app=Flask(__name__)
@app.get("/")
def home():
    _,meta=load_artifacts()
    return render_template("index.html",locations=meta["allowed_locations"],property_types=meta["allowed_property_types"])
@app.post("/predict")
def api_predict():
    try:
        data=request.get_json(silent=True)
        if not isinstance(data,dict): return jsonify({"error":"Request body must be a JSON object."}),400
        return jsonify({"predicted_price":round(predict(data),2),"currency":"INR"})
    except ValueError as exc: return jsonify({"error":str(exc)}),400
    except Exception:
        app.logger.exception("Prediction failed"); return jsonify({"error":"Internal prediction error."}),500
if __name__=="__main__": app.run(host="127.0.0.1",port=5000,debug=False)
