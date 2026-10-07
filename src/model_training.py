"""Train, compare, evaluate and persist house-price models."""
from pathlib import Path
import json, numpy as np, joblib, pandas as pd, sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,mean_absolute_percentage_error
from sklearn.model_selection import KFold,cross_validate,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from data_preprocessing import load_data,add_features
ROOT=Path(__file__).resolve().parents[1]; DATA_PATH=ROOT/"data/house_prices.csv"; MODEL_DIR=ROOT/"models"
def build_preprocessor(X):
    numeric=[c for c in X.columns if X[c].dtype!="object"]; categorical=[c for c in X.columns if X[c].dtype=="object"]
    return ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),numeric),
                              ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)])
def build_models():
    return {"linear_regression":LinearRegression(),
            "random_forest":RandomForestRegressor(n_estimators=400,min_samples_leaf=2,random_state=42,n_jobs=-1),
            "gradient_boosting":GradientBoostingRegressor(n_estimators=300,learning_rate=.03,max_depth=3,random_state=42,loss="huber")}
def evaluate(y_true,y_pred):
    return {"MAE":float(mean_absolute_error(y_true,y_pred)),"RMSE":float(np.sqrt(mean_squared_error(y_true,y_pred))),
            "R2":float(r2_score(y_true,y_pred)),"MAPE":float(mean_absolute_percentage_error(y_true,y_pred))}
def train():
    df=load_data(DATA_PATH); X=add_features(df.drop(columns=["Price"])); y=df["Price"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
    results={}; candidates={}
    for name,estimator in build_models().items():
        pipe=Pipeline([("preprocessor",build_preprocessor(X)),("model",estimator)])
        pipe.fit(X_train,y_train); results[name]=evaluate(y_test,pipe.predict(X_test)); candidates[name]=pipe
    best_name=max(results,key=lambda k:results[k]["R2"]); best=candidates[best_name]
    cv=KFold(n_splits=5,shuffle=True,random_state=42)
    scores=cross_validate(best,X,y,cv=cv,scoring={"r2":"r2","mae":"neg_mean_absolute_error","rmse":"neg_root_mean_squared_error"},n_jobs=-1)
    cv_summary={"r2_mean":float(scores["test_r2"].mean()),"r2_std":float(scores["test_r2"].std()),
                "mae_mean":float((-scores["test_mae"]).mean()),"rmse_mean":float((-scores["test_rmse"]).mean())}
    best.fit(X,y); MODEL_DIR.mkdir(exist_ok=True); version="1.0.0"
    joblib.dump(best,MODEL_DIR/f"house_price_model_v{version}.joblib")
    meta={"model_version":version,"algorithm":type(best.named_steps["model"]).__name__,"target":"Price",
          "metrics_test":results[best_name],"cross_validation":cv_summary,
          "allowed_locations":sorted(df["Location"].unique().tolist()),"allowed_property_types":sorted(df["Property_Type"].unique().tolist()),
          "scikit_learn":sklearn.__version__}
    (MODEL_DIR/f"metadata_v{version}.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(json.dumps({"best_model":best_name,"results":results,"cv":cv_summary},indent=2))
if __name__=="__main__": train()
