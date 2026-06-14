from fastapi import FastAPI
from mangum import Mangum
from pydantic import BaseModel
import joblib
import numpy as np
import os
import pandas as pd

app = FastAPI()
handler = Mangum(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    preprocessor = joblib.load(os.path.join(BASE_DIR, 'preprocessor.pkl'))
    model = joblib.load(os.path.join(BASE_DIR, 'sales_forecasting.pkl'))
except Exception as e:
    preprocessor = None
    model = None

class PredictRequest(BaseModel):
    data: list

@app.get("/")
def read_root():
    return {"message": "ML Model API is Live!", "model_loaded": model is not None}

@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        return {"error": "Model not loaded"}
    try:
        # Replace these with your actual column names in correct order
        columns = ["Product","Category", "Region", "Quantity", "Year", "Month", "Day"]
        
        input_df = pd.DataFrame([request.data], columns=columns)
        transformed_data = preprocessor.transform(input_df)
        prediction = model.predict(transformed_data)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}