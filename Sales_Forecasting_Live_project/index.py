from fastapi import FastAPI
import joblib
import numpy as np
from mangum import Mangum

app = FastAPI()

handler = Mangum(app)

preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('sales_forecasting.pkl')

@app.get("/")
def read_root():
    return{"message": "ML Model API is Live!"}

@app.post("/predict")
def predict(data: list):
    try:
        input_data = np.array(data).reshape(1,-1)
        transformed_data = preprocessor.transform(input_data)

        prediction = model.predict(transformed_data)

        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}