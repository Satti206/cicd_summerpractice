from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

print("🔍 Current working dir:", os.getcwd())
print("📁 Available files in models/:", os.listdir("models"))


app = FastAPI(title="Uber Ride Volume Predictor")

# Загружаем модель при запуске
model = joblib.load("models/uber_model.joblib")


# Входные данные
class UberFeatures(BaseModel):
    hour: int  # 0–23
    day_of_week: int  # 0–6
    month: int  # 1–12


@app.post("/predict")
def predict(data: UberFeatures):
    try:
        # Преобразуем входные данные в DataFrame
        df = pd.DataFrame([data.dict()])

        # Проверим, что модельные признаки совпадают
        X = df.reindex(columns=model.feature_names_in_, fill_value=0)

        # Получаем предсказание
        prediction = model.predict(X)[0]
        return {"predicted_rides": round(prediction, 2)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
