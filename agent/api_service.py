import numpy as np
import joblib
from tensorflow import keras
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Grid Demand Forecast API")

# Model ve scaler, servis baslarken BIR KEZ yuklenir (basit, dusuk riskli desen)
model = keras.models.load_model("grid_model.keras")
scaler = joblib.load("grid_scaler.pkl")

WINDOW_SIZE = 24
N_FEATURES = 9  # AEP_MW, hour, day_of_week, month, is_weekend, is_holiday, avg_temp_f, HDD, CDD


class ForecastRequest(BaseModel):
    # Son 24 saatin GERCEK (olceklenmemis) degerleri, her biri 9 ozellik
    readings: list[list[float]]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: ForecastRequest):
    raw = np.array(request.readings)  # shape: (24, 9)

    if raw.shape != (WINDOW_SIZE, N_FEATURES):
        return {"error": f"Beklenen sekil (24, 9), gelen: {raw.shape}"}

    scaled = scaler.transform(raw)
    model_input = scaled.reshape(1, WINDOW_SIZE, N_FEATURES)

    pred_scaled = model.predict(model_input).flatten()[0]

    dummy_row = np.zeros((1, N_FEATURES))
    dummy_row[0, 0] = pred_scaled
    forecast_mw = scaler.inverse_transform(dummy_row)[0, 0]

    return {"forecast_mw": float(forecast_mw)}
