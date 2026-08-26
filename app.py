# ============================================================
# API de inferencia - Atenciones de Urgencia
# Trabajo Final - Tópicos en Data Science II
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import json
from datetime import datetime
import os


# ------------------------------------------------------------
# 1. Configuración de la API
# ------------------------------------------------------------

app = FastAPI(
    title="API MLOps - Atenciones de Urgencia",
    description="Servicio local de predicción semanal de atenciones de urgencia",
    version="1.0"
)


# ------------------------------------------------------------
# 2. Carga del modelo y variables
# ------------------------------------------------------------

MODELO_PATH = "modelo_urgencias.joblib"
FEATURES_PATH = "features.json"

modelo = None
features_modelo = None

if os.path.exists(MODELO_PATH) and os.path.exists(FEATURES_PATH):

    modelo = joblib.load(MODELO_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as archivo:
        features_modelo = json.load(archivo)


# ------------------------------------------------------------
# 3. Estructura de datos recibidos
# ------------------------------------------------------------

class DatosPrediccion(BaseModel):

    semana_anio: int
    mes: int
    lag_1: float
    lag_2: float
    lag_4: float
    media_4_sem: float


# ------------------------------------------------------------
# 4. Endpoint de estado
# ------------------------------------------------------------

@app.get("/ping")
def ping():

    return {
        "estado": "ok",
        "modelo_cargado": modelo is not None
    }


# ------------------------------------------------------------
# 5. Endpoint de predicción
# ------------------------------------------------------------

@app.post("/predict")
def predict(datos: DatosPrediccion):

    if modelo is None:

        return {
            "error": "El modelo todavía no está disponible."
        }

    entrada = pd.DataFrame(
        [{
            "semana_anio": datos.semana_anio,
            "mes": datos.mes,
            "lag_1": datos.lag_1,
            "lag_2": datos.lag_2,
            "lag_4": datos.lag_4,
            "media_4_sem": datos.media_4_sem
        }]
    )

    prediccion = modelo.predict(entrada)[0]


    # --------------------------------------------------------
    # Registro de la predicción
    # --------------------------------------------------------

    registro = entrada.copy()

    registro["timestamp"] = datetime.now().isoformat()
    registro["prediccion"] = prediccion

    archivo_log = "predicciones_api.csv"

    registro.to_csv(
        archivo_log,
        mode="a",
        header=not os.path.exists(archivo_log),
        index=False
    )


    return {
        "prediccion_atenciones": round(float(prediccion), 0)
    }