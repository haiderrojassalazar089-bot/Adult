"""
Servicio de inferencia para el modelo Adult Income.

Responsabilidad:
- Cargar artefactos entrenados (modelo + preprocesador)
- Recibir datos vía HTTP
- Aplicar las mismas transformaciones del entrenamiento
- Generar predicción
- Registrar predicciones para monitoreo futuro

Este módulo representa la capa de DESPLIEGUE
dentro del ciclo de vida MLOps.
"""

# ==================================================
# IMPORTS
# ==================================================

import joblib
import pandas as pd
import json
import logging

from pathlib import Path
from datetime import datetime, UTC
from fastapi import FastAPI

from src.schemas import AdultRequest


# ==================================================
# CONFIGURACIÓN DE LOGGING
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ==================================================
# INICIALIZACIÓN DE LA API
# ==================================================

app = FastAPI(
    title="Adult Income Prediction API",
    description="API para predicción de ingresos basada en modelo entrenado",
    version="1.0"
)


# ==================================================
# RUTAS DE ARTEFACTOS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_PATH = BASE_DIR / "models"
ARTIFACTS_PATH = BASE_DIR / "artifacts"

MODEL_FILE = MODELS_PATH / "income_classifier.joblib"
PREPROCESSOR_FILE = ARTIFACTS_PATH / "preprocessor.joblib"
LOG_FILE = ARTIFACTS_PATH / "predictions_log.json"


# ==================================================
# CARGA DE ARTEFACTOS ENTRENADOS
# ==================================================

try:
    model = joblib.load(MODEL_FILE)
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    logger.info("Modelo y preprocesador cargados correctamente.")
except Exception as e:
    logger.error(f"Error cargando artefactos: {e}")
    raise RuntimeError("No se pudieron cargar los artefactos del modelo.")


# ==================================================
# ENDPOINT DE SALUD
# ==================================================

@app.get("/")
def health_check():
    """
    Endpoint para verificar que la API está activa.
    """
    return {"status": "API funcionando correctamente"}


# ==================================================
# ENDPOINT DE PREDICCIÓN
# ==================================================

@app.post("/predict")
def predict(request: AdultRequest):
    """
    Genera predicción de ingreso para un individuo.

    Flujo:
    1. Convertir input a DataFrame
    2. Aplicar preprocesamiento entrenado
    3. Generar predicción
    4. Registrar resultado para monitoreo
    """

    try:
        # 1️⃣ Convertir input a DataFrame
        # 1️⃣ Convertir input a DataFrame
        input_df = pd.DataFrame([request.model_dump()])

        # 2️⃣ Renombrar columnas para que coincidan con entrenamiento
        input_df = input_df.rename(columns={
            "education_num": "education-num",
            "marital_status": "marital-status",
            "capital_gain": "capital-gain",
            "capital_loss": "capital-loss",
            "hours_per_week": "hours-per-week",
            "native_country": "native-country"
            })

        # 2️⃣ Aplicar transformaciones del entrenamiento
        X_processed = preprocessor.transform(input_df)

        # 3️⃣ Generar predicción
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0].max()

        response = {
            "prediction": str(prediction),
            "probability": float(probability),
            "timestamp_utc": datetime.now(UTC).isoformat()
        }

        # 4️⃣ Registrar predicción para monitoreo
        log_prediction(input_df, response)

        return response

    except Exception as e:
        logger.error(f"Error durante predicción: {e}")
        return {"error": "Ocurrió un error durante la predicción"}


# ==================================================
# FUNCIÓN DE LOG PARA MONITOREO
# ==================================================

def log_prediction(input_df: pd.DataFrame, response: dict):
    """
    Registra cada predicción en un archivo JSON.

    Esto permite:
    - Monitoreo de drift futuro
    - Auditoría de decisiones
    - Trazabilidad en producción
    """

    record = {
        "input": input_df.to_dict(orient="records")[0],
        "output": response
    }

    if LOG_FILE.exists():
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(record)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)

    logger.info("Predicción registrada correctamente.")