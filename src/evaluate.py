"""
Responsabilidad:
Evaluar el modelo entrenado sobre un conjunto de datos
independiente (hold-out o test set).

Incluye:
- Carga de artefactos serializados
- Transformación con preprocesador entrenado
- Métricas estándar
- Registro reproducible

Representa la etapa de EVALUACIÓN dentro del pipeline MLOps.
"""

import pandas as pd
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime, UTC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)


# ==================================================
# CONFIGURACIÓN LOGGING
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def evaluate_model(
    data_dir: str = "data/raw",
    artifacts_dir: str = "artifacts",
    models_dir: str = "models"
):

    logging.info("Iniciando evaluación del modelo...")

    # --------------------------------------------------
    # 1️⃣ CARGA DE DATOS
    # --------------------------------------------------
    X = pd.read_parquet(f"{data_dir}/features.parquet")
    y = pd.read_parquet(f"{data_dir}/targets.parquet")

    target_col = y.columns[0]
    y = y[target_col]

    # --------------------------------------------------
    # 2️⃣ CARGA DE ARTEFACTOS
    # --------------------------------------------------
    model = joblib.load(f"{models_dir}/income_classifier.joblib")
    preprocessor = joblib.load(f"{artifacts_dir}/preprocessor.joblib")

    logging.info("Modelo y preprocesador cargados correctamente.")

    # --------------------------------------------------
    # 3️⃣ TRANSFORMACIÓN
    # --------------------------------------------------
    X_t = preprocessor.transform(X)

    # --------------------------------------------------
    # 4️⃣ PREDICCIONES
    # --------------------------------------------------
    y_pred = model.predict(X_t)
    y_proba = model.predict_proba(X_t)[:, 1]

    # --------------------------------------------------
    # 5️⃣ MÉTRICAS
    # --------------------------------------------------
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, pos_label=">50K"),
        "recall": recall_score(y, y_pred, pos_label=">50K"),
        "f1_score": f1_score(y, y_pred, pos_label=">50K"),
        "roc_auc": roc_auc_score((y == ">50K").astype(int), y_proba),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "timestamp_utc": datetime.now(UTC).isoformat()
    }

    logging.info(f"Métricas de evaluación: {metrics}")

    # --------------------------------------------------
    # 6️⃣ GUARDAR RESULTADOS
    # --------------------------------------------------
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{artifacts_dir}/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    logging.info("Métricas de evaluación guardadas.")
    logging.info("Evaluación finalizada correctamente.")


if __name__ == "__main__":
    evaluate_model()