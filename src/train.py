"""
train.py

Responsabilidad:
Entrenar un modelo de clasificación para Adult Income
usando el preprocesador ya entrenado.

Incluye:
- Split train/validación estratificado
- Modelo baseline robusto
- Métricas estándar
- Serialización de artefactos
- Metadata reproducible

Representa la etapa de ENTRENAMIENTO del pipeline MLOps.
"""

import pandas as pd
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime, UTC

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# ==================================================
# CONFIGURACIÓN LOGGING
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def train_model(
    data_dir: str = "data/raw",
    artifacts_dir: str = "artifacts",
    models_dir: str = "models",
    test_size: float = 0.2,
    random_state: int = 42
):

    logging.info("Iniciando entrenamiento del modelo (nivel pipeline MLOps)...")

    # --------------------------------------------------
    # 1️⃣ CARGA DE DATOS
    # --------------------------------------------------
    X = pd.read_parquet(f"{data_dir}/features.parquet")
    y = pd.read_parquet(f"{data_dir}/targets.parquet")

    # Guardar dataset base para monitoreo de drift
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    X.to_parquet(f"{artifacts_dir}/train_reference.parquet")
    logging.info("Dataset de referencia guardado para monitoreo.")

    target_col = y.columns[0]
    y = y[target_col]

    # --------------------------------------------------
    # 2️⃣ CARGA DEL PREPROCESADOR
    # --------------------------------------------------
    preprocessor = joblib.load(f"{artifacts_dir}/preprocessor.joblib")

    # --------------------------------------------------
    # 3️⃣ SPLIT TRAIN / VALIDATION
    # --------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logging.info(f"Train size: {len(X_train)}")
    logging.info(f"Validation size: {len(X_val)}")

    # --------------------------------------------------
    # 4️⃣ TRANSFORMACIÓN
    # --------------------------------------------------
    X_train_t = preprocessor.transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    # --------------------------------------------------
    # 5️⃣ ENTRENAMIENTO
    # --------------------------------------------------
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train_t, y_train)

    logging.info("Modelo entrenado correctamente.")

    # --------------------------------------------------
    # 6️⃣ EVALUACIÓN
    # --------------------------------------------------
    y_pred = model.predict(X_val_t)

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, pos_label=">50K"),
        "recall": recall_score(y_val, y_pred, pos_label=">50K"),
        "f1_score": f1_score(y_val, y_pred, pos_label=">50K"),
        "confusion_matrix": confusion_matrix(y_val, y_pred).tolist()
    }

    logging.info(f"Métricas de validación: {metrics}")

    # --------------------------------------------------
    # 7️⃣ SERIALIZACIÓN DE ARTEFACTOS
    # --------------------------------------------------
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    joblib.dump(model, f"{models_dir}/income_classifier.joblib")

    with open(f"{artifacts_dir}/training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    metadata = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model_type": "LogisticRegression",
        "test_size": test_size,
        "random_state": random_state,
        "n_train_samples": len(X_train),
        "n_validation_samples": len(X_val),
        "target_classes": sorted(y.unique().tolist())
    }

    with open(f"{artifacts_dir}/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    logging.info("📦 Modelo, métricas y metadata guardadas.")
    logging.info("✅ Entrenamiento finalizado con éxito.")


if __name__ == "__main__":
    train_model()