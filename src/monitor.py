"""
Script de monitoreo de drift y disparador de retrain.

Responsabilidad:
- Comparar datos de producción vs entrenamiento
- Detectar drift estadístico
- Ejecutar reentrenamiento si se supera umbral
"""

import json
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp

# =========================
# CONFIGURACIÓN
# =========================

ARTIFACTS_PATH = Path("artifacts")
REFERENCE_FILE = ARTIFACTS_PATH / "train_reference.parquet"
LOG_FILE = ARTIFACTS_PATH / "predictions_log.json"

DRIFT_THRESHOLD = 0.05  # p-value límite


# =========================
# CARGAR DATOS
# =========================

def load_reference_data():
    return pd.read_parquet(REFERENCE_FILE)


def load_production_data():
    if not LOG_FILE.exists():
        print("No hay datos de producción aún.")
        return None

    with open(LOG_FILE, "r") as f:
        logs = json.load(f)

    records = [entry["input"] for entry in logs]
    return pd.DataFrame(records)


# =========================
# DETECCIÓN DE DRIFT
# =========================

def detect_drift(reference_df, production_df):
    drift_detected = False

    numerical_cols = reference_df.select_dtypes(include=np.number).columns

    for col in numerical_cols:
        if col not in production_df.columns:
            continue

        stat, p_value = ks_2samp(reference_df[col], production_df[col])

        print(f"Columna: {col} | p-value: {p_value:.5f}")

        if p_value < DRIFT_THRESHOLD:
            print(f"⚠️ Drift detectado en {col}")
            drift_detected = True

    return drift_detected


# =========================
# RETRAIN
# =========================

def retrain():
    print("🔁 Ejecutando reentrenamiento...")
    subprocess.run(["poetry", "run", "python", "src/train.py"])
    print("✅ Retrain completado.")


# =========================
# MAIN
# =========================

def main():
    reference_df = load_reference_data()
    production_df = load_production_data()

    if production_df is None or production_df.empty:
        print("No hay datos suficientes para monitoreo.")
        return

    drift = detect_drift(reference_df, production_df)

    if drift:
        retrain()
    else:
        print("✅ No se detectó drift significativo.")


if __name__ == "__main__":
    main()