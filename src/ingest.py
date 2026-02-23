"""
Módulo: ingest.py

Responsabilidad:
Leer features.csv y targets.csv,
realizar limpieza básica,
normalizar el target,
convertir a parquet,
y generar reporte JSON de ingesta.
"""

import pandas as pd
from pathlib import Path
import logging
import json
from datetime import datetime, UTC


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def ingest_data(
    input_dir: str = "data/raw",
    artifacts_dir: str = "artifacts"
) -> dict:

    logging.info("Iniciando proceso de ingesta desde CSV...")

    input_path = Path(input_dir)
    artifacts_path = Path(artifacts_dir)  # ✅ ahora sí definido correctamente

    features_csv = input_path / "features.csv"
    targets_csv = input_path / "targets.csv"

    if not features_csv.exists() or not targets_csv.exists():
        raise FileNotFoundError(
            "No se encontraron features.csv o targets.csv en data/raw"
        )

    # --------------------------------------------------
    # 1. CARGA
    # --------------------------------------------------
    X = pd.read_csv(features_csv)
    y = pd.read_csv(targets_csv)

    logging.info(f"Features shape: {X.shape}")
    logging.info(f"Targets shape: {y.shape}")

    # Si el target tiene más de una columna, tomar la primera
    # Si hay una columna llamada "income", usarla
    if "income" in y.columns:
        y = y[["income"]]
    else:
       # eliminar columnas tipo índice si existen
        y = y.loc[:, ~y.columns.str.contains("^Unnamed")]
        if y.shape[1] != 1:
           raise ValueError("No se pudo identificar correctamente la columna target.")

    original_shape = X.shape

    # --------------------------------------------------
    # 2. LIMPIEZA STRINGS
    # --------------------------------------------------
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].str.strip()

    target_col = y.columns[0]

    y[target_col] = (
        y[target_col]
        .astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
    )

    # --------------------------------------------------
    # 3. MÉTRICAS DE CALIDAD
    # --------------------------------------------------
    null_counts = X.isnull().sum()
    duplicates = int(X.duplicated().sum())
    target_classes = sorted(list(y[target_col].unique()))

    # --------------------------------------------------
    # 4. GUARDAR PARQUET
    # --------------------------------------------------
    X.to_parquet(input_path / "features.parquet", index=False)
    y.to_parquet(input_path / "targets.parquet", index=False)

    # --------------------------------------------------
    # 5. GENERAR REPORTE JSON
    # --------------------------------------------------
    artifacts_path.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "status": "success",
        "rows": original_shape[0],
        "columns": original_shape[1],
        "duplicates_detected": duplicates,
        "null_values_per_column": {
            col: int(val)
            for col, val in null_counts.items()
            if val > 0
        },
        "target_classes": target_classes
    }

    with open(artifacts_path / "ingestion_report.json", "w") as f:
        json.dump(report, f, indent=4)

    logging.info("Ingesta finalizada correctamente.")
    logging.info("Reporte generado en artifacts/ingestion_report.json")

    return report


if __name__ == "__main__":
    ingest_data()