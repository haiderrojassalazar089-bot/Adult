"""
Módulo: ingest.py

Responsabilidad:
Leer features.csv y targets.csv,
realizar limpieza básica,
normalizar el target,
y convertir a formato parquet.
"""

import pandas as pd
from pathlib import Path
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def ingest_data(
    input_dir: str = "data/raw",
    output_dir: str = "data/raw"
) -> None:

    logging.info("Iniciando proceso de ingesta desde CSV...")

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    features_csv = input_path / "features.csv"
    targets_csv = input_path / "targets.csv"

    if not features_csv.exists() or not targets_csv.exists():
        raise FileNotFoundError(
            "No se encontraron features.csv o targets.csv en data/raw"
        )

    # ----------------------------
    # 1. Cargar CSV
    # ----------------------------
    X = pd.read_csv(features_csv)
    y = pd.read_csv(targets_csv)

    logging.info(f"Features shape: {X.shape}")
    logging.info(f"Targets shape: {y.shape}")

    # ----------------------------
    # 2. Normalizar columnas tipo string
    # ----------------------------
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].str.strip()

    target_col = y.columns[0]

    y[target_col] = (
        y[target_col]
        .astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
    )

    # ----------------------------
    # 3. Guardar como Parquet
    # ----------------------------
    X.to_parquet(output_path / "features.parquet", index=False)
    y.to_parquet(output_path / "targets.parquet", index=False)

    logging.info("Archivos parquet generados correctamente.")


if __name__ == "__main__":
    ingest_data()