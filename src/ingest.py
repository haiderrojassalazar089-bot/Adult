"""
Responsabilidad:
Descargar el dataset Adult desde UCI,
almacenarlo en formato Parquet
y generar metadata como artefacto reproducible.

Este módulo representa la etapa de INGESTA
dentro del pipeline MLOps.
"""

import pandas as pd
from ucimlrepo import fetch_ucirepo
from pathlib import Path
import logging
import json
from datetime import datetime


# CONFIGURACIÓN DE LOGGING

# Permite monitorear ejecución y errores.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def ingest_adult(
    output_dir: str = "data/raw",
    artifacts_dir: str = "artifacts"
) -> dict:
    """
    Descarga el dataset Adult desde UCI ML Repository,
    guarda features y targets en formato Parquet
    y genera un archivo de metadata como artefacto.

    Parámetros
    ----------
    output_dir : str
        Directorio donde se almacenarán los datos crudos.
    artifacts_dir : str
        Directorio donde se guardará metadata del proceso.

    Retorna
    -------
    dict
        Diccionario con información resumida del dataset.
    """

    logging.info("Iniciando ingesta del dataset Adult...")

    
    # 1. DESCARGA DEL DATASET
    
    try:
        adult = fetch_ucirepo(id=2)
    except Exception as e:
        logging.error("Error al descargar el dataset.")
        raise e

    X = adult.data.features.copy()
    y = adult.data.targets.copy()

    logging.info(f"Dataset descargado con {len(X)} filas.")

    
    # 2. VALIDACIÓN BÁSICA DE INTEGRIDAD
    
    if len(X) != len(y):
        raise ValueError(
            "Las features y el target no tienen el mismo número de filas."
        )

    
    # 3. CREACIÓN DE DIRECTORIOS
    
    raw_path = Path(output_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    
    # 4. ALMACENAMIENTO EN FORMATO PARQUET
    
    # Parquet es más eficiente que CSV y es estándar en ingeniería de datos.
    X.to_parquet(raw_path / "features.parquet", index=False)
    y.to_parquet(raw_path / "targets.parquet", index=False)

    logging.info("Datos guardados en formato Parquet.")

    
    # 5. GENERACIÓN DE METADATA COMO ARTEFACTO
    
    metadata = {
        "dataset": "Adult",
        "source": "UCI ML Repository",
        "n_rows": len(X),
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "target_distribution": y.value_counts().to_dict(),
        "ingestion_timestamp_utc": datetime.utcnow().isoformat()
    }

    with open(artifacts_path / "ingestion_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    logging.info("Metadata de ingesta generada correctamente.")

    return metadata


if __name__ == "__main__":
    ingest_adult()