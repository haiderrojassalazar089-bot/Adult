"""
Responsabilidad:
Validar calidad e integridad del dataset Adult
antes de continuar con el pipeline MLOps.

Incluye:
- Validación estructural con Pandera
- Chequeo de nulos
- Chequeo de duplicados
- Validación de rangos
- Consistencia entre features y target
- Generación de artefacto de validación
"""

import pandas as pd
import pandera as pa
from pandera import Column, Check
from pathlib import Path
import json
import logging
from datetime import datetime, UTC


# --------------------------------------------------
# CONFIGURACIÓN DE LOGGING
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def validate_data(
    input_dir: str = "data/raw",
    artifacts_dir: str = "artifacts"
) -> dict:
    """
    Ejecuta validaciones de calidad sobre el dataset.

    Retorna
    -------
    dict
        Reporte detallado del proceso de validación.
    """

    logging.info("Iniciando validación de datos...")

    # --------------------------------------------------
    # 1. CARGA DE DATOS
    # --------------------------------------------------
    X = pd.read_parquet(f"{input_dir}/features.parquet")
    y = pd.read_parquet(f"{input_dir}/targets.parquet")

    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "status": "success",
        "errors": [],
        "warnings": []
    }

    # --------------------------------------------------
    # 2. VALIDACIÓN DE CONSISTENCIA FILAS
    # --------------------------------------------------
    if len(X) != len(y):
        report["status"] = "failed"
        report["errors"].append(
            "Features y target no tienen el mismo número de filas."
        )

    # --------------------------------------------------
    # 3. VALIDACIÓN DE DUPLICADOS
    # --------------------------------------------------
    duplicates = int(X.duplicated().sum())
    if duplicates > 0:
        report["warnings"].append(
            f"Se encontraron {duplicates} filas duplicadas."
        )

    # --------------------------------------------------
    # 4. VALIDACIÓN DE NULOS
    # --------------------------------------------------
    null_counts = X.isnull().sum()
    null_columns = null_counts[null_counts > 0]

    if len(null_columns) > 0:
        report["warnings"].append(
            f"Columnas con nulos detectadas: {null_columns.to_dict()}"
        )

    # --------------------------------------------------
    # 5. VALIDACIÓN ESTRUCTURAL CON PANDERA
    # --------------------------------------------------
    try:
        schema = pa.DataFrameSchema(
            {
                "age": Column(int, Check.in_range(17, 90)),
                "education-num": Column(int, Check.in_range(1, 16)),
            },
            strict=False  # permite columnas adicionales
        )

        schema.validate(X, lazy=True)

    except pa.errors.SchemaErrors as e:
        report["status"] = "failed"
        report["errors"].append("Error estructural detectado por Pandera.")
        report["errors"].append(str(e.failure_cases.head()))

    # --------------------------------------------------
    # 6. VALIDACIÓN DEL TARGET
    # --------------------------------------------------
    target_series = y.iloc[:, 0]
    valid_classes = set(target_series.unique())

    if not valid_classes.issubset({">50K", "<=50K"}):
        report["status"] = "failed"
        report["errors"].append(
            f"Clases inesperadas en target: {valid_classes}"
        )

    # --------------------------------------------------
    # 7. GUARDAR REPORTE
    # --------------------------------------------------
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    with open(artifacts_path / "validation_report.json", "w") as f:
        json.dump(report, f, indent=4)

    logging.info(f"Validación finalizada con estado: {report['status']}")

    if report["status"] == "failed":
        raise ValueError("La validación falló. Revisar validation_report.json")

    return report


if __name__ == "__main__":
    validate_data()