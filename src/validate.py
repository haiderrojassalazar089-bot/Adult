"""
validate.py

Validación avanzada de calidad de datos.
Incluye métricas estructurales, estadísticas y de dominio.
"""

import pandas as pd
import numpy as np
import pandera as pa
from pandera import Column, Check
from pathlib import Path
import json
import logging
from datetime import datetime, UTC
from scipy.stats import entropy


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def validate_data(
    input_dir: str = "data/raw",
    artifacts_dir: str = "artifacts"
) -> dict:

    logging.info("Iniciando validación avanzada de datos...")

    X = pd.read_parquet(f"{input_dir}/features.parquet")
    y = pd.read_parquet(f"{input_dir}/targets.parquet")

    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "status": "success",
        "errors": [],
        "warnings": [],
        "structure": {},
        "quality": {},
        "statistics": {},
        "target_analysis": {}
    }

    # --------------------------------------------------
    # 1️⃣ ESTRUCTURA
    # --------------------------------------------------
    report["structure"]["rows"] = len(X)
    report["structure"]["columns"] = len(X.columns)

    if len(X) != len(y):
        report["status"] = "failed"
        report["errors"].append("Mismatch entre filas de X e y.")

    report["structure"]["dtypes"] = {
        col: str(dtype) for col, dtype in X.dtypes.items()
    }

    # --------------------------------------------------
    # 2️⃣ CALIDAD GENERAL
    # --------------------------------------------------
    null_pct = (X.isnull().mean() * 100).round(3)
    report["quality"]["null_percentage"] = {
        col: float(val) for col, val in null_pct.items() if val > 0
    }

    high_null_5 = null_pct[null_pct > 5]
    high_null_20 = null_pct[null_pct > 20]

    if len(high_null_5) > 0:
        report["warnings"].append(
            f"Columnas con >5% nulos: {high_null_5.to_dict()}"
        )

    if len(high_null_20) > 0:
        report["status"] = "failed"
        report["errors"].append(
            f"Columnas con >20% nulos: {high_null_20.to_dict()}"
        )

    duplicates = int(X.duplicated().sum())
    report["quality"]["duplicates"] = duplicates

    # --------------------------------------------------
    # 3️⃣ NUMÉRICAS
    # --------------------------------------------------
    numeric_cols = X.select_dtypes(include=np.number).columns
    numeric_stats = {}

    for col in numeric_cols:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((X[col] < lower) | (X[col] > upper)).sum()

        numeric_stats[col] = {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "mean": float(X[col].mean()),
            "outliers_iqr_count": int(outliers)
        }

    report["statistics"]["numeric"] = numeric_stats

    # --------------------------------------------------
    # 4️⃣ CATEGÓRICAS
    # --------------------------------------------------
    cat_cols = X.select_dtypes(include="object").columns
    cat_stats = {}

    for col in cat_cols:
        vc = X[col].value_counts(normalize=True)
        rare = vc[vc < 0.01].count()

        cat_stats[col] = {
            "cardinality": int(X[col].nunique()),
            "rare_categories_<1pct": int(rare)
        }

    report["statistics"]["categorical"] = cat_stats

    # --------------------------------------------------
    # 5️⃣ TARGET
    # --------------------------------------------------
    target_col = y.columns[0]
    dist = y[target_col].value_counts(normalize=True)
    target_classes = sorted(list(dist.index))

    report["target_analysis"]["distribution"] = dist.round(4).to_dict()
    report["target_analysis"]["classes"] = target_classes
    report["target_analysis"]["imbalance_ratio"] = float(
        dist.max() / dist.min()
    )
    report["target_analysis"]["entropy"] = float(
        entropy(dist)
    )

    if set(target_classes) != {"<=50K", ">50K"}:
        report["status"] = "failed"
        report["errors"].append("Clases inesperadas en el target.")

    if dist.min() < 0.15:
        report["warnings"].append("Desbalance fuerte detectado.")

    # --------------------------------------------------
    # 6️⃣ SCORE GLOBAL DE CALIDAD
    # --------------------------------------------------
    score = 100

    score -= len(report["warnings"]) * 5
    score -= len(report["errors"]) * 20

    report["quality_score_over_100"] = max(score, 0)

    # --------------------------------------------------
    # Guardar JSON
    # --------------------------------------------------
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

    with open(f"{artifacts_dir}/validation_report.json", "w") as f:
        json.dump(report, f, indent=4)

    logging.info(f"Validación completada con estado: {report['status']}")
    logging.info(f"Quality Score: {report['quality_score_over_100']}/100")

    if report["status"] == "failed":
        raise ValueError("Validación fallida.")

    return report


if __name__ == "__main__":
    validate_data()