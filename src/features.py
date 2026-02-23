"""
Responsabilidad:
Construir y serializar el pipeline de preprocesamiento
para el dataset Adult.

Incluye:
- Identificación automática de columnas
- Imputación de valores faltantes
- Escalamiento de variables numéricas
- Codificación robusta de variables categóricas
- Serialización del preprocesador como artefacto
- Reporte detallado de transformaciones aplicadas
"""

import pandas as pd
from pathlib import Path
import joblib
import json
import logging
from datetime import datetime, UTC

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# ==================================================
# CONFIGURACIÓN LOGGING
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ==================================================
# IDENTIFICACIÓN DE COLUMNAS
# ==================================================

def identify_column_types(df: pd.DataFrame):

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    return num_cols, cat_cols


# ==================================================
# CONSTRUCCIÓN DEL PREPROCESSOR
# ==================================================

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:

    num_cols, cat_cols = identify_column_types(df)

    logging.info(f"Columnas numéricas detectadas: {num_cols}")
    logging.info(f"Columnas categóricas detectadas: {cat_cols}")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    return preprocessor, num_cols, cat_cols


# ==================================================
# FEATURE ENGINEERING
# ==================================================

def run_feature_engineering(
    input_path: str = "data/raw/features.parquet",
    artifacts_dir: str = "artifacts"
):

    logging.info("Iniciando feature engineering...")

    df = pd.read_parquet(input_path)

    preprocessor, num_cols, cat_cols = build_preprocessor(df)

    # Fit
    preprocessor.fit(df)

    # Transformación para calcular dimensiones finales
    X_transformed = preprocessor.transform(df)

    n_final_features = X_transformed.shape[1]

    # Obtener número de columnas generadas por OneHot
    ohe = preprocessor.named_transformers_["cat"]["encoder"]
    ohe_feature_count = len(ohe.get_feature_names_out(cat_cols))

    # ==================================================
    # SERIALIZACIÓN
    # ==================================================

    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, artifacts_path / "preprocessor.joblib")

    logging.info("Preprocesador serializado correctamente.")

    # ==================================================
    # METADATA PROFESIONAL
    # ==================================================

    metadata = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "n_rows": len(df),
        "n_original_features": df.shape[1],
        "n_numeric_features": len(num_cols),
        "n_categorical_features": len(cat_cols),
        "n_final_features_after_encoding": n_final_features,
        "one_hot_generated_features": ohe_feature_count,
        "transformations": {
            "numeric": {
                "imputation": "median",
                "scaling": "standard_scaler"
            },
            "categorical": {
                "imputation": "most_frequent",
                "encoding": "one_hot",
                "handle_unknown": "ignore"
            }
        }
    }

    with open(artifacts_path / "feature_engineering_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    logging.info("Metadata de feature engineering generada.")
    logging.info(f"Features finales generadas: {n_final_features}")


if __name__ == "__main__":
    run_feature_engineering()