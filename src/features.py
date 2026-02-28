"""
Responsabilidad:
Construir y serializar el pipeline de preprocesamiento
para el dataset Adult.

Incluye:
- Identificación automática de columnas
- Limpieza de columnas basura (e.g., Unnamed)
- Imputación de valores faltantes
- Escalamiento de variables numéricas
- Codificación robusta de variables categóricas
- Serialización del preprocesador como artefacto
- Reporte detallado de transformaciones aplicadas
- Guardado de nombres de features transformadas
- Sanity check del preprocesador
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor, num_cols, cat_cols


# ==================================================
# FEATURE ENGINEERING
# ==================================================

def run_feature_engineering(
    input_path: str = "data/raw/features.parquet",
    artifacts_dir: str = "artifacts"
):
    logging.info("🚀 Iniciando feature engineering (modo PRO)...")

    df = pd.read_parquet(input_path)

    # Limpieza de columnas basura tipo índice
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    preprocessor, num_cols, cat_cols = build_preprocessor(df)

    # Fit del preprocesador
    preprocessor.fit(df)

    # Sanity check: transformar algunas filas
    _ = preprocessor.transform(df.head(5))

    # Transformar todo para conocer dimensiones finales
    X_transformed = preprocessor.transform(df)
    n_final_features = X_transformed.shape[1]

    # Obtener nombres de features transformadas
    ohe = preprocessor.named_transformers_["cat"]["encoder"]
    num_feature_names = num_cols
    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()
    final_feature_names = num_feature_names + cat_feature_names

    # ==================================================
    # SERIALIZACIÓN
    # ==================================================

    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, artifacts_path / "preprocessor.joblib")
    logging.info("📦 Preprocesador serializado correctamente.")

    # Guardar nombres de features originales y transformadas
    with open(artifacts_path / "original_feature_order.json", "w") as f:
        json.dump(list(df.columns), f, indent=4)

    with open(artifacts_path / "transformed_feature_names.json", "w") as f:
        json.dump(final_feature_names, f, indent=4)

    # ==================================================
    # METADATA PROFESIONAL
    # ==================================================

    metadata = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "n_rows": int(len(df)),
        "n_original_features": int(df.shape[1]),
        "n_numeric_features": int(len(num_cols)),
        "n_categorical_features": int(len(cat_cols)),
        "n_final_features_after_encoding": int(n_final_features),
        "one_hot_generated_features": int(len(cat_feature_names)),
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

    logging.info("📄 Metadata de feature engineering generada.")
    logging.info(f"🎯 Features finales generadas: {n_final_features}")
    logging.info("✅ Feature engineering completado con éxito.")


if __name__ == "__main__":
    run_feature_engineering()