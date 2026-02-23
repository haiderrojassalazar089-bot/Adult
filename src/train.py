import pandas as pd
from pathlib import Path
import joblib
import json
import logging
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

import mlflow
import mlflow.sklearn

# CONFIG LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

RAW_PATH = Path("data/raw")
ARTIFACTS_PATH = Path("artifacts")
MODELS_PATH = Path("models")

FEATURES_PATH = RAW_PATH / "features.parquet"
TARGETS_PATH = RAW_PATH / "targets.parquet"
PREPROCESSOR_PATH = ARTIFACTS_PATH / "preprocessor.joblib"

MODELS_PATH.mkdir(exist_ok=True, parents=True)

MODEL_PATH = MODELS_PATH / "income_classifier.joblib"
TRAIN_METADATA_PATH = ARTIFACTS_PATH / "training_metadata.json"

def main():
    logging.info("🚀 Iniciando entrenamiento del modelo...")

    # -------- Cargar datos --------
    X = pd.read_parquet(FEATURES_PATH)
    y = pd.read_parquet(TARGETS_PATH)["income"]

    # -------- Cargar preprocesador --------
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # -------- Split --------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -------- Transform --------
    X_train_t = preprocessor.transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    # -------- Modelo --------
    model = GradientBoostingClassifier(
        random_state=42,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3
    )

    # -------- MLflow --------
    mlflow.set_experiment("adult_income_training")

    with mlflow.start_run():
        mlflow.log_params({
            "model_type": "GradientBoostingClassifier",
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "test_size": 0.2,
            "random_state": 42
        })

        model.fit(X_train_t, y_train)

        y_pred = model.predict(X_val_t)
        y_proba = model.predict_proba(X_val_t)[:, 1]

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, pos_label=">50K")
        roc = roc_auc_score((y_val == ">50K").astype(int), y_proba)

        mlflow.log_metrics({
            "accuracy": acc,
            "f1_score": f1,
            "roc_auc": roc
        })

        mlflow.sklearn.log_model(model, "model")

    # -------- Guardar modelo --------
    joblib.dump(model, MODEL_PATH)
    logging.info(f"Modelo guardado en {MODEL_PATH}")

    # -------- Metadata --------
    metadata = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "model_path": str(MODEL_PATH),
        "metrics": {
            "accuracy": round(float(acc), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc), 4)
        },
        "data_split": {
            "train_size": int(len(X_train)),
            "val_size": int(len(X_val)),
            "random_state": 42
        }
    }

    with open(TRAIN_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    logging.info("📄 Metadata de entrenamiento generada.")
    logging.info("🎉 Entrenamiento finalizado correctamente.")

if __name__ == "__main__":
    main()
