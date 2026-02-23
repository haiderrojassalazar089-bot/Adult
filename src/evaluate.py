import pandas as pd
import pandera as pa
from pandera import Column, Check
from pathlib import Path
import json
from datetime import datetime

RAW_PATH = Path("data/raw")
ARTIFACTS_PATH = Path("artifacts")

FEATURES_PATH = RAW_PATH / "features.parquet"
TARGETS_PATH = RAW_PATH / "targets.parquet"
INGEST_METADATA_PATH = ARTIFACTS_PATH / "ingestion_metadata.json"
REPORT_PATH = ARTIFACTS_PATH / "validation_report.json"

ARTIFACTS_PATH.mkdir(exist_ok=True, parents=True)

ALLOWED_SEX = ["Male", "Female"]
ALLOWED_INCOME = ["<=50K", ">50K"]

def build_schema():
    return pa.DataFrameSchema({
        "age": Column(int, Check.in_range(17, 90), nullable=False),
        "workclass": Column(str, nullable=True),
        "fnlwgt": Column(int, Check.ge(0), nullable=False),
        "education": Column(str, nullable=False),
        "education-num": Column(int, Check.in_range(1, 16), nullable=False),
        "marital-status": Column(str, nullable=False),
        "occupation": Column(str, nullable=True),
        "relationship": Column(str, nullable=False),
        "race": Column(str, nullable=False),
        "sex": Column(str, Check.isin(ALLOWED_SEX), nullable=False),
        "capital-gain": Column(int, Check.ge(0), nullable=False),
        "capital-loss": Column(int, Check.ge(0), nullable=False),
        "hours-per-week": Column(int, Check.in_range(1, 99), nullable=False),
        "native-country": Column(str, nullable=True),
    })

def validate_target(y: pd.DataFrame):
    schema_y = pa.DataFrameSchema({
        "income": Column(str, Check.isin(ALLOWED_INCOME), nullable=False)
    })
    schema_y.validate(y, lazy=True)

def drift_checks(X: pd.DataFrame):
    issues = []

    # Tamaño esperado del Adult (~48k)
    if not (30_000 <= len(X) <= 60_000):
        issues.append(f"Número de filas fuera de rango esperado: {len(X)}")

    # Duplicados
    dup_pct = X.duplicated().mean()
    if dup_pct > 0.05:
        issues.append(f"Duplicados >5%: {round(dup_pct*100, 2)}%")

    # Nulos altos
    null_rates = X.isnull().mean()
    high_null_cols = null_rates[null_rates > 0.3].index.tolist()
    if high_null_cols:
        issues.append(f"Columnas con >30% nulos: {high_null_cols}")

    # Categorías críticas
    if not set(X["sex"].dropna().unique()).issubset(set(ALLOWED_SEX)):
        issues.append(f"Valores inesperados en 'sex': {set(X['sex'].unique())}")

    return issues

def validate_against_ingest_metadata(X: pd.DataFrame, y: pd.DataFrame):
    issues = []

    if INGEST_METADATA_PATH.exists():
        with open(INGEST_METADATA_PATH, "r") as f:
            meta = json.load(f)

        if meta.get("n_rows") != len(X):
            issues.append("Número de filas no coincide con ingestion_metadata.json")

        if set(meta.get("feature_names", [])) != set(X.columns):
            issues.append("Columnas no coinciden con ingestion_metadata.json")

        if meta.get("n_features") != X.shape[1]:
            issues.append("Número de features no coincide con ingestion_metadata.json")
    else:
        issues.append("No se encontró ingestion_metadata.json para validación cruzada")

    return issues

def main():
    print("🔎 Validación alineada a la ingesta (modo PRO)...")

    if not FEATURES_PATH.exists() or not TARGETS_PATH.exists():
        raise FileNotFoundError("❌ No se encontraron los parquet. Ejecuta ingest primero.")

    X = pd.read_parquet(FEATURES_PATH)
    y = pd.read_parquet(TARGETS_PATH)

    # -------- Schema --------
    schema = build_schema()
    schema.validate(X, lazy=True)
    validate_target(y)
    print("✅ Schema de features y target OK")

    # -------- Métricas --------
    duplicates_pct = float(X.duplicated().mean() * 100)
    null_pct = (X.isnull().mean() * 100).to_dict()
    target_dist = y["income"].value_counts(normalize=True).to_dict()

    # -------- Drift --------
    drift_issues = drift_checks(X)

    # -------- Consistencia con metadata de ingesta --------
    ingest_issues = validate_against_ingest_metadata(X, y)

    all_issues = drift_issues + ingest_issues

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_rows": int(len(X)),
        "duplicates_pct": round(duplicates_pct, 2),
        "null_percentage_by_column": {k: round(v, 2) for k, v in null_pct.items()},
        "target_distribution_pct": {k: round(v * 100, 2) for k, v in target_dist.items()},
        "drift_issues": drift_issues,
        "ingest_consistency_issues": ingest_issues,
        "status": "FAILED" if all_issues else "PASSED"
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    print(f"📄 Reporte de validación guardado en: {REPORT_PATH}")

    if all_issues:
        raise ValueError(f"❌ Validación falló: {all_issues}")

    print("🎉 Validación exitosa. Datos listos para feature engineering.")

if __name__ == "__main__":
    main()
