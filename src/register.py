"""
Responsabilidad:
Registrar artefactos versionados del pipeline entrenado.

Este módulo representa la etapa de REGISTRO DE ARTEFACTOS
en el ciclo de vida MLOps.
"""

from pathlib import Path
import shutil
import json
import logging
from datetime import datetime, UTC


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def register_artifacts(
    artifacts_dir: str = "artifacts",
    registry_dir: str = "artifacts/registry"
):

    logging.info("📦 Iniciando registro de artefactos...")

    artifacts_path = Path(artifacts_dir)
    registry_path = Path(registry_dir)

    files_to_register = [
        "preprocessor.joblib",
        "training_metrics.json",
        "training_metadata.json",
        "feature_engineering_metadata.json",
        "validation_report.json",
        "ingestion_report.json"
    ]

    for fname in files_to_register:
        if not (artifacts_path / fname).exists():
            raise FileNotFoundError(f"Falta artefacto requerido: {fname}")

    run_id = datetime.now(UTC).isoformat().replace(":", "-")
    run_dir = registry_path / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for fname in files_to_register:
        shutil.copy(artifacts_path / fname, run_dir / fname)

    run_metadata = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "registered_artifacts": files_to_register
    }

    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=4)

    logging.info(f"✅ Registro completado en: {run_dir}")


if __name__ == "__main__":
    register_artifacts()