from ucimlrepo import fetch_ucirepo
from pathlib import Path

def ingest_adult(output_dir="data/raw"):
    adult = fetch_ucirepo(id=2)

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    adult.data.features.to_parquet(path / "features.parquet")
    adult.data.targets.to_parquet(path / "targets.parquet")

    print("Ingesta completada")

if __name__ == "__main__":
    ingest_adult()