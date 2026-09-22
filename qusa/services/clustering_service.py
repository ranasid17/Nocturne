"""Ticker-scoped clustering workflow without CLI or interactive plotting state."""

from pathlib import Path

import pandas as pd

from qusa.analysis.clustering import ClusterAnalyzer
from qusa.storage.artifacts import atomic_write_csv
from qusa.storage.runs import RunRepository


def run_clustering_workflow(ticker, config, repository=None):
    ticker = ticker.upper()
    paths = config["data"]["paths"]
    processed_dir = Path(paths["processed_data_dir"])
    figures_dir = Path(paths["figures_dir"]) / ticker.lower()
    data_path = processed_dir / f"{ticker}_processed.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data is missing: {data_path}")
    repository = repository or RunRepository.from_config(config)
    run = repository.create_run("clustering", ticker=ticker)
    repository.transition_run(run["id"], "running")
    try:
        data = pd.read_csv(data_path)
        analyzer = ClusterAnalyzer(n_clusters=4, algorithm="kmeans")
        optimal = analyzer.find_optimal_clusters(data, max_k=8)
        clustered = analyzer.fit_clusters(data, feature_cols=None)
        output_path = processed_dir / f"{ticker}_processed_clustered.csv"
        atomic_write_csv(clustered, output_path)
        figures_dir.mkdir(parents=True, exist_ok=True)
        stats_path = figures_dir / "regime_statistics.json"
        clustered.groupby("cluster").size().rename("count").reset_index().to_json(stats_path, orient="records", indent=2)
        repository.record_artifact(run["id"], "clustered_data", output_path)
        repository.record_artifact(run["id"], "regime_statistics", stats_path)
        repository.transition_run(run["id"], "succeeded")
        return {"success": True, "run_id": run["id"], "ticker": ticker, "optimal_k": optimal["optimal_k"], "output_path": str(output_path), "statistics_path": str(stats_path)}
    except Exception as exc:
        repository.transition_run(run["id"], "failed", error_message=exc)
        raise
