"""Run-owned research outputs; optional reports never determine model success."""

import json
import math
from pathlib import Path
import pandas as pd

from qusa.storage.artifacts import atomic_write_csv
from qusa.utils.errors import safe_error


def json_value(value):
    if isinstance(value, pd.DataFrame):
        return json_value(value.to_dict(orient="index"))
    if isinstance(value, pd.Series):
        return json_value(value.to_dict())
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(value), default=str, allow_nan=False, indent=2), encoding="utf-8")
    return path


def save_numerical_outputs(result, config, directory, repository, backtester=None):
    run_id = result["run_id"]
    for name, phase in result["phases"].items():
        if phase["status"] != "succeeded":
            continue
        if config.get(name, {}).get("save_results", name == "training"):
            path = save_json(phase["metrics"], directory / f"{name}_metrics.json")
            repository.record_artifact(run_id, f"{name}_metrics", path)
    if backtester is not None:
        if config["backtest"].get("save_results", False):
            path = atomic_write_csv(backtester.results, directory / "backtest_results.csv")
            repository.record_artifact(run_id, "backtest_results", path)
        if config["backtest"].get("save_plots", False):
            path = directory / "backtest.png"
            directory.mkdir(parents=True, exist_ok=True)
            backtester.plot_results(str(path))
            repository.record_artifact(run_id, "backtest_plot", path)


def generate_reports(result, config, model_path, repository, backtester=None):
    reporting = config.get("reporting", {})
    if not reporting.get("enabled", False):
        return {}
    from qusa.model.reporter import StrategyReporter
    from qusa.model.interpreter import ModelInterpreter

    outcomes = {}
    paths = config["data"]["paths"]
    directory = Path(paths.get("reports_dir", Path(paths["figures_dir"]).parent / "reports")) / result["ticker"].lower() / result["run_id"]
    save = config.get("defaults", {}).get("reporter", {}).get("save", True)
    for name in reporting.get("enabled_report_types", []):
        phase_name = "training" if name == "interpretation" else name
        phase = result["phases"].get(phase_name, {})
        if phase.get("status") != "succeeded":
            outcomes[name] = {"status": "skipped"}
            continue
        try:
            if name == "interpretation":
                report = ModelInterpreter(model_path, config=config).generate_interpretation_summary(
                    evaluation_metrics=result["phases"].get("evaluation", {}).get("metrics")
                )
                path = directory / "interpretation.json"
                should_save = config.get("interpretation", {}).get("save_results", True)
                if should_save:
                    save_json(report, path)
            else:
                reporter = StrategyReporter(config=config, output_dir=directory)
                if name == "training":
                    report = reporter.generate_training_report(result["ticker"], phase["metrics"], config["model"]["parameters"], save=False)
                elif name == "evaluation":
                    report = reporter.generate_evaluation_report(result["ticker"], phase["metrics"], save=False)
                elif name == "backtest":
                    report = reporter.generate_backtest_report(result["ticker"], phase["metrics"], backtest_results=backtester.results, save=False)
                else:
                    raise ValueError("Unsupported report type.")
                path = directory / f"{name}.txt"
                should_save = save
                if should_save:
                    path.write_text(report, encoding="utf-8")
            if should_save:
                repository.record_artifact(result["run_id"], f"{name}_report", path)
            outcomes[name] = {"status": "succeeded", "saved": should_save}
        except Exception as exc:
            outcomes[name] = {"status": "failed", "error": safe_error(exc)}
    return outcomes
