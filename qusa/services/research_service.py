"""Importable training, evaluation, and backtest workflow services."""

from pathlib import Path
import shutil

from qusa.model.backtest import ModelBacktester
from qusa.model.evaluate import evaluate_model
from qusa.model.train import train_model, validate_training_config
from qusa.storage.runs import RunRepository
from qusa.storage.locks import ticker_lock
from qusa.services.research_outputs import generate_reports, json_value, save_numerical_outputs
from qusa.utils.errors import safe_error


def _model_config(config):
    parameters = config["model"]["parameters"]
    return validate_training_config({
        "max_depth": parameters.get("max_depth", 5),
        "min_samples_leaf": parameters.get("min_samples_leaf", 10),
        "min_samples_split": parameters.get("min_samples_split", 20),
        "class_weight": parameters.get("class_weight", "balanced"),
        "random_state": parameters.get("random_state", 42),
        "test_size": parameters.get("test_size", 0.25),
        "cv": parameters.get("cv", 5),
        "probability_threshold": parameters.get("probability_threshold", 0.6),
        "monte_carlo": config.get("monte_carlo", {}),
        "tuning": parameters.get("tuning", {}),
    })


def run_model_workflow(ticker, config, volatility_override=None, logger=None, repository=None, _lock_held=False):
    """Run enabled research phases and return their structured outcomes."""

    ticker = ticker.upper()
    if not _lock_held:
        data_paths = config["data"]["paths"]
        lock_root = data_paths.get("raw_data_dir", data_paths["processed_data_dir"])
        with ticker_lock(lock_root, ticker):
            return run_model_workflow(ticker, config, volatility_override, logger, repository, _lock_held=True)
    paths = {
        "data": Path(config["data"]["paths"]["processed_data_dir"]),
        "models": Path(config["model"]["output"]["model_output_path"]),
        "figures": Path(config["data"]["paths"]["figures_dir"]),
    }
    paths["models"].mkdir(parents=True, exist_ok=True)
    paths["figures"].mkdir(parents=True, exist_ok=True)
    if repository is None:
        try:
            repository = RunRepository.from_config(config)
        except KeyError:
            repository = RunRepository(paths["figures"].parent / "qusa.sqlite3")
    run = repository.create_run("research", ticker=ticker)
    repository.transition_run(run["id"], "running")
    result = {"run_id": run["id"], "ticker": ticker, "success": False, "phases": {}}
    data_path = paths["data"] / f"{ticker}_processed.csv"
    model_path = paths["models"] / f"{ticker.lower()}_model.pkl"
    snapshot_path = paths["models"] / ticker.lower() / run["id"] / "model.pkl"
    output_dir = paths["figures"] / ticker.lower() / run["id"]
    backtester = None
    pipeline = config.get("pipeline", {})

    try:
        if pipeline.get("skip_training", False):
            result["phases"]["training"] = {"status": "skipped"}
        elif not data_path.exists():
            result["phases"]["training"] = {"status": "failed", "error": "Processed data is missing."}
        else:
            model = train_model(str(data_path), str(snapshot_path), _model_config(config))
            metrics = getattr(model, "metrics", {})
            repository.record_model(model.model_id, snapshot_path, {"training_metrics": metrics})
            repository.record_artifact(run["id"], "model", snapshot_path)
            temporary = model_path.with_suffix(f".{run['id']}.tmp")
            try:
                shutil.copyfile(snapshot_path, temporary)
                temporary.replace(model_path)
            finally:
                temporary.unlink(missing_ok=True)
            model_path = snapshot_path
            result["phases"]["training"] = {"status": "succeeded", "metrics": metrics}

        if model_path.exists() and model_path != snapshot_path:
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(model_path, snapshot_path)
            model_path = snapshot_path
            repository.record_artifact(run["id"], "model", snapshot_path)

        if pipeline.get("skip_evaluation", False):
            result["phases"]["evaluation"] = {"status": "skipped"}
        elif not model_path.exists() or not data_path.exists():
            result["phases"]["evaluation"] = {"status": "failed", "error": "Model or processed data is missing."}
        else:
            metrics = evaluate_model(str(model_path), str(data_path))
            result["phases"]["evaluation"] = {"status": "succeeded", "metrics": metrics}

        if pipeline.get("skip_backtest", False):
            result["phases"]["backtest"] = {"status": "skipped"}
        elif not model_path.exists() or not data_path.exists():
            result["phases"]["backtest"] = {"status": "failed", "error": "Model or processed data is missing."}
        else:
            backtester = ModelBacktester(str(model_path), str(data_path))
            outcome_end = backtester.dataset_metadata.get("training_outcome_end")
            if not outcome_end:
                result["phases"]["backtest"] = {"status": "failed", "error": "Model lacks a training outcome boundary."}
            else:
                backtest = config["backtest"]
                volatility_filter = backtest.get("volatility_filter", {"enabled": False}).copy()
                if volatility_override is not None:
                    volatility_filter.update({"enabled": True, "max_atr_pct": volatility_override})
                backtester.run_backtest(backtest["initial_capital"], backtest["position_size"], backtest["transaction_cost"], volatility_filter, minimum_outcome_date=outcome_end)
                result["phases"]["backtest"] = {"status": "succeeded", "metrics": backtester.calculate_metrics(backtest["initial_capital"])}

        failures = [phase for phase in result["phases"].values() if phase["status"] == "failed"]
        if result["phases"]["backtest"]["status"] != "succeeded":
            backtester = None
        save_numerical_outputs(result, config, output_dir, repository, backtester)
        result["reports"] = generate_reports(result, config, model_path, repository, backtester)
        result["success"] = not failures
        repository.update_metadata(run["id"], json_value(result))
        repository.transition_run(run["id"], "succeeded" if result["success"] else "failed", error_message=failures[0].get("error") if failures else None)
        return json_value(result)
    except Exception as exc:
        result["error"] = safe_error(exc)
        repository.update_metadata(run["id"], json_value(result))
        repository.transition_run(run["id"], "failed", error_message=exc)
        raise
