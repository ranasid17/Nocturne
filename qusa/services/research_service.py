"""Importable training, evaluation, and backtest workflow services."""

from pathlib import Path

from qusa.model.backtest import ModelBacktester
from qusa.model.evaluate import evaluate_model
from qusa.model.train import train_model, validate_training_config
from qusa.storage.runs import RunRepository


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


def run_model_workflow(ticker, config, volatility_override=None, logger=None, repository=None):
    """Run enabled research phases and return their structured outcomes."""

    ticker = ticker.upper()
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
    pipeline = config.get("pipeline", {})

    try:
        if pipeline.get("skip_training", False):
            result["phases"]["training"] = {"status": "skipped"}
        elif not data_path.exists():
            result["phases"]["training"] = {"status": "failed", "error": "Processed data is missing."}
        else:
            model = train_model(str(data_path), str(model_path), _model_config(config))
            metrics = getattr(model, "metrics", {})
            repository.record_model(model_path.name, model_path, {"training_metrics": metrics})
            repository.record_artifact(run["id"], "model", model_path)
            result["phases"]["training"] = {"status": "succeeded", "metrics": metrics}

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
        result["success"] = not failures
        repository.transition_run(run["id"], "succeeded" if result["success"] else "failed", error_message=failures[0].get("error") if failures else None)
        return result
    except Exception as exc:
        repository.transition_run(run["id"], "failed", error_message=exc)
        raise
