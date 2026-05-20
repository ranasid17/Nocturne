#!/usr/bin/env python3
# qusa/qusa/scripts/run_model_pipeline.py

"""
Master model pipeline orchestrator for Nocturne.
Runs model training, evaluation, and backtesting for provided tickers.
"""

import argparse
import csv
import json
import sys

from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qusa.model import (
    ModelBacktester,
    evaluate_model,
    generate_backtest_report,
    generate_evaluation_report,
    generate_model_interpretation_report,
    generate_training_report,
    train_model,
)
from qusa.utils.config import load_config
from qusa.utils.logger import setup_logger
from qusa.utils.formatting import format_header, format_box


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Nocturne model training, evaluation, and backtesting."
    )
    parser.add_argument(
        "-ticker",
        "--ticker",
        "--tickers",
        dest="tickers",
        nargs="+",
        required=True,
        help="Ticker symbol(s) to process, for example -ticker AMZN AAPL",
    )
    return parser.parse_args()


def _build_model_config(config):
    """
    Build the model parameter dictionary expected by train_model().
    """

    model_params = config["model"]["parameters"]
    return {
        "max_depth": model_params.get("max_depth", 5),
        "min_samples_leaf": model_params.get("min_samples_leaf", 10),
        "min_samples_split": model_params.get("min_samples_split", 20),
        "class_weight": model_params.get("class_weight", "balanced"),
        "random_state": model_params.get("random_state", 42),
        "test_size": model_params.get("test_size", 0.25),
        "cv": model_params.get("cv", 5),
        "probability_threshold": model_params.get("probability_threshold", 0.6),
        "monte_carlo": config.get("monte_carlo", {}),
    }


def _reports_enabled(config, report_type):
    reporting_config = config.get("reporting", {})
    enabled_types = reporting_config.get("enabled_report_types", [])
    return reporting_config.get("enabled", True) and report_type in enabled_types


def _generate_optional_report(report_name, logger, report_func, **kwargs):
    try:
        report_func(**kwargs)
    except Exception as exc:
        logger.warning(f"{report_name} generation failed: {exc}")
        logger.debug("Full traceback:", exc_info=True)


def _run_training(ticker, paths, config, logger):
    data_path = paths["processed_data_dir"] / f"{ticker}_processed.csv"
    model_save_path = paths["model_output_dir"] / f"{ticker.lower()}_model.pkl"

    if not data_path.exists():
        logger.warning(f"Skipping training for {ticker}: Data not found at {data_path}")
        return False

    model_config = _build_model_config(config)
    logger.info(f"Training model for {ticker}...")
    model = train_model(
        data_path=str(data_path),
        save_path=str(model_save_path),
        config=model_config,
    )

    metrics = getattr(model, "metrics", {})
    results_box = format_box(
        [
            f"Accuracy:  {metrics.get('accuracy', 0.0):.4f}",
            f"Precision: {metrics.get('precision', 0.0):.4f}",
            f"Recall:    {metrics.get('recall', 0.0):.4f}",
            f"Model saved to: {model_save_path}"
        ],
        title=f"Training Metrics: {ticker}"
    )
    for line in results_box.split("\n"):
        logger.info(line)

    if _reports_enabled(config, "training"):
        logger.info("Generating training report...")
        _generate_optional_report(
            "Training report",
            logger,
            generate_training_report,
            ticker=ticker,
            model_metrics=metrics,
            training_config=model_config,
            config=config,
        )

    if _reports_enabled(config, "interpretation"):
        logger.info("Generating model interpretation report...")
        data = pd.read_csv(data_path)
        _generate_optional_report(
            "Model interpretation report",
            logger,
            generate_model_interpretation_report,
            model_path=str(model_save_path),
            data=data,
            evaluation_metrics=metrics,
            config=config,
        )

    return True


def _get_data_summaries(df, config):
    """
    Extract MC and Cluster summaries from DataFrame if columns exist.
    """
    mc_summary = None
    cluster_summary = None
    
    # Monte Carlo summary
    mc_cols = [col for col in df.columns if col.startswith("mc_")]
    if mc_cols:
        from qusa.features.monte_carlo import MonteCarloFeatures
        mc = MonteCarloFeatures(config=config.get("monte_carlo", {}))
        mc_summary = mc.get_feature_summary_string(df)
        
    # Cluster summary (simplified extraction from data)
    if "cluster" in df.columns:
        cluster_stats = df.groupby("cluster").agg(
            count=("cluster", "size"),
            mean_overnight=("overnight_delta_pct", "mean"),
            mean_volume=("volume_ratio", "mean")
        )
        cluster_summary = "Cluster Summary from Data:\n"
        cluster_summary += str(cluster_stats)
        
    return mc_summary, cluster_summary


def _run_evaluation(ticker, paths, config, logger):
    model_path = paths["model_output_dir"] / f"{ticker.lower()}_model.pkl"
    eval_data_path = paths["processed_data_dir"] / f"{ticker}_processed.csv"

    if not model_path.exists():
        logger.warning(f"Skipping evaluation for {ticker}: Model not found at {model_path}")
        return False
    if not eval_data_path.exists():
        logger.warning(
            f"Skipping evaluation for {ticker}: Data not found at {eval_data_path}"
        )
        return False

    logger.info(f"Evaluating model for {ticker}...")
    metrics = evaluate_model(
        model_path=str(model_path),
        eval_data_path=str(eval_data_path),
    )
    logger.info(f"Evaluation results for {ticker}: {metrics}")

    if _reports_enabled(config, "evaluation"):
        logger.info("Generating evaluation report...")
        _generate_optional_report(
            "Evaluation report",
            logger,
            generate_evaluation_report,
            ticker=ticker,
            metrics=metrics,
            config=config,
        )

    return True


def _save_backtest_artifacts(backtester, metrics, output_dir, ticker, config, logger):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"backtest_results_{ticker}_{timestamp}.csv"
    backtester.results.to_csv(csv_path, index=False)
    logger.info(f"Backtest results saved to {csv_path}")

    serialized_metrics = {
        key: (float(value) if hasattr(value, "__float__") else value)
        for key, value in metrics.items()
    }
    serialized_metrics["ticker"] = ticker
    serialized_metrics["timestamp"] = timestamp

    metrics_path = output_dir / f"backtest_metrics_{ticker}_{timestamp}.json"
    with open(metrics_path, "w") as metrics_file:
        json.dump(serialized_metrics, metrics_file, indent=4)
    logger.info(f"Backtest metrics saved to {metrics_path}")

    if config["backtest"].get("save_plots", True):
        plot_path = output_dir / f"backtest_plot_{ticker}_{timestamp}.png"
        backtester.plot_results(save_path=str(plot_path))
        logger.info(f"Backtest plot saved to {plot_path}")

    if _reports_enabled(config, "backtest"):
        logger.info("Generating backtest report...")
        _generate_optional_report(
            "Backtest report",
            logger,
            generate_backtest_report,
            ticker=ticker,
            metrics=metrics,
            backtest_results=backtester.results,
            config=config,
            output_filename=f"backtest_report_{ticker}_{timestamp}.txt",
        )


def _log_experiment_results(ticker, metrics, has_mc_features, config, logger):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mc_config = config.get("monte_carlo", {})
    mc_enabled = has_mc_features and mc_config.get("enabled", False)

    row = {
        "timestamp": timestamp,
        "experiment_name": "mc_enhanced" if has_mc_features else "baseline",
        "has_mc_features": str(has_mc_features).lower(),
        "ticker": ticker,
        "accuracy": "NA",
        "precision": "NA",
        "recall": "NA",
        "f1": "NA",
        "high_conf_acc": "NA",
        "high_conf_cov": "NA",
        "backtest_sharpe": f"{metrics.get('sharpe_ratio', 0):.4f}",
        "backtest_return": f"{metrics.get('strategy_return', 0):.4f}",
        "backtest_alpha": f"{metrics.get('alpha', 0):.4f}",
        "max_drawdown": f"{metrics.get('max_draw_down', 0):.4f}",
        "mc_enabled": str(mc_enabled).lower(),
        "mc_window_size": mc_config.get("window_size", "NA") if mc_enabled else "NA",
        "mc_iterations": mc_config.get("iterations", "NA") if mc_enabled else "NA",
        "notes": "MC POC - 7 features" if has_mc_features else "Baseline run",
    }

    log_path = PROJECT_ROOT / "logs" / "experiments.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()

    with open(log_path, "a", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info(f"Experiment logged to {log_path}")


def _run_backtest(ticker, paths, config, logger):
    model_path = paths["model_output_dir"] / f"{ticker.lower()}_model.pkl"
    data_path = paths["processed_data_dir"] / f"{ticker}_processed.csv"

    if not model_path.exists():
        logger.warning(f"Skipping backtest for {ticker}: Model not found at {model_path}")
        return False
    if not data_path.exists():
        logger.warning(f"Skipping backtest for {ticker}: Data not found at {data_path}")
        return False

    backtest_config = config["backtest"]
    backtester = ModelBacktester(
        model_path=str(model_path),
        backtest_data_path=str(data_path),
    )

    logger.info(f"Running backtest for {ticker}...")
    backtester.run_backtest(
        initial_capital=backtest_config["initial_capital"],
        position_size=backtest_config["position_size"],
        transaction_cost=backtest_config["transaction_cost"],
    )

    metrics = backtester.calculate_metrics(backtest_config["initial_capital"])
    backtest_box = format_box(
        [
            f"Strategy Return: {metrics.get('strategy_return', 0.0) * 100:.2f}%",
            f"Sharpe Ratio:    {metrics.get('sharpe_ratio', 0.0):.2f}",
            f"Max Drawdown:    {metrics.get('max_draw_down', 0.0) * 100:.2f}%",
            f"Total Trades:    {len(backtester.results.loc[backtester.results['signal'] != 0])}"
        ],
        title=f"Backtest Results: {ticker}"
    )
    for line in backtest_box.split("\n"):
        logger.info(line)

    if backtest_config.get("save_results", True):
        _save_backtest_artifacts(
            backtester=backtester,
            metrics=metrics,
            output_dir=paths["figures_dir"],
            ticker=ticker,
            config=config,
            logger=logger,
        )

    # Check if MC features are present in the model or data
    has_mc = any(f.startswith("mc_") for f in backtester.features)
    _log_experiment_results(ticker, metrics, has_mc, config, logger)

    return True


def main():
    """
    Main execution script.
    """

    args = parse_args()
    tickers = [t.upper() for t in args.tickers]

    try:
        config_path = PROJECT_ROOT / "qusa" / "utils" / "config.yaml"
        config = load_config(config_path)
        logger = setup_logger(
            "pipeline_orchestrator",
            log_file=str(PROJECT_ROOT / "logs" / "model_pipeline.log"),
        )
    except Exception as exc:
        print(f"Error loading configuration: {exc}")
        return 1

    paths = {
        "processed_data_dir": Path(config["data"]["paths"]["processed_data_dir"]).expanduser(),
        "model_output_dir": Path(config["model"]["output"]["model_output_path"]).expanduser(),
        "figures_dir": Path(config["data"]["paths"]["figures_dir"]).expanduser(),
    }
    paths["model_output_dir"].mkdir(parents=True, exist_ok=True)

    pipeline_config = config.get("pipeline", {})
    skip_training = pipeline_config.get("skip_training", False)
    skip_evaluation = pipeline_config.get("skip_evaluation", False)
    skip_backtest = pipeline_config.get("skip_backtest", False)

    for line in format_header("STARTING NOCTURNE MODEL PIPELINE").split("\n"):
        logger.info(line)
    logger.info(f"Tickers: {tickers}")
    logger.info(
        "Skips: "
        f"training={skip_training}, evaluation={skip_evaluation}, backtest={skip_backtest}"
    )

    success_count = 0
    for ticker in tickers:
        logger.info(f"\n>>> PROCESSING TICKER: {ticker} <<<")

        try:
            phase_results = []

            if skip_training:
                logger.info("Skipping training as per configuration.")
            else:
                phase_results.append(_run_training(ticker, paths, config, logger))

            if skip_evaluation:
                logger.info("Skipping evaluation as per configuration.")
            else:
                phase_results.append(_run_evaluation(ticker, paths, config, logger))

            if skip_backtest:
                logger.info("Skipping backtest as per configuration.")
            else:
                phase_results.append(_run_backtest(ticker, paths, config, logger))

            if all(phase_results) if phase_results else True:
                success_count += 1
                logger.info(f"Pipeline successful for {ticker}")
            else:
                logger.warning(f"Pipeline completed with skipped or failed phases for {ticker}")

        except Exception as exc:
            logger.error(f"Pipeline failed for {ticker}: {exc}", exc_info=True)
            continue

    for line in format_header(f"MODEL PIPELINE COMPLETE: {success_count}/{len(tickers)} SUCCESSFUL").split("\n"):
        logger.info(line)

    return 0 if success_count == len(tickers) else 1


if __name__ == "__main__":
    sys.exit(main())