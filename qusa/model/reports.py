# qusa/qusa/analysis/reports.py

"""
Convenience functions for generating reports.
"""

from qusa.model.reporter import StrategyReporter
from qusa.model.interpreter import ModelInterpreter


def generate_backtest_report(
    ticker,
    metrics,
    backtest_results=None,
    config=None,
    save=None,
    output_filename=None,
    mc_summary=None,
    cluster_summary=None,
):
    """
    Generate backtest report using StrategyReporter.

    Parameters:
        1) ticker (str): Stock ticker symbol
        2) metrics (dict): Backtest metrics
        3) backtest_results (pd.DataFrame, optional): Full backtest results
        4) config (dict, optional): Configuration dictionary
        5) save (bool, optional): Whether to save report
        6) output_filename (str, optional): Custom output filename
        7) mc_summary (str, optional): Summary of MC features
        8) cluster_summary (str, optional): Summary of cluster regimes

    Returns:
        1) report (str): Generated report text
    """

    # initialize object and generate report
    reporter = StrategyReporter(config=config)
    report = reporter.generate_backtest_report(
        ticker=ticker,
        metrics=metrics,
        backtest_results=backtest_results,
        save=save,
        output_filename=output_filename,
        mc_summary=mc_summary,
        cluster_summary=cluster_summary,
    )

    return report


def generate_evaluation_report(
    ticker,
    metrics,
    config=None,
    save=None,
    output_filename=None,
    mc_summary=None,
    cluster_summary=None,
):
    """
    Generate evaluation report using StrategyReporter.

    Parameters:
        1) ticker (str): Stock ticker symbol
        2) metrics (dict): Evaluation metrics
        3) config (dict, optional): Configuration dictionary
        4) save (bool, optional): Whether to save report
        5) output_filename (str, optional): Custom output filename
        6) mc_summary (str, optional): Summary of MC features
        7) cluster_summary (str, optional): Summary of cluster regimes

    Returns:
        1) report (str): Generated report text
    """

    # initialize object and generate report
    reporter = StrategyReporter(config=config)
    report = reporter.generate_evaluation_report(
        ticker=ticker,
        metrics=metrics,
        save=save,
        output_filename=output_filename,
        mc_summary=mc_summary,
        cluster_summary=cluster_summary,
    )

    return report


def generate_training_report(
    ticker, model_metrics, training_config, config=None, save=None, output_filename=None
):
    """
    Generate training report using StrategyReporter.

    Parameters:
        1) ticker (str): Stock ticker symbol
        2) model_metrics (dict): Training metrics
        3) training_config (dict): Model hyperparameters
        4) config (dict, optional): Configuration dictionary
        5) save (bool, optional): Whether to save report
        6) output_filename (str, optional): Custom output filename

    Returns:
        1) report (str): Generated report text
    """

    reporter = StrategyReporter(config=config)
    report = reporter.generate_training_report(
        ticker=ticker,
        model_metrics=model_metrics,
        training_config=training_config,
        save=save,
        output_filename=output_filename,
    )
    return report


def generate_model_interpretation_report(
    model_path,
    data=None,
    evaluation_metrics=None,
    config=None,
):
    """
    Generate model interpretation using ModelInterpreter.

    Parameters:
        1) model_path (str): Path to saved model
        2) data (pd.DataFrame, optional): Training/test data for analysis
        3) evaluation_metrics (dict, optional): Evaluation metrics
        4) config (dict, optional): Configuration dictionary

    Returns:
        1) summary (dict): Interpretation summary
    """

    # initialize object and generate report
    interpreter = ModelInterpreter(model_path=model_path, config=config)
    report = interpreter.generate_interpretation_summary(
        data=data, evaluation_metrics=evaluation_metrics
    )
    # print summary
    interpreter.print_interpretation(data=data, evaluation_metrics=evaluation_metrics)

    return report
