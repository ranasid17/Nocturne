# qusa/qusa/model/reporter.py

"""
Generate intelligent reports using local LLM for trading system analysis.
Includes deep model interpretation, feature analysis, and limitation detection.
"""

import logging
import requests

from datetime import datetime
from pathlib import Path


class StrategyReporter:
    """
    Generate intelligent reports for trading strategies using local LLM.
    """

    def __init__(
        self,
        config=None,
        model_name=None,
        base_url=None,
        temperature=None,
        max_tokens=None,
        output_dir=None,
        save_default=None,
        logger=None,
    ):
        """
        Initialize reporter with config or explicit parameters.

        Parameters:
            1) config (dict, optional): Configuration dictionary
            2) model_name (str, optional): LLM model name (overrides config)
            3) base_url (str, optional): Ollama API URL (overrides config)
            4) temperature (float, optional): Sampling temperature (overrides config)
            5) max_tokens (int, optional): Max response tokens (overrides config)
            6) output_dir (str, optional): Report output directory (overrides config)
            7) save_default (bool, optional): Default save behavior (overrides config)
            8) logger (logging.Logger, optional): Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

        if not config:
            raise ValueError("config is required to initialize StrategyReporter")

        # extract parameter dicts from config
        llm_config = config.get("reporting", {}).get("llm", {})
        paths_config = config.get("data", {}).get("paths", {})
        defaults_config = config.get("defaults", {}).get("reporter", {})

        # set parameters as attributes
        self.model_name = model_name or llm_config.get("model", "gemma4:e4b")
        self.base_url = base_url or llm_config.get(
            "base_url", "http://localhost:11434"
        )
        self.temperature = (
            temperature
            if temperature is not None
            else llm_config.get("temperature", 0.2)
        )
        self.max_tokens = max_tokens or llm_config.get("max_tokens", 1000)
        self.output_dir = output_dir or paths_config.get(
            "reports_dir", "~/Projects/qusa/data/reports"
        )
        self.save_default = (
            save_default
            if save_default is not None
            else defaults_config.get("save", True)
        )

        # confirm output directory path
        self.output_dir = Path(self.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        return

    def _call_llm(self, prompt, system_prompt=None):
        """
        Pass prompt to Ollama-hosted LLM with error handling.

        Parameters:
            1) prompt (str): User prompt
            2) system_prompt (str, optional): System context

        Returns:
            1) str: LLM response text
        """

        # define Ollama endpoint
        url = f"{self.base_url}/api/generate"

        # define payload for LLM
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        # handle case where user passes system prompt
        if system_prompt:
            payload["system"] = system_prompt

        try:
            # send payload to Ollama endpoint and store response as JSON
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()

            return result.get("response", "")

        except Exception as e:
            return f"[LLM Error: {str(e)}]"

    def generate_backtest_report(
        self, ticker, metrics, backtest_results=None, save=None, output_filename=None, mc_summary=None, cluster_summary=None
    ):
        """
        Generate backtest report using Ollama-hosted LLM.

        Parameters:
            1) ticker (str): Stock ticker symbol
            2) metrics (dict): Backtest performance metrics
            3) backtest_results (pd.DataFrame, optional): Full backtest results
            4) save (bool, optional): Whether to save report (uses default if None)
            5) output_filename (str, optional): Custom output filename
            6) mc_summary (str, optional): Additional Monte Carlo context
            7) cluster_summary (str, optional): Additional clustering context

        Returns:
            1) report (str): Generated report text
        """

        # define system prompt for LLM
        system_prompt = (
            "You are a quantitative trading analyst. Generate a concise, "
            "professional summary of backtest results. Focus on key metrics, "
            "risk-adjusted returns, and actionable insights."
        )

        # build metrics summary
        metrics_summary = f"""
            Ticker: {ticker}
            Strategy Return: {metrics.get('strategy_return', 0) * 100:.2f}%
            Buy & Hold Return: {metrics.get('buy_hold_return', 0) * 100:.2f}%
            Alpha: {metrics.get('alpha', 0) * 100:.2f}%
            Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}
            Max Draw_down: {metrics.get('max_draw_down', 0) * 100:.2f}%
            Total Trades: {metrics.get('total_trades', 0)}
            Win Rate: {metrics.get('win_rate', 0) * 100:.1f}%
            Annual Volatility: {metrics.get('annual_volatility', 0):.4f}
            """

        # assemble user prompt and pass to LLM
        prompt = (
            f"Analyze these backtest results and provide insights:\n\n{metrics_summary}"
        )
        
        if mc_summary:
            prompt += f"\n\nMonte Carlo Context:\n{mc_summary}"
        if cluster_summary:
            prompt += f"\n\nClustering Context (Market Regimes):\n{cluster_summary}"

        report = self._call_llm(prompt, system_prompt)

        # determine if save LLM report
        if save is not None:
            should_save = save
        else:
            should_save = self.save_default

        # handle case where user wants to save LLM report
        if should_save:
            # build filename for LLM report
            filename = (
                output_filename
                or f"backtest_report_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )

            # build path to save LLM report and write to file
            report_path = self.output_dir / filename

            with open(report_path, "w") as f:
                f.write(f"BACKTEST REPORT: {ticker}\n")
                f.write("=" * 80 + "\n\n")
                f.write(metrics_summary)
                if mc_summary:
                    f.write(f"\nMC CONTEXT:\n{mc_summary}\n")
                if cluster_summary:
                    f.write(f"\nCLUSTER CONTEXT:\n{cluster_summary}\n")
                f.write("\n" + "=" * 80 + "\n")
                f.write("ANALYSIS\n")
                f.write("=" * 80 + "\n\n")
                f.write(report)

            self.logger.info(f"✓ Report saved to {report_path}")

        return report

    def generate_evaluation_report(
        self, ticker, metrics, save=None, output_filename=None, mc_summary=None, cluster_summary=None
    ):
        """
        Generate narrative report from model evaluation metrics.

        Parameters:
            1) ticker (str): Stock ticker symbol
            2) metrics (dict): Evaluation metrics
            3) save (bool, optional): Whether to save report
            4) output_filename (str, optional): Custom output filename
            5) mc_summary (str, optional): Additional Monte Carlo context
            6) cluster_summary (str, optional): Additional clustering context

        Returns:
            1) str: Generated report text
        """

        # define system prompt for LLM
        system_prompt = (
            "You are a machine learning engineer evaluating a trading model. "
            "Provide a balanced assessment of model performance, highlighting "
            "strengths, weaknesses, and reliability concerns."
        )

        # build metrics summary
        metrics_summary = f"""
            Ticker: {ticker}
            Accuracy: {metrics.get('accuracy', 0):.3f}
            Precision: {metrics.get('precision', 0):.3f}
            Recall: {metrics.get('recall', 0):.3f}
            F1 Score: {metrics.get('f1', 0):.3f}
            High Confidence Coverage: {metrics.get('high_confidence_coverage', 0) * 100:.1f}%
            High Confidence Accuracy: {metrics.get('high_confidence_accuracy', 0):.3f}
            """

        # extract and store confusion matrix when present
        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            metrics_summary += f"""
                Confusion Matrix:
                  True Negatives:  {metrics.get('true_negatives', 0)}
                  False Positives: {metrics.get('false_positives', 0)}
                  False Negatives: {metrics.get('false_negatives', 0)}
                  True Positives:  {metrics.get('true_positives', 0)}
                """

        # assemble user prompt and pass to LLM
        prompt = f"Evaluate this trading model's performance:\n\n{metrics_summary}"
        
        if mc_summary:
            prompt += f"\n\nMonte Carlo Context:\n{mc_summary}"
        if cluster_summary:
            prompt += f"\n\nClustering Context (Market Regimes):\n{cluster_summary}"

        report = self._call_llm(prompt, system_prompt)

        # determine if save LLM report
        if save is not None:
            should_save = save
        else:
            should_save = self.save_default

        # handle case where user wants to save LLM report
        if should_save:
            # build filename for LLM report
            filename = (
                output_filename
                or f"evaluation_report_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            report_path = self.output_dir / filename

            # build path to save LLM report and write to file
            with open(report_path, "w") as f:
                f.write(f"EVALUATION REPORT: {ticker}\n")
                f.write("=" * 80 + "\n\n")
                f.write(metrics_summary)
                if mc_summary:
                    f.write(f"\nMC CONTEXT:\n{mc_summary}\n")
                if cluster_summary:
                    f.write(f"\nCLUSTER CONTEXT:\n{cluster_summary}\n")
                f.write("\n" + "=" * 80 + "\n")
                f.write("ANALYSIS\n")
                f.write("=" * 80 + "\n\n")
                f.write(report)

            self.logger.info(f"✓ Report saved to {report_path}")

        return report

    def generate_training_report(
        self, ticker, model_metrics, training_config, save=None, output_filename=None
    ):
        """
        Generate narrative report from training process.

        Parameters:
            1) ticker (str): Stock ticker symbol
            2) model_metrics (dict): Training metrics
            3) training_config (dict): Model configuration
            4) save (bool, optional): Whether to save report
            5) output_filename (str, optional): Custom output filename

        Returns:
            1) str: Generated report text
        """

        # define system prompt for LLM
        system_prompt = (
            "You are a quantitative researcher documenting model training. "
            "Summarize the training process, hyperparameters, and initial "
            "performance indicators."
        )

        # build metrics summary
        summary = f"""
            Ticker: {ticker}
            Model Type: Decision Tree Classifier
            Training Accuracy: {model_metrics.get('accuracy', 0):.3f}
            CV Score: {model_metrics.get('cv_mean', 0):.3f} (+/- {model_metrics.get('cv_std', 0):.3f})
            
            Hyperparameters:
              Max Depth: {training_config.get('max_depth', 'N/A')}
              Min Samples Leaf: {training_config.get('min_samples_leaf', 'N/A')}
              Min Samples Split: {training_config.get('min_samples_split', 'N/A')}
              Class Weight: {training_config.get('class_weight', 'N/A')}
            """

        # handle case where user wants to save LLM report
        if "feature_importance" in model_metrics:

            # sort features by descending importance
            top_features = sorted(
                model_metrics["feature_importance"].items(),
                key=lambda x: x[1],
                reverse=True,
            )

            summary += "\nTop Features:\n"
            for feat, imp in top_features:
                summary += f"  {feat}: {imp:.4f}\n"

        # assemble user prompt and pass to LLM
        prompt = f"Summarize this model training session:\n\n{summary}"
        report = self._call_llm(prompt, system_prompt)

        # determine if save LLM report
        if save is not None:
            should_save = save
        else:
            should_save = self.save_default

        # handle case where user wants to save LLM report
        if should_save:
            # build filename for LLM report
            filename = (
                output_filename
                or f"training_report_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            report_path = self.output_dir / filename

            # build path to save LLM report and write to file
            with open(report_path, "w") as f:
                f.write(f"TRAINING REPORT: {ticker}\n")
                f.write("=" * 80 + "\n\n")
                f.write(summary)
                f.write("\n" + "=" * 80 + "\n")
                f.write("ANALYSIS\n")
                f.write("=" * 80 + "\n\n")
                f.write(report)

            self.logger.info(f"✓ Report saved to {report_path}")

        return report
