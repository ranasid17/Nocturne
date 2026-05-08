# qusa/qusa/model/backtest.py

"""
Backtest overnight delta prediction model.
"""

import logging
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from qusa.model.train import prepare_model_features

logger = logging.getLogger(__name__)


class ModelBacktester:
    """
    Backtest trading strategy based on
    model predictions.
    """

    def __init__(self, model_path, backtest_data_path):
        """
        Class constructor.

        Parameters:
            1) model_path (str): Path to saved/trained model
            2) backtest_data_path (str): Path to data for backtesting
        """

        # store paths to model and dataset as attributes
        self.model_path = os.path.expanduser(model_path)
        self.backtest_data_path = os.path.expanduser(backtest_data_path)

        # load model and dataset as attributes
        self._load_model()
        self._load_data()

        self.results = None

    def _load_model(self):
        """
        Load saved/trained model from attribute path.
        """
        bundle = joblib.load(self.model_path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.threshold = bundle["threshold"]

        logger.info(f"✓ Model loaded")

    def _load_data(self):
        """
        Load backtest dataset from attribute path.
        """
        # load data from path and confirm datetime type
        self.data = pd.read_csv(self.backtest_data_path)
        self.data["date"] = pd.to_datetime(self.data["date"])

        logger.info(f"✓ Loaded {len(self.data)} days of data")

    def run_backtest(self, initial_capital, position_size, transaction_cost):
        """
        Simulate backtest with pure Overnight logic.
        Buy Close -> Sell Open next day if signal is high confidence.

        Parameters:
            1) initial_capital (float): Starting balance for backtest.
            2) position_size (float): Proportion of balance to allocate per trade.
            3) transaction_cost (float): Trading cost per side (%).
        """

        logger.info("\n" + "=" * 80)
        logger.info(f"RUNNING BACKTEST (Overnight Only | Cost: {transaction_cost}% per side)")
        logger.info("=" * 80)

        # extract features and probabilities for full dataset
        X = prepare_model_features(self.data, self.features)
        y_prob = self.model.predict_proba(X)[:, 1]

        # store relevant columns from dataset for backtest
        results = self.data[["date", "close", "overnight_delta"]].copy()
        results["probability_up"] = y_prob

        # identify high confidence signals
        results["signal"] = 0
        results.loc[results["probability_up"] >= self.threshold, "signal"] = 1
        results.loc[results["probability_up"] <= (1 - self.threshold), "signal"] = -1

        # calculate returns per trade
        # Overnight return = (Open_t+1 - Close_t) / Close_t
        results["overnight_return"] = results["overnight_delta"] / 100

        # simulate strategy
        results["strategy_return"] = 0.0
        results["trade_count"] = 0

        # trade positive signals (long)
        results.loc[results["signal"] == 1, "strategy_return"] = (
            results["overnight_return"] - (transaction_cost / 100) * 2
        ) * position_size
        results.loc[results["signal"] == 1, "trade_count"] = 1

        # trade negative signals (short)
        results.loc[results["signal"] == -1, "strategy_return"] = (
            -results["overnight_return"] - (transaction_cost / 100) * 2
        ) * position_size
        results.loc[results["signal"] == -1, "trade_count"] = 1

        # cumulative performance
        results["strategy_cumulative"] = (1 + results["strategy_return"]).cumprod()
        results["strategy_value"] = initial_capital * results["strategy_cumulative"]

        # Benchmark (Buy & Hold)
        first_close = results["close"].iloc[0]
        results["buy_hold_value"] = initial_capital * (results["close"] / first_close)

        self.results = results
        self.data = results
        return results

    def calculate_metrics(self, initial_capital):
        """
        Calculate backtest model performance.

        Parameters:
            1) initial_capital (float): Starting balance for backtest.

        Returns:
            1) metrics (dict): Dictionary with backtest metrics.
        """

        if self.results is None:
            raise ValueError("Backtest has not been run yet.")

        logger.info("\n" + "=" * 80)
        logger.info("PERFORMANCE METRICS")
        logger.info("=" * 80)

        # Basic trade metrics
        total_trades = self.results["trade_count"].sum()
        winning_trades = (
            (self.results["trade_count"] == 1) & (self.results["strategy_return"] > 0)
        ).sum()
        losing_trades = (
            (self.results["trade_count"] == 1) & (self.results["strategy_return"] < 0)
        ).sum()

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Return metrics
        final_val = self.results["strategy_value"].iloc[-1]
        bh_final_val = self.results["buy_hold_value"].iloc[-1]

        strategy_return = (final_val - initial_capital) / initial_capital
        bh_return = (bh_final_val - initial_capital) / initial_capital

        # Risk metrics
        daily_returns = self.results["strategy_return"]
        annual_vol = daily_returns.std() * np.sqrt(252)
        sharpe = (
            (daily_returns.mean() / daily_returns.std() * np.sqrt(252))
            if daily_returns.std() > 0
            else 0
        )

        # Max drawdown
        peak = self.results["strategy_value"].cummax()
        drawdown = (self.results["strategy_value"] - peak) / peak
        max_dd = drawdown.min()

        metrics = {
            "initial_capital": initial_capital,
            "final_value": final_val,
            "strategy_value": final_val,
            "buy_hold_value": bh_final_val,
            "strategy_return": strategy_return,
            "buy_hold_return": bh_return,
            "alpha": strategy_return - bh_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "max_draw_down": max_dd,
            "total_trades": int(total_trades),
            "winning_trades": int(winning_trades),
            "losing_trades": int(losing_trades),
            "win_rate": win_rate,
            "loss_rate": 1 - win_rate if total_trades > 0 else 0,
        }

        return metrics

    def print_summary(self, metrics):
        """
        Pretty print summary metrics.

        Parameters:
            1) metrics (dict): fill here
        """

        logger.info("\n" + "=" * 30)
        logger.info(" BACKTEST SUMMARY")
        logger.info("=" * 30)
        logger.info(f"Total Trades:       {metrics['total_trades']}")
        logger.info(f"Winning Trades:     {metrics['winning_trades']}")
        logger.info(f"Losing Trades:      {metrics['losing_trades']}")
        logger.info(f"Win Rate:           {metrics['win_rate'] * 100:.2f}%")
        logger.info(f"Loss Rate:          {metrics['loss_rate'] * 100:.2f}%")
        logger.info("-" * 30)
        logger.info(f"Strategy Return:    {metrics['strategy_return'] * 100:.2f}%")
        logger.info(f"Buy & Hold Return:  {metrics['buy_hold_return'] * 100:.2f}%")
        logger.info(f"Alpha:              {metrics['alpha']:.2f}")
        logger.info("-" * 30)
        logger.info(f"Final Strategy Val: ${metrics['strategy_value']:,.2f}")
        logger.info(f"Final Buy & Hold:   ${metrics['buy_hold_value']:,.2f}")
        logger.info("-" * 30)
        logger.info(f"Annual Volatility:  {metrics['annual_volatility']:.4f}")
        logger.info(f"Sharpe Ratio:       {metrics['sharpe_ratio']:.4f}")
        logger.info(f"Max Draw down:      {metrics['max_draw_down'] * 100:.2f}%")
        logger.info("=" * 30)

        return

    def plot_results(self, save_path):
        """
        Plot strategy value over time.

        Parameters:
            1) save_path (str): fill here
        """

        if self.results is None:
            raise ValueError("Backtest has not been run yet.")

        plt.figure(figsize=(12, 6))

        # strategy value
        plt.plot(
            self.results["date"],
            self.results["strategy_value"],
            label="Overnight Strategy",
            color="#274C77",
            linewidth=2,
        )

        # benchmark
        plt.plot(
            self.results["date"],
            self.results["buy_hold_value"],
            label="Buy & Hold (Benchmark)",
            color="#8B90AF",
            linestyle="--",
            alpha=0.7,
        )

        # chart formatting
        plt.title("Strategy Performance vs Benchmark", fontsize=14, fontweight="bold")
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Portfolio Value ($)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper left")
        plt.tight_layout()

        # save plot
        save_path = os.path.expanduser(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"\n✓ Results saved to {save_path}")
