# qusa/qusa/model/backtest.py

"""
Backtest overnight delta prediction model.
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from qusa.model.train import prepare_model_features


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

        print(f"✓ Model loaded")

    def _load_data(self):
        """
        Load backtest dataset from attribute path.
        """
        # load data from path and confirm datetime type
        self.data = pd.read_csv(self.backtest_data_path)
        self.data["date"] = pd.to_datetime(self.data["date"])

        print(f"✓ Loaded {len(self.data)} days of data")

    def run_backtest(self, initial_capital, position_size, transaction_cost):
        """
        Simulate backtest with pure Overnight logic.
        Buy Close -> Sell Open next day if signal is high confidence.

        Parameters:
            1) initial_capital (float): Starting investment capital
            2) position_size (float): Fraction of capital to use per trade
            3) transaction_cost (float): Friction cost per side (%)
        """

        print("\n" + "=" * 80)
        print(f"RUNNING BACKTEST (Overnight Only | Cost: {transaction_cost}% per side)")
        print("=" * 80)

        # 1. Prepare Data
        x = prepare_model_features(self.data, self.features)
        predictions = self.model.predict(x)
        probabilities = self.model.predict_proba(x)[:, 1]

        results = self.data[["date", "close", "overnight_delta"]].copy()
        results["date"] = pd.to_datetime(results["date"])
        results["predicted_direction"] = predictions
        results["predicted_probability"] = probabilities

        # Confidence thresholding
        results["high_confidence"] = (
            results["predicted_probability"] >= self.threshold
        ) | (results["predicted_probability"] <= (1 - self.threshold))

        # 2. Vectorized Return Calculation
        # Since we always exit next open, we don't need a loop.

        # Determine direction multiplier: 1 if Long, -1 if Short
        # Assuming predicted_direction 1 = Long, 0 = Short
        direction_mult = np.where(results["predicted_direction"] == 1, 1, -1)

        # Calculate raw overnight return based on direction
        raw_overnight_return = results["overnight_delta"] * direction_mult

        # Apply Position Sizing
        gross_return = raw_overnight_return * position_size

        # Apply Costs: We pay cost on Entry (Close) and Exit (Open) = 2 * cost
        round_trip_cost = 2 * transaction_cost

        # Calculate Net Strategy Return (only where we traded)
        results["strategy_return"] = np.where(
            results["high_confidence"], gross_return - round_trip_cost, 0.0
        )

        # Track Trade Execution (for metrics)
        results["trade_executed"] = results["high_confidence"].astype(int)

        # Trade PnL is the same as strategy return for 1-day hold
        results["trade_pnl"] = results["strategy_return"]

        # 3. Finalize Portfolio Value
        # Cumulative Product of Daily Returns
        # Note: We divide by 100 if returns are in percentage (e.g. 1.5 for 1.5%)
        # Assuming overnight_delta is standard percentage (e.g., 0.5 = 0.5%)
        results["cumulative_return"] = (
            1 + (results["strategy_return"] / 100)
        ).cumprod()
        results["portfolio_value"] = initial_capital * results["cumulative_return"]

        # Benchmark (Buy & Hold)
        first_close = results["close"].iloc[0]
        results["buy_hold_value"] = initial_capital * (results["close"] / first_close)

        self.results = results
        return results

    def calculate_metrics(self, initial_capital):
        """
        Calculate backtest model performance.
        """

        print("\n" + "=" * 80)
        print("PERFORMANCE METRICS")
        print("=" * 80)

        if self.results is None:
            return {}

        equity = self.results["portfolio_value"]

        # Extract trades (every high confidence day is a trade now)
        completed_trades = self.results[self.results["trade_executed"] == 1]
        trade_pnls = completed_trades["trade_pnl"]

        # All daily returns (including 0s for non-trading days)
        daily_returns = self.results["strategy_return"]

        strategy_value = equity.iloc[-1]
        buy_hold_value = self.results["buy_hold_value"].iloc[-1]

        strategy_return = (strategy_value - initial_capital) / initial_capital
        buy_hold_return = (buy_hold_value - initial_capital) / initial_capital
        alpha = strategy_return - buy_hold_return

        # Win Rate Statistics
        wins = trade_pnls[trade_pnls > 0]
        losses = trade_pnls[trade_pnls < 0]
        win_rate = len(wins) / len(trade_pnls) if len(trade_pnls) > 0 else 0.0

        # Annualized Volatility (Standard Deviation of Daily Returns)
        if len(daily_returns) > 1:
            annual_volatility = (daily_returns / 100).std() * np.sqrt(252)
        else:
            annual_volatility = 0.0

        # Sharpe Ratio
        if daily_returns.std() > 0:
            sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            sharpe = 0.0

        # Drawdown
        running_max = equity.cummax()
        draw_down = (equity - running_max) / running_max
        max_draw_down = abs(draw_down.min())

        metrics = {
            "total_trades": len(trade_pnls),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": win_rate,
            "loss_rate": 1 - win_rate,
            "strategy_value": strategy_value,
            "buy_hold_value": buy_hold_value,
            "strategy_return": strategy_return,
            "buy_hold_return": buy_hold_return,
            "alpha": alpha,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe,
            "max_draw_down": max_draw_down,
        }

        self._print_metrics(metrics)

        return metrics

    @staticmethod
    def _print_metrics(metrics):
        """
        Print performance metrics to console.
        """
        print("\n" + "=" * 30)
        print(" BACKTEST SUMMARY")
        print("=" * 30)
        print(f"Total Trades:       {metrics['total_trades']}")
        print(f"Winning Trades:     {metrics['winning_trades']}")
        print(f"Losing Trades:      {metrics['losing_trades']}")
        print(f"Win Rate:           {metrics['win_rate'] * 100:.2f}%")
        print(f"Loss Rate:          {metrics['loss_rate'] * 100:.2f}%")
        print("-" * 30)
        print(f"Strategy Return:    {metrics['strategy_return'] * 100:.2f}%")
        print(f"Buy & Hold Return:  {metrics['buy_hold_return'] * 100:.2f}%")
        print(f"Alpha:              {metrics['alpha']:.2f}")
        print("-" * 30)
        print(f"Final Strategy Val: ${metrics['strategy_value']:,.2f}")
        print(f"Final Buy & Hold:   ${metrics['buy_hold_value']:,.2f}")
        print("-" * 30)
        print(f"Annual Volatility:  {metrics['annual_volatility']:.4f}")
        print(f"Sharpe Ratio:       {metrics['sharpe_ratio']:.4f}")
        print(f"Max Draw down:      {metrics['max_draw_down'] * 100:.2f}%")
        print("=" * 30)

    def plot_results(self, save_path):
        """
        Plot backtest results.
        """
        fig, ax = plt.subplots(3, 1, figsize=(12, 10))

        # plot 1: portfolio value vs buy & hold
        ax[0].plot(
            self.results["date"],
            self.results["portfolio_value"],
            label="Strategy",
            color="#0f4c5c",
            linestyle="--",
        )
        ax[0].plot(
            self.results["date"],
            self.results["buy_hold_value"],
            label="Buy & Hold",
            color="#9a031e",
        )
        ax[0].set_title("Portfolio Value", fontsize=14)
        ax[0].set_ylabel("Value ($)", fontsize=12)
        ax[0].legend()

        # plot 2: draw down
        cumulative = self.results["portfolio_value"]
        running_max = cumulative.expanding().max()
        draw_down = (cumulative - running_max) / running_max * 100
        ax[1].plot(self.results["date"], draw_down, color="#9a031e")
        ax[1].fill_between(
            self.results["date"],
            draw_down,
            0,
            where=(draw_down < 0),
            color="#ef233c",
            alpha=0.3,
        )
        ax[1].set_title("Draw Down", fontsize=14)
        ax[1].set_ylabel("Draw Down (%)", fontsize=12)

        # plot 3: trade distribution
        trades = self.results[self.results["trade_executed"] == 1]

        if not trades.empty:
            colors = trades["trade_pnl"].apply(
                lambda x: "#0f4c5c" if x > 0 else "#9a031e"
            )
            ax[2].bar(
                trades["date"],
                trades["trade_pnl"],
                color=colors,
            )

        ax[2].axhline(0, color="black", linestyle="-", linewidth=0.8)
        ax[2].set_title("Daily Trade Returns (PnL %)", fontsize=14)
        ax[2].set_xlabel("Date", fontsize=12)
        ax[2].set_ylabel("Trade Return (%)", fontsize=12)

        plt.tight_layout()

        # save plot
        save_path = os.path.expanduser(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\n✓ Results saved to {save_path}")
