# Nocturne: Quantitative Overnight Regime Analysis & Alpha Discovery

A Python-based quantitative analysis framework for feature engineering, signal identification, and pattern discovery 
in US equity markets. **Nocturne** focuses on overnight price movements (close-to-open gaps), technical indicator analysis, 
and unsupervised clustering to identify trading regimes.

## Overview

Nocturne provides a comprehensive toolkit for analyzing stock market data through:

- **Feature Engineering**: Calculate technical indicators (RSI, ATR, volume metrics) and calendar-based features
- **Overnight Analysis**: Identify and analyze overnight price gaps and abnormal movements
- **Clustering Analysis**: Discover market regimes and trading patterns using K-Means and DBSCAN
- **Predictive Modeling**: Train decision tree models to predict overnight price direction
- **Backtesting**: Evaluate trading strategies with realistic transaction costs

The framework is designed for researchers and quantitative analysts who want to explore pattern-based trading 
signals beyond traditional technical analysis.

## Key Features

### Feature Engineering
- **Technical Indicators**: RSI, ATR, Volume ratios, 52-week high/low proximity, and momentum metrics.
- **Overnight Calculations**: Close-to-open gaps, abnormal movement z-scores, and gap pattern statistics.
- **Calendar Features**: Day of week, month of year, and month start/end effects.

### Unified History & Deconfliction
- **Consolidated Storage**: Maintains a single `{TICKER}_history.csv` source of truth for each ticker.
- **Automated Deconfliction**: Automatically merges new fetches with existing data, removes duplicates, and archives fragmented files.
- **Standardized CLI**: Unified `-ticker` flag across all scripts for a consistent user experience.

### Clustering Analysis
- Unsupervised learning (K-Means/DBSCAN) to group trading days into interpretable regimes.
- PCA-based visualization and feature importance ranking by cluster separation.

### Machine Learning & Backtesting
- Decision tree classifiers for overnight direction prediction with high-confidence filtering.
- Comprehensive backtesting engine with realistic costs and Sharpe/Alpha/Drawdown metrics.
- **AI-Powered Reporting**: Automated report generation using local LLMs (via Ollama).

## Getting Started

### Prerequisites

- Python 3.8+
- Polygon.io API key (for data fetching)
- Ollama (optional, for AI-powered reports)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/nocturne.git
cd nocturne
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
Create a `.env` file in the project root:
```bash
POLYGON_API_KEY=your_api_key_here
```

4. **Configure the project**:
Edit `qusa/utils/config.yaml` to customize hyperparameters and paths.

## Recommended Workflow

QUSA follows a standardized CLI pattern. You can use `-ticker` or `--ticker` interchangeably.

### 1. Research & Model Development

Use this workflow to build and evaluate a trading strategy for a ticker.

**Step A: Fetch Historical Data**
Fetch exactly the amount of history you need. Repeated fetches will be automatically deconflicted.
```bash
python scripts/fetch_data.py -ticker UPRO --days 504
```

**Step B: Generate Features**
Processes the consolidated history into engineered indicators.
```bash
python scripts/run_FE_pipeline.py -ticker UPRO
```

**Step C: Train & Backtest**
Trains the model and evaluates performance.
```bash
python scripts/run_model_pipeline.py -ticker UPRO
```

### 2. Live Prediction (One-Step)

Once a model is trained, use this command for live "overnight" prediction tests. The `--fetch` flag automates data retrieval and feature engineering in a single step.

```bash
python scripts/model_prediction.py -ticker UPRO --fetch
```

**Output**:
- Prediction direction (UP/DOWN) and confidence level.
- Historical log entry in `data/predictions/prediction_log.csv`.

---

## Detailed Usage Guide

### Fetching Data (`scripts/fetch_data.py`)
- Fetch last $N$ trading days: `python scripts/fetch_data.py -ticker AMZN --days 252`
- Fetch specific range: `python scripts/fetch_data.py -ticker AMZN --start 2024-01-01 --end 2024-05-01`
*Fragmented source files are moved to `data/raw/archive/` after consolidation.*

### Clustering Analysis (`scripts/run_clustering.py`)
Discover market regimes:
```bash
python scripts/run_clustering.py -ticker AMZN
```
**Output**: Elbow curves, PCA cluster plots, and feature heatmaps in `data/figures/`.

### Full Model Pipeline (`scripts/run_model_pipeline.py`)
Supports multiple tickers:
```bash
python scripts/run_model_pipeline.py -ticker AMZN AAPL MSFT
```
**Output**: Trained `.pkl` bundles in `saved_models/` and performance metrics in `data/figures/`.

---

## Data Pipeline Architecture

```
[Polygon.io API]
    ↓
(fetch_data.py) → [data/raw/{ticker}_history.csv] ← (Archive fragmented files)
    ↓
[Feature Engineering Pipeline]
    ↓
[data/processed/{ticker}_processed.csv]
    ↓
[Model Training & Backtesting]
    ↓
[saved_models/{ticker}_model.pkl] → [AI Reports & Figures]
```

## Configuration

Key settings in `qusa/utils/config.yaml`:

```yaml
data:
  start_date: '2023-12-01'  # Legacy default
  end_date: '2025-12-01'    # Legacy default

features:
  rsi_window: 14
  atr_window: 14

model:
  parameters:
    probability_threshold: 0.7  # Cutoff for "High Confidence" predictions

backtest:
  initial_capital: 10000
  transaction_cost: 0.05       # % cost per trade (slippage + commission)
```

## Dependencies

- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Machine learning and clustering
- `matplotlib` - Visualization
- `requests` - API communication
- `ollama` - Local LLM integration

## Disclaimer

This software is for educational and research purposes only. It is not intended as financial advice. Trading stocks involves substantial risk of loss. Past performance does not guarantee future results.
