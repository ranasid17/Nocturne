# QUSA: Quantitative US Stock Analysis

A Python-based quantitative analysis framework for feature engineering, signal identification, and pattern discovery 
in US equity markets. QUSA focuses on overnight price movements, technical indicator analysis, and unsupervised 
clustering to identify trading patterns.

## Overview

QUSA provides a comprehensive toolkit for analyzing stock market data through:

- **Feature Engineering**: Calculate technical indicators (RSI, ATR, volume metrics) and calendar-based features
- **Overnight Analysis**: Identify and analyze overnight price gaps and abnormal movements
- **Clustering Analysis**: Discover market regimes and trading patterns using K-Means and DBSCAN
- **Predictive Modeling**: Train decision tree models to predict overnight price direction
- **Backtesting**: Evaluate trading strategies with realistic transaction costs

The framework is designed for researchers and quantitative analysts who want to explore pattern-based trading 
signals beyond traditional technical analysis.

## Key Features

### Feature Engineering
- **Technical Indicators**:
  - Relative Strength Index (RSI) with overbought/oversold signals
  - Average True Range (ATR) for volatility measurement
  - Volume spike detection and moving averages
  - 52-week high/low proximity analysis
  - Intraday and late-day momentum calculations
  
- **Overnight Calculations**:
  - Overnight price change (close-to-open gaps)
  - Abnormal movement detection using z-scores
  - Statistical analysis of gap patterns

- **Calendar Features**:
  - Day of week effects (one-hot encoded)
  - Month of year seasonality
  - Month start/end effects (first/last 5 trading days)

### Clustering Analysis
- K-Means and DBSCAN clustering algorithms
- Automatic optimal cluster determination (elbow method + silhouette scores)
- PCA-based dimensionality reduction for visualization
- Cluster profiling and interpretation
- Feature importance ranking by cluster separation

### Machine Learning
- Decision tree classifier for overnight direction prediction
- Cross-validation and train/test split evaluation
- High-confidence prediction filtering
- Feature importance analysis
- Model interpretation and limitation detection

### Backtesting
- Pure overnight strategy simulation (buy close → sell open)
- Configurable position sizing and transaction costs
- Performance metrics: Sharpe ratio, maximum drawdown, alpha vs. buy-and-hold
- Visualization of equity curves, drawdowns, and trade distributions

### AI-Powered Reporting
- Local LLM integration (via Ollama) for intelligent report generation
- Automated training, evaluation, and backtest summaries
- Model interpretation reports with limitation analysis

## Getting Started

### Prerequisites

- Python 3.8+
- Polygon.io API key (for data fetching)
- Ollama (optional, for AI-powered reports)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/qusa.git
cd qusa
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
Edit `qusa/utils/config.yaml` to customize:
- Data paths
- Feature engineering parameters
- Model hyperparameters
- Backtesting settings
- LLM reporting options

### Optional: Enable AI Reports

To enable LLM-powered reports, install and start Ollama:

```bash
# Install Ollama (see https://ollama.ai)
curl https://ollama.ai/install.sh | sh

# Pull the model specified in config.yaml (default: gemma3:4b)
ollama pull gemma3:4b

# Start Ollama server
ollama serve
```

## Usage Guide

### 1. Fetch Raw Data

Place your stock data CSV files in `data/raw/` with the naming convention:
```
{TICKER}_{START_DATE}_{END_DATE}.csv
```

Expected columns: `date`, `open`, `high`, `low`, `close`, `volume`

Or use the provided fetcher script for the most recent trading day:
```bash
python scripts/get_most_recent_day.py
```

### 2. Feature Engineering Pipeline

Generate all technical indicators, overnight calculations, and calendar features:

```bash
python scripts/run_FE_pipeline.py AMZN
```

**Output**: `data/processed/{ticker}_processed.csv` with 30+ engineered features

### 3. Clustering Analysis

Discover market regimes and trading patterns:

```bash
python scripts/run_clustering.py AMZN
```

**Output**:
- `data/figures/elbow_curve.png` - Optimal cluster selection
- `data/figures/pca_clusters.png` - Cluster visualization
- `data/figures/cluster_profiles_heatmap.png` - Feature importance
- `data/figures/cluster_time_series.png` - Pattern evolution
- `data/processed/{ticker}_processed_clustered.csv` - Labeled data
- `data/processed/cluster_statistics.json` - Cluster metrics

### 4. Model Pipeline

Train, evaluate, and backtest the overnight direction model:

```bash
python scripts/run_model_pipeline.py AMZN
```

**Output**:
- `saved_models/{ticker}_model.pkl` - Trained model bundle
- `data/reports/training/training_report_{ticker}_{timestamp}.txt` (if AI reports enabled)
- Console metrics: accuracy, precision, recall, F1, confusion matrix
- `data/reports/evaluation/evaluation_report_{ticker}_{timestamp}.txt` (if AI reports enabled)
- `data/figures/backtest_plot_{ticker}_{timestamp}.png` - Equity curves and draw down
- `data/figures/backtest_results_{ticker}_{timestamp}.csv` - Trade-by-trade results
- `data/figures/backtest_metrics_{ticker}_{timestamp}.json` - Performance metrics
- `data/reports/backtest/backtest_report_{ticker}_{timestamp}.txt` (if AI reports enabled)

Use `pipeline.skip_training`, `pipeline.skip_evaluation`, and `pipeline.skip_backtest`
in `qusa/utils/config.yaml` to run only selected model stages.
Pass multiple tickers to process them in one run, for example
`python scripts/run_model_pipeline.py AMZN AAPL`.

### 5. Live Prediction

Make predictions on the most recent trading day:

```bash
python scripts/model_prediction.py AMZN
```

**Output**:
- Console prediction with direction, probability, and confidence level
- `data/predictions/prediction_log.csv` - Historical prediction log

**Note**: Requires `run_FE_pipeline.py` and `run_model_pipeline.py` to be executed first.

## Data Pipeline

```
data/raw/{ticker}_{start}_{end}.csv
    ↓
[Feature Engineering Pipeline]
    ↓
data/processed/{ticker}_processed.csv
    ↓
[Clustering Analysis] → data/processed/{ticker}_processed_clustered.csv
    ↓                   data/figures/cluster_*.png
[Model Training]
    ↓
saved_models/{ticker}_model.pkl
    ↓
[Evaluation] → Console metrics + AI report
    ↓
[Backtesting] → data/figures/backtest_*.{png,csv,json} + AI report
    ↓
[Live Prediction] → data/predictions/prediction_log.csv
```

## Directory Structure

```
qusa/
│
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── LICENSE                             # MIT License
├── config.yaml                         # Configuration (moved to qusa/utils/)
├── .env                                # API keys and secrets (not tracked)
├── .gitignore                          # Git ignore rules
│
├── data/                               # Data storage (gitignored)
│   ├── raw/                            # Original CSV files from Polygon.io
│   ├── processed/                      # Engineered features + cluster labels
│   ├── figures/                        # Plots and visualizations
│   ├── predictions/                    # Live prediction logs
│   └── reports/                        # AI-generated analysis reports
│
├── saved_models/                       # Trained model bundles (gitignored)
│
├── qusa/                               # Main Python package
│   ├── __init__.py
│   │
│   ├── features/                       # Feature engineering modules
│   │   ├── __init__.py
│   │   ├── overnight.py                # Overnight gap calculations
│   │   ├── technical.py                # Technical indicators (RSI, ATR, etc.)
│   │   ├── calendar.py                 # Calendar/temporal features
│   │   └── pipeline.py                 # Unified feature pipeline
│   │
│   ├── analysis/                       # Pattern discovery and clustering
│   │   ├── __init__.py
│   │   └── clustering.py               # K-Means/DBSCAN clustering
│   │
│   ├── model/                          # Machine learning models
│   │   ├── __init__.py
│   │   ├── train.py                    # Model training logic
│   │   ├── evaluate.py                 # Model evaluation metrics
│   │   ├── predict.py                  # Live prediction interface
│   │   ├── backtest.py                 # Strategy backtesting engine
│   │   ├── interpreter.py              # Model interpretation and explainability
│   │   ├── reporter.py                 # LLM-powered report generation
│   │   └── reports.py                  # Report convenience functions
│   │
│   ├── data/                           # Data utilities (placeholders)
│   │   ├── __init__.py
│   │   ├── fetcher.py
│   │   └── loader.py
│   │
│   └── utils/                          # Configuration and logging
│       ├── __init__.py
│       ├── config.py                   # YAML config loader
│       ├── config.yaml                 # Main configuration file
│       └── logger.py                   # Logging setup
│
└── scripts/                            # Executable workflow scripts
    ├── __init__.py
    ├── get_most_recent_day.py          # Fetch latest OHLCV from Polygon.io
    ├── run_FE_pipeline.py              # Feature engineering orchestration
    ├── run_clustering.py               # Clustering analysis + visualizations
    ├── model_prediction.py             # Make live predictions
    └── run_model_pipeline.py           # Full modeling workflow
```

## Configuration

Key settings in `qusa/utils/config.yaml`:

```yaml
data:
  start_date: '2023-12-01'
  end_date: '2025-12-01'

features:
  rsi_window: 14
  atr_window: 14
  volume_ma_window: 20
  rolling_window_52w: 252

model:
  parameters:
    max_depth: 10
    min_samples_leaf: 10
    probability_threshold: 0.7          # High-confidence cutoff

backtest:
  initial_capital: 10000
  position_size: 0.95                   # 95% of capital per trade
  transaction_cost: 0.05                # 0.05% per side

reporting:
  enabled: true                         # Enable/disable AI reports
  llm:
    model: "gemma3:4b"                  # Ollama model
    base_url: "http://localhost:11434"
```

## Example Output

### Clustering Analysis
```
CLUSTER 0: "High Volume Spike with Significant Overnight Change"
  Size: 45 days (12.3%)
  Avg overnight delta: 2.15%
  Avg volume ratio: 2.8x
  Avg RSI: 65.2

CLUSTER 1: "Low Volatility, Stable Trading Day"
  Size: 180 days (49.1%)
  Avg overnight delta: 0.05%
  Avg volume ratio: 0.9x
  Avg RSI: 48.5
```

### Model Performance
```
Test Accuracy: 0.xxx
Precision: 0.yyy
Recall: 0.uuu
F1 Score: 0.vvv

High-Confidence Predictions (>= 0.70):
  Coverage: ww.w%
  Accuracy: 0.zzz
```

### Backtest Results
```
Strategy Return:    aa.aa%
Buy & Hold Return:  b.bb%
Alpha:              c.cc%

Annual Volatility:  0.dddd
Sharpe Ratio:       0.ee
Max Drawdown:       -f.ff%

Total Trades:       gg
Win Rate:           hh.h%
```

## Dependencies

Core libraries:
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Machine learning and clustering
- `matplotlib` - Visualization
- `requests` - API communication
- `pyyaml` - Configuration management
- `joblib` - Model serialization
- `ollama` - Local LLM integration (optional)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational and research purposes only. It is not intended as financial advice. 
Trading stocks involves substantial risk of loss. Past performance does not guarantee future results.

## Acknowledgments

- Data provided by [Polygon.io](https://polygon.io)
- AI reports powered by [Ollama](https://ollama.ai)
