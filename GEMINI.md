# GEMINI.md - QUSA (Quantitative US Stock Analysis)

## Project Overview
QUSA is a Python-based quantitative analysis framework designed for feature engineering, signal identification, and pattern discovery in US equity markets. Its primary focus is on overnight price movements (close-to-open gaps), technical indicators, and unsupervised clustering to identify trading regimes.

### Main Technologies
- **Language**: Python 3.8+
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn` (Decision Trees, K-Means, DBSCAN)
- **Visualization**: `matplotlib`
- **Data Fetching**: `requests` (Polygon.io API)
- **Reporting**: `ollama` (Local LLM integration for AI-powered reports)
- **Serialization**: `joblib` (Model persistence)

### Architecture
The project follows a sequential data pipeline:
1.  **Data Fetching**: Retrieval of OHLCV data from Polygon.io.
2.  **Feature Engineering**: Calculation of technical indicators (RSI, ATR), overnight gaps, and calendar features. Includes an optional Monte Carlo simulation component.
3.  **Clustering**: Unsupervised learning to group trading days into regimes.
4.  **Modeling**: Supervised learning (Decision Trees) to predict overnight price direction.
5.  **Evaluation & Backtesting**: Rigorous testing of model performance and strategy simulation.
6.  **Inference**: Live prediction capabilities for recent trading days.

---

## Building and Running

### Prerequisites
- **API Key**: A Polygon.io API key is required, stored in a `.env` file as `POLYGON_API_KEY`.
- **Environment**: Python 3.8+ environment with dependencies installed.
- **Local LLM (Optional)**: [Ollama](https://ollama.ai) must be running with the model specified in `config.yaml` (default: `gemma3:4b`) to generate AI reports.

### Key Commands

- **Install Dependencies**:
  ```bash
  pip install -r requirements.txt
  ```

- **Fetch Latest Data**:
  ```bash
  python scripts/get_most_recent_day.py
  ```

- **Run Feature Engineering Pipeline**:
  ```bash
  python scripts/run_FE_pipeline.py <TICKER>
  ```

- **Run Clustering Analysis**:
  ```bash
  python scripts/run_clustering.py <TICKER>
  ```

- **Run Full Model Pipeline (Train/Eval/Backtest)**:
  ```bash
  python scripts/run_model_pipeline.py --tickers <TICKER1> <TICKER2>
  ```

- **Run Live Prediction**:
  ```bash
  python scripts/model_prediction.py <TICKER>
  ```

- **Run Tests**:
  ```bash
  pytest
  ```

---

## Development Conventions

### Configuration
- All project settings (data paths, hyperparameters, reporting options) are centralized in `qusa/utils/config.yaml`.
- Use the `load_config` utility in `qusa.utils.config` to access settings.

### Directory Structure & Data Management
- `data/raw/`: Original CSV files (format: `{TICKER}_{START}_{END}.csv`).
- `data/processed/`: Feature-engineered and clustered data.
- `data/figures/`: Plots, visualizations, and backtest results.
- `data/reports/`: AI-generated analysis reports (categorized by type).
- `saved_models/`: Serialized model bundles (`.pkl`).
- `logs/`: Application and experiment logs.

### Coding Style
- **Path Management**: Use `pathlib.Path` for all file system operations. The codebase typically resolves `PROJECT_ROOT` relative to the script location.
- **Logging**: Use the centralized logging setup in `qusa.utils.logger`.
- **Error Handling**: Use the logger to capture and report errors, especially in pipeline orchestration.
- **Type Safety**: While not strictly enforced with static types, internal data flows rely heavily on `pandas.DataFrame` structures with expected columns (e.g., `date`, `open`, `high`, `low`, `close`, `volume`).

### Testing
- Tests are located in the `tests/` directory.
- `test_smoke.py` contains integration and unit tests for key components like the config loader, feature pipeline, and data fetcher.
- Mocks and patches (using `unittest.mock`) are preferred for external dependencies like APIs or LLM calls.
