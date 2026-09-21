# Flask Conversion Execution Plan

## Sprint 1: Repository Initialization & Core Logic Extraction

Goal: Prepare the repository and extract importable service-layer functions from CLI scripts without changing existing CLI behavior.

### Ticket 1

- **Ticket Name:** Create Flask App Skeleton Directories
- **Objective:** Add a minimal Flask app structure that can coexist with the current CLI package.
- **Context Required:** `ls -la`, current `requirements.txt`, and current repository tree.
- **Implementation Directives:**
  - Create `web_app/`, `web_app/templates/`, `web_app/static/css/`, and `web_app/static/js/`.
  - Add empty `web_app/__init__.py`.
  - Do not move existing `qusa/` or `scripts/` files.
- **Acceptance Criteria:** Running `python -c "import web_app"` from repo root succeeds.

### Ticket 2

- **Ticket Name:** Add Application Factory
- **Objective:** Create a minimal Flask application factory for local web execution.
- **Context Required:** `requirements.txt` and an empty/new `web_app/__init__.py`.
- **Implementation Directives:**
  - Implement `create_app()` in `web_app/__init__.py`.
  - Configure Flask with template and static folders inside `web_app`.
  - Register only a simple `/health` route returning JSON.
- **Acceptance Criteria:** `python -c "from web_app import create_app; app=create_app(); print(app.test_client().get('/health').json)"` prints a healthy response.

### Ticket 3

- **Ticket Name:** Extract Feature Pipeline Service
- **Objective:** Move feature-engineering workflow logic out of `scripts/run_FE_pipeline.py` into an importable function.
- **Context Required:** `scripts/run_FE_pipeline.py`, `qusa/features/pipeline.py`, `qusa/data/loader.py`, `qusa/utils/config.py`, `qusa/utils/config.yaml`.
- **Implementation Directives:**
  - Create `qusa/services/pipeline_service.py`.
  - Add `run_feature_pipeline(ticker: str, fetch_latest: bool = False, config_path=None) -> dict`.
  - Return structured data with `ticker`, `rows`, `shape`, `output_path`, and `success`.
- **Acceptance Criteria:** Calling the service directly for a ticker with existing raw data creates/updates `data/processed/{TICKER}_processed.csv`.

### Ticket 4

- **Ticket Name:** Convert Feature CLI To Service Wrapper
- **Objective:** Keep `scripts/run_FE_pipeline.py` working while delegating business logic to the new service.
- **Context Required:** `scripts/run_FE_pipeline.py` and `qusa/services/pipeline_service.py`.
- **Implementation Directives:**
  - Leave argparse in the script only.
  - Replace duplicated pipeline execution with a call to `run_feature_pipeline`.
  - Preserve exit code behavior: return `0` on success and `1` on failure.
- **Acceptance Criteria:** `python scripts/run_FE_pipeline.py -ticker UPRO` still works as before.

### Ticket 5

- **Ticket Name:** Extract Prediction Service
- **Objective:** Create an importable prediction workflow for Flask to call without subprocesses.
- **Context Required:** `scripts/model_prediction.py`, `qusa/model/predict.py`, `qusa/data/loader.py`, `qusa/features/pipeline.py`, `qusa/utils/config.yaml`.
- **Implementation Directives:**
  - Add `make_latest_prediction(ticker: str, fetch_latest: bool = False, volatility_override=None, config_path=None) -> dict`.
  - Save prediction logs using the existing CSV format.
  - Return structured prediction data and paths used.
- **Acceptance Criteria:** Directly calling the service appends a row to `data/predictions/prediction_log.csv`.

### Ticket 6

- **Ticket Name:** Convert Prediction CLI To Service Wrapper
- **Objective:** Keep `scripts/model_prediction.py` working while delegating to the prediction service.
- **Context Required:** `scripts/model_prediction.py` and `qusa/services/prediction_service.py`.
- **Implementation Directives:**
  - Keep argparse and ticker iteration in the CLI.
  - Call `make_latest_prediction` per ticker.
  - Keep logging readable but avoid duplicating service logic.
- **Acceptance Criteria:** `python scripts/model_prediction.py -ticker UPRO` still runs and logs prediction output.

## Sprint 2: Flask Backend & API Routing

Goal: Build Flask routes that call the extracted service layer and expose JSON endpoints for the frontend.

### Ticket 1

- **Ticket Name:** Add Main Flask Entry Point
- **Objective:** Create a root-level Flask entry file for local development.
- **Context Required:** `web_app/__init__.py` and repository root listing.
- **Implementation Directives:**
  - Create root-level `app.py`.
  - Import `create_app` from `web_app`.
  - Set `app = create_app()`.
- **Acceptance Criteria:** `flask --app app run` starts without import errors.

### Ticket 2

- **Ticket Name:** Add API Blueprint
- **Objective:** Introduce a dedicated Flask blueprint for JSON API routes.
- **Context Required:** `web_app/__init__.py`.
- **Implementation Directives:**
  - Create `web_app/api.py`.
  - Define `api_bp = Blueprint("api", __name__, url_prefix="/api")`.
  - Register it inside `create_app()`.
- **Acceptance Criteria:** `/api/health` returns JSON through the blueprint.

### Ticket 3

- **Ticket Name:** Add Ticker Discovery Endpoint
- **Objective:** Return available tickers based on consolidated raw history files.
- **Context Required:** `qusa/utils/config.py`, `qusa/utils/config.yaml`, and current raw data directory naming pattern.
- **Implementation Directives:**
  - Add `GET /api/tickers`.
  - Read `raw_data_dir` from config.
  - Return sorted unique tickers from `*_history.csv` in `{"success": true, "tickers": [...]}`.
- **Acceptance Criteria:** Browser or curl request to `/api/tickers` returns the ticker list in the documented JSON envelope.

### Ticket 4

- **Ticket Name:** Add Feature Pipeline Endpoint
- **Objective:** Let the web app run feature generation for one ticker.
- **Context Required:** `web_app/api.py` and `qusa/services/pipeline_service.py`.
- **Implementation Directives:**
  - Add `POST /api/pipeline/run`.
  - Require a JSON object with a nonempty, filename-safe ticker string and optional boolean `fetch_latest`; reject invalid inputs with HTTP 400 before calling the service.
  - Return the service result or a clear JSON error with HTTP 400/500.
- **Acceptance Criteria:** Posting `{"ticker":"UPRO","fetch_latest":false}` returns a success JSON payload.

### Ticket 5

- **Ticket Name:** Add Prediction Endpoint
- **Objective:** Let the web app run latest prediction for one ticker.
- **Context Required:** `web_app/api.py` and `qusa/services/prediction_service.py`.
- **Implementation Directives:**
  - Add `POST /api/predictions/run`.
  - Validate `ticker` and `fetch_latest` as above; optional `volatility` must be null or a finite, non-negative JSON number.
  - Return the service JSON including nested prediction direction, confidence, probability, and ISO date; map invalid inputs, missing artifacts, and execution errors to JSON responses with HTTP 400, 404, and 500 respectively.
- **Acceptance Criteria:** Posting a valid ticker returns the same core fields currently written to the prediction log.

### Ticket 6

- **Ticket Name:** Add Prediction History Endpoint
- **Objective:** Expose logged predictions for display in the web UI.
- **Context Required:** `qusa/utils/config.yaml`, prediction CSV format from `scripts/model_prediction.py`.
- **Implementation Directives:**
  - Add `GET /api/predictions/history` reading only the configured `prediction.csv_log`; return `{"success": true, "history": []}` for missing or empty logs.
  - Support optional query param `ticker`.
  - Return latest rows first, capped at 50 after filtering, in a `history` JSON list; serialize missing numeric values as null.
- **Acceptance Criteria:** `/api/predictions/history?ticker=UPRO` returns JSON rows from the prediction log.

## Sprint 3: Frontend Web Interface

Goal: Build simple Jinja2 screens and static assets that replace CLI inputs with forms and visible results.

### Ticket 1

- **Ticket Name:** Create Base Layout Template
- **Objective:** Add a shared HTML shell for all web pages.
- **Context Required:** `web_app/templates/` directory and `web_app/static/css/`.
- **Implementation Directives:**
  - Create `web_app/templates/base.html`.
  - Include a header, main content block, and static CSS link.
  - Keep styling simple and local.
- **Acceptance Criteria:** A route rendering `base.html` loads without template errors.

### Ticket 2

- **Ticket Name:** Add Dashboard Page Route
- **Objective:** Serve a browser page at `/` instead of only JSON routes.
- **Context Required:** `web_app/__init__.py`, `web_app/templates/base.html`.
- **Implementation Directives:**
  - Create `web_app/routes.py` with a page blueprint.
  - Add `GET /` rendering `dashboard.html`.
  - Register page blueprint in `create_app()`.
- **Acceptance Criteria:** Visiting `/` displays a basic dashboard page.

### Ticket 3

- **Ticket Name:** Build Ticker Selection Form
- **Objective:** Let users select or type a ticker from the dashboard.
- **Context Required:** `web_app/templates/dashboard.html`, `/api/tickers` response shape.
- **Implementation Directives:**
  - Add a ticker input and optional datalist populated by server-side ticker discovery.
  - Keep the form independent from prediction execution.
  - Display an empty-state message if no tickers exist.
- **Acceptance Criteria:** Reloading `/` shows available tickers or a usable manual ticker input.

### Ticket 4

- **Ticket Name:** Add Run Feature Pipeline Button
- **Objective:** Trigger feature generation from the browser.
- **Context Required:** `web_app/templates/dashboard.html`, `web_app/static/js/`, `/api/pipeline/run`.
- **Implementation Directives:**
  - Add a button labeled "Run Feature Pipeline".
  - Use `fetch()` to POST ticker and `fetch_latest`.
  - Render success/error text in a status panel.
- **Acceptance Criteria:** Clicking the button runs the backend endpoint and displays output path or error.

### Ticket 5

- **Ticket Name:** Add Run Prediction Button
- **Objective:** Trigger latest prediction from the browser and display the result.
- **Context Required:** `web_app/templates/dashboard.html`, frontend JS file, `/api/predictions/run`.
- **Implementation Directives:**
  - Add a "Run Prediction" button.
  - Render direction, confidence, probability up, date, and volatility filter status.
  - Disable the button while the request is running.
- **Acceptance Criteria:** Clicking the button displays a structured prediction result without page reload.

### Ticket 6

- **Ticket Name:** Render Prediction History Table
- **Objective:** Show recent prediction log rows in the dashboard.
- **Context Required:** `/api/predictions/history` and `web_app/templates/dashboard.html`.
- **Implementation Directives:**
  - Add a table for latest predictions.
  - Load history on page load and after successful prediction.
  - Show timestamp, ticker, date, direction, probability, and confidence.
- **Acceptance Criteria:** The table updates after a new prediction is run.

## Sprint 4: Local Deployment & Documentation

Goal: Make the Flask app easy to install, run, test, and commit cleanly on GitHub.

### Ticket 1

- **Ticket Name:** Update Python Dependencies
- **Objective:** Ensure Flask runtime dependencies are captured.
- **Context Required:** Current `requirements.txt`.
- **Implementation Directives:**
  - Add `flask` if missing.
  - Keep existing quant/model dependencies intact.
  - Do not pin every package unless already pinned.
- **Acceptance Criteria:** `python -m pip install -r requirements.txt` installs Flask successfully.

### Ticket 2

- **Ticket Name:** Add Local Environment Example
- **Objective:** Document required environment variables without committing secrets.
- **Context Required:** Current `.env` usage in `qusa/utils/config.py` and README environment section.
- **Implementation Directives:**
  - Create `.env.example`.
  - Include `POLYGON_API_KEY=`, `QUSA_SMTP_USER=`, and `QUSA_SMTP_PASSWORD=`.
  - Do not copy real values from local `.env`.
- **Acceptance Criteria:** `.env.example` exists and contains only placeholder values.

### Ticket 3

- **Ticket Name:** Verify Git Ignore Rules
- **Objective:** Prevent local secrets, generated artifacts, and caches from being committed.
- **Context Required:** Current `.gitignore`, repository tree, generated data/log/model paths.
- **Implementation Directives:**
  - Ensure `.env`, `__pycache__/`, `.pytest_cache/`, logs, generated model files, and generated prediction CSVs are ignored.
  - Keep source files, docs, templates, and static assets trackable.
  - Do not ignore `requirements.txt` or README files.
- **Acceptance Criteria:** `git status --short` does not show secrets or generated runtime artifacts.

### Ticket 4

- **Ticket Name:** Add Flask Smoke Tests
- **Objective:** Confirm app factory and key endpoints work in CI/local tests.
- **Context Required:** `web_app/__init__.py`, `web_app/api.py`, existing `tests/` patterns.
- **Implementation Directives:**
  - Create `tests/test_web_app.py`.
  - Test `/health`, `/api/health`, and `/api/tickers`.
  - Use Flask test client; do not require network or Polygon API calls.
- **Acceptance Criteria:** `pytest -q` passes.

### Ticket 5

- **Ticket Name:** Rewrite README Run Instructions
- **Objective:** Add execution-ready instructions for CLI and Flask usage.
- **Context Required:** Current `README.md`, `requirements.txt`, `.env.example`.
- **Implementation Directives:**
  - Add setup steps: virtualenv, install requirements, copy `.env.example` to `.env`.
  - Add Flask command: `flask --app app run`.
  - Add common CLI commands that still work.
- **Acceptance Criteria:** A fresh developer can follow README steps to start the local app.

### Ticket 6

- **Ticket Name:** Define Clean Commit Milestones
- **Objective:** Create a clear GitHub-ready commit sequence for review.
- **Context Required:** Final `git status --short`.
- **Implementation Directives:**
  - Commit 1: app skeleton and service extraction.
  - Commit 2: API routes.
  - Commit 3: frontend templates/static assets.
  - Commit 4: docs, tests, and deployment cleanup.
- **Acceptance Criteria:** `git log --oneline` shows logical commits and `pytest -q` passes before final push.
