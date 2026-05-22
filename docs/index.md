# Nocturne Small-Team Signal App Roadmap

> Draft for team review before implementation.

## Summary

Build this into a full small-team Flask app, but do it in two layers: keep the current Nocturne/QUSA repo as the research/model engine, publish it as an installable Python package, and create a new Flask app repo as the team-facing product. Use GitHub Pages only for docs, screenshots, setup guides, and roadmap.

Expected first MVP timeline: **4 weeks**. A more polished, production-ready team app is realistically **8-12 weeks**.

## Product Direction

- **Primary app:** Flask web app in a new repo.
- **Engine:** Current repo becomes the packaged Python engine for data fetch, feature generation, training, backtesting, prediction, reports, and notifications.
- **Hosting:** Private local/server VM first.
- **Storage:** SQLite for app metadata; local files for raw data, processed data, models, reports, charts, and logs.
- **Access control:** Per-user accounts with login, password hashing, and basic admin/user roles.
- **Scheduling:** APScheduler inside the Flask app.
- **Alerts:** Send an email after every scheduled prediction run.
- **Static site:** GitHub Pages docs only, not a live prediction app.

## 4-Week MVP Roadmap

### Week 1: Package The Engine

- Clean up current repo so it can be installed as a package by the Flask app.
- Add package metadata and stable import paths for:
  - data fetch
  - feature engineering
  - model training
  - prediction
  - backtest
  - email notification
- Move reusable workflow logic out of scripts where needed so Flask can call Python functions directly instead of shelling out.
- Keep existing CLI scripts working as wrappers around the package API.
- Add docs for installing the package from Git.

### Week 2: Create Flask App Repo

- New repo structure:
  - `app/` Flask application
  - `app/models/` SQLite models
  - `app/routes/` dashboard, auth, jobs, settings
  - `app/services/` QUSA engine integration
  - `app/templates/` server-rendered pages
  - `app/static/` CSS/JS/assets
  - `migrations/` database migrations
- Add per-user auth:
  - login/logout
  - password hashing
  - admin-created users
  - basic role field
- Add app settings screens for:
  - watched tickers
  - SMTP settings
  - recipient lists
  - schedule time
- Do not store plaintext SMTP passwords in SQLite unless encrypted or explicitly marked local-only; prefer environment variables or VM secrets for v1.

### Week 3: Prediction Jobs + Dashboard

- Add manual "Run Prediction" flow for one ticker or watchlist.
- Add scheduled daily prediction job using APScheduler.
- Store run metadata in SQLite:
  - ticker
  - run timestamp
  - status
  - prediction date
  - direction
  - probability up
  - confidence
  - notification status
  - artifact paths
- Build dashboard pages:
  - latest signals
  - prediction history
  - job history
  - per-ticker detail view
  - email delivery status
- Email every scheduled run summary to configured recipients.

### Week 4: Hardening + Docs

- Add tests for:
  - auth
  - scheduler job creation
  - prediction service success/failure
  - email send success/failure
  - SQLite persistence
- Add operational docs:
  - VM setup
  - environment variables
  - package install from Git
  - running Flask with Gunicorn
  - backup/restore for SQLite and artifact folders
- Add GitHub Pages docs site:
  - overview
  - screenshots
  - local setup
  - architecture
  - roadmap
- Run end-to-end acceptance test:
  - create user
  - configure ticker/watchlist
  - run manual prediction
  - receive email
  - wait for or trigger scheduled job
  - verify persisted history

## Feature Roadmap After MVP

- **Portfolio watchlist:** Multi-ticker signal board with filters for confidence, volatility, and latest run status.
- **Research workspace:** Compare training runs, backtests, model versions, feature sets, and performance metrics.
- **Notification controls:** Per-user recipient preferences, high-confidence-only mode, daily digest mode, failed-job alerts.
- **Model registry:** Track model versions, trained date, feature set, threshold, metrics, and artifact path.
- **Report center:** Browse generated LLM reports, backtest reports, charts, and prediction logs from the web UI.
- **Deployment upgrade:** Move from local VM + SQLite to Postgres + object storage if team usage grows.
- **Static docs site:** Keep GitHub Pages as the public/private documentation layer, not the operational app.

## Public Interfaces To Stabilize

- Current repo should expose stable Python functions for:
  - `run_feature_pipeline(ticker, fetch_latest=False)`
  - `run_model_pipeline(ticker, options)`
  - `make_latest_prediction(ticker, fetch_latest=True)`
  - `run_backtest(ticker, options)`
  - `send_prediction_email(config, recipients, prediction, ticker)`
- Flask app should never depend on parsing CLI text output.
- CLI scripts should remain supported for local research and automation.

## Assumptions

- This is for a small private team, not a public trading SaaS.
- Flask is the right full-app direction; Streamlit remains a prototype/reference until the Flask dashboard reaches parity.
- First deployment is a private VM or local server.
- SQLite plus local files is acceptable for v1.
- Scheduled alerts are the highest-priority new capability.
- GitHub Pages is for documentation only.
