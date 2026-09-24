---
title: Overview
nav_order: 1
permalink: /
---

# Nocturne

Nocturne is a local research and prediction application for overnight US equity moves. The Flask dashboard generates features and predictions and reads saved history from SQLite. Explicit command-line workflows fetch data, train and evaluate models, run backtests, and cluster results. The app is for research, not financial advice.

<picture>
  <source media="(max-width: 40rem)" srcset="{{ '/assets/dashboard-mobile.png' | relative_url }}">
  <img src="{{ '/assets/dashboard-desktop.png' | relative_url }}" alt="Nocturne dashboard with synthetic UPRO data">
</picture>

The screenshot uses generated fixture data and a model trained for the integration test; it is not a market recommendation. [View the mobile layout]({{ '/assets/dashboard-mobile.png' | relative_url }}).

## Start here

- [Quickstart]({{ '/quickstart/' | relative_url }}) installs and launches the app.
- [Prediction semantics]({{ '/prediction-semantics/' | relative_url }}) explains what a result does and does not say.
- [Research workflows]({{ '/research/' | relative_url }}) covers the CLI, outputs, and optional reports.
- [API]({{ '/api/' | relative_url }}) lists the local HTTP contract.
- [Operations]({{ '/operations/' | relative_url }}) covers state and recovery.
- [Troubleshooting]({{ '/troubleshooting/' | relative_url }}) covers common failures.

The Flask server is intended for loopback use and has no authentication. GitHub Pages hosts **these static docs only**; it does not host the application. Watchlists, scheduled jobs, login, email controls, and network deployment are not part of the current MVP. The [archived roadmap]({{ '/archive/roadmap/' | relative_url }}) and [historical refactor plan]({{ '/refactor_execution_plan/' | relative_url }}) describe ideas and past work, not currently available features.
