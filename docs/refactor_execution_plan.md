# Nocturne MVP Remediation and Refactor Plan

Status: proposed; implementation has not started.
Date: 2026-09-21.

This plan addresses issues **identified**, not fixed, in the MVP review. It follows the completed Flask conversion. The sprint numbers below refer to this new remediation cycle.

## Decisions and Execution Rules

- Keep one repository, the `qusa` engine, and `web_app` as the canonical Flask application. Preserve existing CLI commands through thin wrappers.
- Use SQLite on local disk for run, prediction, model, and notification metadata. Keep datasets, model binaries, and reports as versioned files; CSV history becomes an export, not the authoritative run ledger.
- Execute tickets sequentially. Give the implementing model only the listed one or two files/snippets, the ticket, and the preceding ticket's public-contract handoff. New test files need no prior context. If more files are needed, split the work into another atomic ticket instead of enlarging the context indiscriminately.
- Each ticket leaves a short handoff: exported signature, input/output fields, errors, changed artifact semantics, and verification command. Interface changes require updating immediate callers before the sprint can pass.
- Tests must use temporary storage, synthetic data, and mocked provider/SMTP calls. Do not fetch paid data, email real recipients, retrain user models, or overwrite user artifacts as acceptance checks.
- Preserve local changes, including `qusa/utils/config.yaml`. Configuration migration must copy and validate settings, not replace user edits.
- Separate behavioral fixes from mechanical moves. Do not silently delete old models, reports, CSVs, or notebooks. Old results must be distinguishable as legacy/unverified.
- No PostgreSQL, second repository, SPA rewrite, broker execution, or distributed queue in these sprints. Network-shared deployment and intraday inference remain separate decisions.

## Required PR Gate After Every Sprint

1. Start `codex/remediation-sprint-N-<topic>` from freshly fetched, merged `main`. Do not reuse a merged sprint branch.
2. Complete that sprint's tickets and acceptance checks. Run the full Python suite, the existing coverage gate, `git diff --check`, and applicable browser/package/migration checks.
3. Commit only the sprint's changes. Open a PR to `main` with linked issue IDs, behavior changes, test evidence, compatibility notes, artifact/database migration instructions, and rollback steps.
4. Request the user's review and provide the PR URL. Do not auto-merge. Address review/CI failures on that branch, and begin the next dependent sprint only after merge is confirmed.

A failed acceptance check is not a completed ticket. If a sprint becomes too large to review coherently, split it into numbered sub-sprints with the same PR gate. The headings are dependency milestones, not promises of equal calendar duration.

## Sprint 1: Correct Returns, Targets, and Public Errors

Goal: Remove known incorrect performance calculations and unsafe error exposure before expanding the product.

### R1.1 Sanitize Public API Errors

- **Objective:** Prevent credentials and raw upstream URLs from appearing in browser errors.
- **Context Required:** `web_app/api.py`; `qusa/data/fetcher.py` request/error-handling snippets.
- **Implementation Directives:** Introduce stable error codes and safe public messages while preserving documented status codes; redact credential-bearing query parameters in diagnostics; add `tests/test_api_errors.py` using a sentinel key, never a real secret.
- **Acceptance Criteria:** Simulated HTTP/provider errors never expose the sentinel in JSON or captured logs; missing-artifact and validation errors remain useful; existing API tests pass.

### R1.2 Fix Return Units and Drawdown

- **Objective:** Calculate dimensionless close-to-next-open returns and include initial capital in the drawdown baseline.
- **Context Required:** `qusa/model/backtest.py`; `qusa/features/overnight.py` definitions of the two overnight columns.
- **Implementation Directives:** Use the next dollar gap divided by the current close, or the equivalent validated next-row percentage; retain explicit per-side cost and position-size semantics; add focused numerical fixtures in `tests/test_backtest_returns.py`.
- **Acceptance Criteria:** $200 to $204 produces 2%, and proportionally scaled prices produce the same return; long/short and two-sided costs are correct; a losing first trade creates drawdown; missing next-session outcomes and invalid prices do not become trades.

### R1.3 Define the Shared Supervised Dataset

- **Objective:** Create one feature-date/outcome-date alignment contract.
- **Context Required:** `qusa/model/train.py` data/preprocessing helpers; `qusa/features/overnight.py` output columns.
- **Implementation Directives:** Add `qusa/model/dataset.py` returning aligned features, labels, feature dates, and outcome dates; normalize or reject unsorted/duplicate dates explicitly and drop unknown future outcomes before classification; preserve zero-gap classification explicitly and add `tests/test_model_dataset.py`.
- **Acceptance Criteria:** An alternating-gap fixture distinguishes current-day from next-day labels; a missing next gap is not labeled negative; final unknown outcomes are excluded; invalid chronology has deterministic handling.

### R1.4 Integrate Training and Save Its Temporal Boundary

- **Objective:** Train using the shared dataset and persist enough metadata to establish out-of-sample boundaries.
- **Context Required:** `qusa/model/train.py`; `qusa/model/dataset.py` from R1.3.
- **Implementation Directives:** Replace duplicate target/preprocessing logic with the shared builder; record model ID, schema/target version, training feature interval, latest training outcome availability, and holdout boundary; reject unsupported one-class training and insufficient split sizes before model publication.
- **Acceptance Criteria:** A synthetic training bundle records exact boundaries; labels unavailable at a validation cutoff cannot enter training; invalid training sets fail clearly; legacy bundles remain identifiable rather than receiving invented metadata.

### R1.5 Align Standalone Evaluation

- **Objective:** Score the same target used in training, using a verified held-out interval.
- **Context Required:** `qusa/model/evaluate.py`; `qusa/model/dataset.py` plus R1.4's bundle-field handoff.
- **Implementation Directives:** Use the shared target/preprocessing contract; restrict normal evaluation to eligible rows beyond training outcome availability and reject unverifiable legacy boundaries; support one-class evaluation through fixed label ordering and include zero-probability observations in calibration bins.
- **Acceptance Criteria:** Current-day labels cannot accidentally replace next-day labels; overlap is rejected or explicitly filtered with counts reported; one-class held-out samples do not crash; calibration includes predictions equal to zero.

### R1.6 Enforce Backtest Boundaries in the CLI Workflow

- **Objective:** Stop presenting whole-history, partly in-sample backtests as held-out performance.
- **Context Required:** `qusa/model/backtest.py`; `scripts/run_model_pipeline.py` evaluation/backtest call sites.
- **Implementation Directives:** Apply the saved temporal boundary before scoring; allow any in-sample diagnostic only as an explicit, prominently labeled mode; version new result metadata and prevent silent replacement of legacy performance artifacts.
- **Acceptance Criteria:** The documented model CLI evaluates and backtests a known synthetic holdout only; overlap and unknown model versions cannot yield ordinary success metrics; generated reports identify evaluation mode and model ID.

**Sprint PR:** `Fix return units, target alignment, and API error disclosure`. Include numerical before/after examples and explain why older reports are unverified. Link follow-ups to issues #18 and #66. Open the PR and stop for review/merge.

## Sprint 2: Make Features and Inference Consistent

Goal: Establish consistent simulation units, feature selection, configuration propagation, and inference validation.

### R2.1 Correct Monte Carlo Mathematics

- **Objective:** Give simulation horizons a consistent daily-unit interpretation.
- **Context Required:** `qusa/features/monte_carlo.py`; R1.4 model-version handoff.
- **Implementation Directives:** Document daily log-return mean/variance and simulate horizon h consistently, avoiding an extra drift adjustment if the estimated mean is already a log-return mean; use a local generator instead of global NumPy seeding; add analytical-distribution checks in `tests/test_monte_carlo_distribution.py` and increment feature semantics version.
- **Acceptance Criteria:** Simulated log-return mean and variance agree with h-day analytical expectations within justified tolerances; quantiles are ordered; zero volatility is deterministic; unrelated random streams are unchanged.

### R2.2 Make Recalculation Deterministic

- **Objective:** Prevent populated MC columns or batch layout from silently changing feature validity.
- **Context Required:** `qusa/features/monte_carlo.py`; `tests/test_monte_carlo_distribution.py`.
- **Implementation Directives:** Prefer full deterministic recomputation until a verified cache exists; if reuse is retained, require input/config/version fingerprints and stable row-specific randomness; make validation cover configured horizons rather than only `mc_1d_*`.
- **Acceptance Criteria:** Identical inputs reproduce features across batch sizes; appending rows preserves prior results; revised bars/settings invalidate affected outputs; 3/7-day-only configurations validate without invented one-day columns.

### R2.3 Derive the Active Feature Manifest

- **Objective:** Ensure the model requires only features actually generated by enabled components.
- **Context Required:** `qusa/model/train.py`; `qusa/features/pipeline.py` feature-name helper.
- **Implementation Directives:** Separate non-MC base features from configured horizon features; use one manifest for training and pipeline contracts; record its ordered names and semantic version in model metadata.
- **Acceptance Criteria:** MC disabled produces no required `mc_*` features; horizons `[3, 7]` do not require `mc_1d_*`; duplicate names are impossible; the generated manifest can be selected from a matching feature output.

### R2.4 Forward Custom Feature Settings

- **Objective:** Make user-specified volatility windows reach the calculator.
- **Context Required:** `qusa/services/pipeline_service.py`; `qusa/features/pipeline.py` constructor.
- **Implementation Directives:** Standardize the feature configuration key used by technical and volatility components; preserve a narrow compatibility path where necessary; add a service-construction test with non-default windows.
- **Acceptance Criteria:** VWAP, short/long regime, and advanced volatility windows receive the supplied values; existing defaults still work; no silent fallback caused by a mismatched key remains.

### R2.5 Forward Training Options

- **Objective:** Make advertised tuning options reach the existing training implementation.
- **Context Required:** `scripts/run_model_pipeline.py` `_build_model_config`; `qusa/model/train.py` tuning configuration contract.
- **Implementation Directives:** Forward validated tuning and horizon settings through the projection; reject malformed settings instead of ignoring them; test construction/delegation without running a costly search.
- **Acceptance Criteria:** Enabled tuning reaches the estimator with the supplied grid; disabled tuning preserves the normal path; invalid grids return actionable errors.

### R2.6 Unify Prediction Validation and Volatility Status

- **Objective:** Make inference enforce the model's feature contract and distinguish unavailable risk checks from passed checks.
- **Context Required:** `qusa/model/predict.py`; `qusa/model/dataset.py` preprocessing contract.
- **Implementation Directives:** Reuse preprocessing and validate class/schema compatibility; return an explicit volatility state of disabled/pass/blocked/unavailable plus actual ATR and applied threshold; retain legacy fields temporarily but never encode unavailable as an ordinary pass.
- **Acceptance Criteria:** Infinite/missing features follow the documented policy; missing/NaN ATR is unavailable; threshold breach is blocked; incompatible models fail clearly; valid existing response fields remain available for the web adapter.

**Sprint PR:** `Correct Monte Carlo units and unify feature contracts`. Include disabled-MC, alternate-horizon, and non-finite-input evidence. Link issue #82 as a blocked follow-on experiment, not a completed experiment. Open the PR and stop for review/merge.

## Sprint 3: Establish Reliable Configuration and Market Data

Goal: Make local data access portable and protect canonical market history before adding unattended writes.

### R3.1 Introduce Validated Local Settings

- **Objective:** Replace machine-specific implicit paths with one explicit configuration/data-root contract.
- **Context Required:** `qusa/utils/config.py`; committed `qusa/utils/config.yaml` settings schema, excluding secrets/local edits.
- **Implementation Directives:** Add a canonical settings module with defaults plus optional local overrides, resolving paths against one documented root; retain `load_config` as a compatibility wrapper and keep environment loading at application/CLI bootstrap boundaries through the declared dotenv library; provide migration guidance that preserves existing overrides.
- **Acceptance Criteria:** Temporary checkouts resolve their own paths from different working directories; exported environment values win; invalid settings fail early; existing user configuration is never rewritten automatically.

### R3.2 Separate File Access from Provider Initialization

- **Objective:** Allow feature generation from local history without an API key.
- **Context Required:** `qusa/data/loader.py`; `qusa/data/fetcher.py` constructor.
- **Implementation Directives:** Initialize/inject the provider only for network operations; validate required OHLCV columns, dates, and finite/positive prices before publishing history; make `load_range` create its destination directory.
- **Acceptance Criteria:** Local consolidation works with no key; actual fetch attempts fail clearly without credentials; a first historical-range fetch succeeds in an empty temporary directory using a mock provider; malformed bars cannot be silently accepted.

### R3.3 Define Trading-Session and Availability Semantics

- **Objective:** Replace yesterday/weekend arithmetic with a tested exchange-session policy.
- **Context Required:** `qusa/data/fetcher.py`; new `qusa/data/sessions.py` contract.
- **Implementation Directives:** Use a maintained exchange-calendar dependency after checking its current official documentation; inject the clock, handle Eastern timezone/DST/holidays/early closes, and distinguish completed session from available provider data; verify returned dates and represent unavailable data explicitly rather than silently relabeling older bars.
- **Acceptance Criteria:** Frozen-time tests cover before/after close, weekend, holiday, early close, DST, and delayed-provider responses; no live API key is needed. Dependency updates are a separate mechanical subtask after selecting the library.

### R3.4 Preserve Revisions and Publish History Atomically

- **Objective:** Ensure a validated newer bar replaces an older revision without losing canonical history on failure.
- **Context Required:** `qusa/data/loader.py`; new `qusa/storage/artifacts.py` file-publication helper.
- **Implementation Directives:** Define deterministic source precedence with provenance instead of trusting glob order; write and validate a temporary file on the destination filesystem before atomic replacement; archive only successfully ingested inputs and retain/quarantine unreadable inputs.
- **Acceptance Criteria:** A corrected close replaces an old close; a malformed fragment remains recoverable; injected write failure leaves original history readable; repeated consolidation is stable.

### R3.5 Serialize Writers Across CLI and Flask

- **Objective:** Prevent concurrent workflows from racing on shared per-ticker artifacts.
- **Context Required:** `qusa/services/pipeline_service.py`; `qusa/services/prediction_service.py`.
- **Implementation Directives:** Introduce one cross-process, per-ticker coordination boundary used by both services, with bounded waiting and clear busy errors; avoid reentrant deadlock when prediction invokes feature generation; use atomic publication for processed files and preflight model availability before an unnecessary network fetch.
- **Acceptance Criteria:** Two processes for the same ticker serialize or return a defined busy response; different tickers can proceed; failures release ownership; a prediction that refreshes features does not deadlock. SQLite run ownership can replace this mechanism in Sprint 4 without changing the service contract.

### R3.6 Add Prediction Readiness Metadata

- **Objective:** Make stale inputs and unknown forecast timing visible in every prediction result.
- **Context Required:** `qusa/services/prediction_service.py`; `qusa/data/sessions.py` from R3.3.
- **Implementation Directives:** Return feature-as-of, observed availability time, target session, generated time, model ID, and readiness status; keep descriptive direction separate from eligibility and represent legacy timing as unknown; enforce a documented policy for stale/manual research runs versus normal operational runs.
- **Acceptance Criteria:** A stale or unavailable bar cannot produce an ordinary ready result; dates are timezone-aware where timestamps are required; manual historical use is labeled; provider-after-midnight availability does not imply a pre-close executable signal.

**Sprint PR:** `Make market data portable, revision-aware, and concurrency-safe`. Include calendar fixtures and cross-process/write-failure evidence. Link #70 and #73; do not enable a schedule. Open the PR and stop for review/merge.

## Sprint 4: Introduce SQLite and Durable Run History

Goal: Replace CSV-based run tracking with transactional metadata while retaining user artifacts and export compatibility.

### R4.1 Add the Database and Versioned Schema

- **Objective:** Create an explicitly configured SQLite database on local disk.
- **Context Required:** canonical settings module from R3.1; new `qusa/storage/database.py`.
- **Implementation Directives:** Define migration-managed tables for runs, predictions, model metadata, artifact references, and notification attempts; enable foreign keys, WAL where supported, and bounded busy handling per connection; keep database transactions short and never hold them during provider/model work.
- **Acceptance Criteria:** Empty database initialization and repeat initialization are safe; migrations preserve fixtures; foreign keys are enforced; schema version is observable; failures roll back. Document one chosen migration approach rather than adding competing migration systems.

### R4.2 Implement Run Transitions and Idempotency

- **Objective:** Give each operation an auditable lifecycle independent of a browser session.
- **Context Required:** `qusa/storage/database.py`; new `qusa/storage/runs.py`.
- **Implementation Directives:** Support queued/running/succeeded/failed/interrupted states with validated transitions and ownership; enforce uniqueness for a logical retry key while allowing explicitly separate manual runs; recover abandoned runs without silently repeating external effects.
- **Acceptance Criteria:** Concurrent submissions with the same retry key create one logical run; separate manual requests may create distinct runs; illegal transitions fail; restart recovery is deterministic; timestamps and sanitized errors persist.

### R4.3 Persist Service Results and Artifact Identity

- **Objective:** Make successful predictions and failures survive restarts, including their input/model lineage.
- **Context Required:** `qusa/services/prediction_service.py`; `qusa/storage/runs.py` public API.
- **Implementation Directives:** Record run start/end and prediction metadata through the repository; publish versioned artifacts before committing successful references and handle failed publication explicitly; preserve existing response fields while adding run/model identity.
- **Acceptance Criteria:** Service success creates one linked prediction; service failure records a failed run with no successful prediction; invalid artifact publication cannot leave a success record; restart reads the same result. Adapt pipeline-only run tracking as a separate two-file subtask using the same contract.

### R4.4 Import Legacy CSV History and Support Export

- **Objective:** Preserve existing records without inventing provenance or multiplying them on repeated imports.
- **Context Required:** `qusa/storage/runs.py`; existing CSV field contract from `qusa/services/prediction_service.py`.
- **Implementation Directives:** Add an explicit dry-run/import command with source fingerprint and row ordinal identity; mark unknown model/timezone/schema values as legacy/unknown and retain source rows; provide CSV export without keeping dual authoritative write paths.
- **Acceptance Criteria:** Reimporting the same source is idempotent, distinct repeated rows are not accidentally collapsed, malformed rows are reported, and original CSVs are unchanged; exported rows have a documented stable schema.

### R4.5 Read History Through the Repository

- **Objective:** Move history/latest-result/run queries out of CSV parsing in HTTP routes.
- **Context Required:** `web_app/api.py`; `qusa/storage/runs.py` query contract.
- **Implementation Directives:** Preserve the history envelope, ticker filtering, latest-first order, and 50-row default; add bounded pagination and read-only latest-result/run-detail endpoints; return model/readiness/volatility metadata needed by the UI.
- **Acceptance Criteria:** API contract tests use temporary SQLite; ordering/filtering/pagination are stable; reads never run predictions or send notifications; migrated legacy records display honestly.

### R4.6 Define Notification Delivery and Recovery

- **Objective:** Prevent email-on-status-poll and make delivery failures recoverable before future scheduling.
- **Context Required:** `qusa/notifications/email.py`; `qusa/storage/runs.py` or the notification-table contract from R4.1.
- **Implementation Directives:** Persist an outbox/delivery attempt associated with a run and recipient set; send only from an explicit worker action, never GET routes; define pending/sent/failed/unknown outcomes and the crash-after-send policy, recognizing SMTP cannot guarantee exactly-once delivery.
- **Acceptance Criteria:** Polling never sends mail; concurrent workers cannot claim the same active attempt; retries of known failures are bounded; ambiguous post-send crashes remain visible rather than being blindly resent. Use SMTP mocks only.

**Sprint PR:** `Add SQLite run tracking and legacy-history migration`. Include migration dry-run output, restart/concurrency tests, export examples, and a rollback procedure that preserves the database and source CSVs. Open the PR and stop for review/merge.

## Sprint 5: Consolidate Services and Improve the Canonical UI

Goal: Make the supported application use one engine and show the operational context necessary to interpret results.

### R5.1 Package the Existing Engine

- **Objective:** Install the engine and web entry point without repository-path hacks.
- **Context Required:** `requirements.txt`; new `pyproject.toml` plus current package names as a short tree listing.
- **Implementation Directives:** Package the current layout and templates/static assets without an unsolicited `src/` migration; declare supported Python versions and justified optional dependencies; maintain a tested local-install requirements path and verify a built wheel from outside the checkout.
- **Acceptance Criteria:** An isolated wheel installation imports the services and serves templates/assets; CLI wrappers still run; web-only imports do not require optional research/legacy dependencies after subsequent extraction.

### R5.2 Extract the Model Workflow

- **Objective:** Move training/evaluation/backtest orchestration behind a stable service.
- **Context Required:** `scripts/run_model_pipeline.py`; new `qusa/services/research_service.py`.
- **Implementation Directives:** Move workflow logic without changing the corrected numerical contracts; keep argparse/ticker iteration/exit handling in the CLI; return structured per-phase results and use the Sprint 4 run/artifact APIs.
- **Acceptance Criteria:** CLI and direct service calls agree on synthetic inputs; missing-data/phase failures have consistent results and exit codes; no subprocess is needed to invoke the workflow from Python.

### R5.3 Extract Clustering and Scope Its Outputs

- **Objective:** Preserve research functionality without CLI-bound plotting or cross-ticker overwrites.
- **Context Required:** `scripts/run_clustering.py`; new `qusa/services/clustering_service.py`.
- **Implementation Directives:** Extract orchestration first, then move plotting helpers to `qusa/reporting/plots.py` in a separate two-file subtask; key regime statistics and figures by ticker/run instead of one shared filename; retain a thin CLI wrapper and explicit legacy artifact lookup policy.
- **Acceptance Criteria:** Two ticker runs retain separate statistics/plots; service and CLI agree; readers cannot show another ticker's regime file; no unlabeled shared output is overwritten.

### R5.4 Separate Optional Reporting Imports

- **Objective:** Keep LLM and plotting dependencies outside ordinary prediction startup.
- **Context Required:** `qusa/model/reporter.py` and `qusa/model/reports.py`; inspect/update their public exports as a subsequent two-file subtask.
- **Implementation Directives:** Move implementation into `qusa/reporting/` with temporary compatibility exports where needed; remove eager reporting imports from core prediction paths; return explicit report failure metadata instead of writing an error string as a successful report.
- **Acceptance Criteria:** Core prediction imports work without optional reporting packages; report-disabled workflows do not contact Ollama; mocked report failures are visible and do not invalidate an otherwise completed model run.

### R5.5 Present Readiness and the Saved Latest Result

- **Objective:** Show model/data context and correct risk status when opening the dashboard.
- **Context Required:** `web_app/templates/dashboard.html`; `web_app/static/js/dashboard.js`, with R4.5 endpoint handoff.
- **Implementation Directives:** Load saved latest results and distinguish them from current-session requests; render model ID/cutoff, feature date, target session, actual ATR/applied limit, and readiness; use the explicit volatility state and label the estimate as model probability rather than implying calibration.
- **Acceptance Criteria:** Refresh preserves the latest saved result; stale, unavailable, blocked, disabled, and legacy states have distinct displays; ticker-switch race tests pass; no unavailable state is rendered as within-limit.

### R5.6 Separate Routine Prediction from Research Controls

- **Objective:** Make prerequisites and failed runs discoverable without expanding into an all-purpose dashboard.
- **Context Required:** dashboard template; dashboard JavaScript.
- **Implementation Directives:** Place manual feature generation in a Data/Research section and keep readiness-based preparation accessible; add bounded run history/details using R4.5; preserve labeled controls, loading/error feedback, keyboard operation, and narrow-screen table scrolling.
- **Acceptance Criteria:** Fresh and prepared checkouts both have actionable states; failures show a run ID and safe explanation; desktop/mobile layouts do not overlap; no model fetch/training happens merely by opening or refreshing the page.

**Sprint PR:** `Consolidate workflows and surface model readiness in Flask`. Include packaging checks, CLI/service parity evidence, and desktop/mobile screenshots. Record explicitly which legacy capabilities are supported, deferred, or rejected. Open the PR and stop for review/merge.

## Sprint 6: Retire Duplicates and Reconcile the Release

Goal: Finish the refactor with one supported runtime, trustworthy documentation, reproducible tests, and an actionable backlog.

### R6.1 Decide and Retire Legacy Dashboards

- **Objective:** Remove competing runtimes only after their remaining users and capabilities are accounted for.
- **Context Required:** `dashboard/app.py` and `flask_dashboard/app.py`; Sprint 5 parity table as the handoff.
- **Implementation Directives:** Inventory signals/performance/regimes/email requirements and record each as migrated, explicitly deferred, or intentionally unsupported; retire legacy entry points/assets in a separate mechanical commit only after that decision is approved in review; remove unused dependencies and the obsolete Jinja monkeypatch through small follow-up edits after import/usage checks.
- **Acceptance Criteria:** The PR makes every removed capability explicit; the canonical app has no imports or documentation pointing to deleted modules; the Git history retains legacy implementations and no generated user data is deleted. If parity is undecided, deprecate rather than delete and report the remaining gate.

### R6.2 Remove Accidental Notebook Engine Duplication

- **Objective:** Keep one maintained source for numerical logic.
- **Context Required:** `tools/build_pm_standalone_notebook.py`; `notebooks/qusa_standalone_pm.ipynb` code-cell extracts.
- **Implementation Directives:** Replace duplication with a small package-importing example under `examples/`; if standalone portable delivery is explicitly still required, retain it with an owner and contract-equivalence tests instead; remove the generator only after the replacement/retention decision is approved.
- **Acceptance Criteria:** A clean notebook execution uses the corrected engine and synthetic fixtures; no example silently retains old return/MC formulas; documented installation and artifact locations work.

### R6.3 Make Browser and Service Verification Reproducible

- **Objective:** Make CI detect contract regressions that the earlier smoke suite missed.
- **Context Required:** `tests/web_dashboard.cjs`; `.github/workflows/ci.yml`.
- **Implementation Directives:** Move browser tests under `tests/browser/` with pinned test-tool dependencies and temporary fixture setup; exercise the actual Flask/SQLite integration while mocking only external provider/SMTP boundaries; organize numerical/temporal tests by domain and measure core plus web coverage without weakening the existing gate.
- **Acceptance Criteria:** A fresh CI checkout with no user data/models passes browser and service tests; concurrency, legacy import, prediction failure, and readiness cases run deterministically; screenshots/failure logs are available as CI artifacts.

### R6.4 Publish Current Documentation

- **Objective:** Replace the stale two-repository roadmap with documentation of the delivered architecture.
- **Context Required:** `docs/index.md`; `docs/_config.yml`. Update README and GEMINI pointers in a separate two-file subtask.
- **Implementation Directives:** Archive the old roadmap and completed conversion plan with working links; publish overview, setup, data/target contract, architecture, and operations/backup pages in bounded one/two-file tasks; remove duplicated H1, update supported versions/commands, and distinguish current functionality from future work.
- **Acceptance Criteria:** Local docs build and link checks pass; after merge, published Pages shows the current single-repository SQLite architecture; screenshots and setup are reachable; no page claims unimplemented accounts or scheduling exist.

### R6.5 Verify Backup, Restore, and Release Migration

- **Objective:** Make SQLite and versioned artifacts recoverable using documented procedures.
- **Context Required:** `qusa/storage/database.py`; new `docs/operations.md`.
- **Implementation Directives:** Use SQLite's supported backup mechanism rather than copying only the main file during WAL activity; describe a consistent database/artifact snapshot and restore path; rehearse upgrade/import/rollback on a temporary legacy fixture and preserve schema/version evidence.
- **Acceptance Criteria:** A restored application resolves all referenced artifacts and reads the same runs/predictions; backups during normal use are consistent; rollback does not discard newly created records or source CSVs.

### R6.6 Reconcile GitHub Issues and the Project

- **Objective:** Make tracking reflect completed remediation and unresolved product choices.
- **Context Required:** latest bodies/statuses of #81, #73, #60, #82 and PR #65; the approved parity/release handoff.
- **Implementation Directives:** Use one issue per deliverable, linked PRs, clear acceptance criteria, and Correctness/Unified App/Automation milestones; close #81 only after its remaining scope is explicitly split or accepted, keep #73 gated on timing/durability, and propose deferral/supersession of #60/#65 rather than merging intraday behavior; retain #82 as an experiment comparing corrected baselines, not a promise to ship more features.
- **Acceptance Criteria:** Project status is consistent with merged work; obsolete In Progress cards have an explained disposition; the user approves scope-changing closures/deferrals as part of the sprint review. Missing project permissions are reported, never treated as a successful update.

**Sprint PR:** `Retire duplicate runtimes and publish the remediated MVP`. Include the deletion/parity inventory, tested migration/restore procedure, CI evidence, and proposed GitHub dispositions. Open the PR and stop for review/merge; verify Pages and perform approved tracking updates after merge.

## Explicitly Gated Follow-On Work

These are not prerequisites for declaring the six-sprint remediation complete and must not be silently added to its scope.

| Work | Gate before implementation | PR requirement |
| --- | --- | --- |
| Daily automation (#73) | Agree on provider entitlement/availability, forecast target, eligible run window, retries, stale-data policy, and notification behavior; Sprints 1-4 merged | A separate sprint/PR for a single scheduler owner plus restart, holiday, DST, and duplicate-trigger tests; never create one scheduler per Flask worker |
| Intraday prediction (#60/#65) | A separate as-of dataset, compatible target/entry-time definition, accessible data source, and out-of-sample evaluation | New research PR; do not reuse the old PR as a shortcut around current service contracts |
| MC horizon research (#82) | Corrected MC/config/holdout contracts from Sprints 1-2 | Experiment PR comparing no-MC, 1-day, 3-day, and 7-day settings using held-out metrics; q50 is median, not generally expected value |
| Watchlists and read-only performance/regime views | Reliable single-ticker run/model identity and explicit parity priority | Small feature PRs; do not revive duplicate runtime implementations |
| Team/network deployment | Confirm actual need, auth/authorization, appropriate CSRF/origin protections, production server, and backup ownership | Separate deployment sprint/PR; keep the present application loopback-only until approved |
| PostgreSQL | Multiple app/worker hosts or measured write contention requiring it | Dedicated schema/data migration PR with tests against PostgreSQL; SQLite is the current choice |

## Completion Checklist

- All ten review findings map to merged fixes or an explicit, documented scope decision.
- Return/target/MC behavior has numerical regression evidence, not only passing shape assertions.
- Existing CLI commands and Flask share corrected services and versioned model/feature contracts.
- SQLite owns run/prediction metadata; imports, idempotency, concurrency, and recovery are tested.
- One runtime is supported; any retained legacy/portable surface has an explicit owner and scope.
- Public docs, README, contributor guidance, GitHub issues, and the project agree about current capabilities.
- Each sprint has its own review PR. No user artifacts or unrelated local edits enter those PRs.
