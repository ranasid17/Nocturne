# Legacy Surfaces

The supported runtime is `flask --app app run`, backed by `qusa.services` and the
local SQLite ledger.

`dashboard/app.py` and `flask_dashboard/app.py` are retained as deprecated
historical prototypes. They are not covered by the supported Flask API contract,
may not share the current SQLite run history, and must not be used for unattended
or production workflows. Removal needs explicit review approval after a parity
decision for performance, regimes, and email behavior.

`tools/build_pm_standalone_notebook.py` and
`notebooks/qusa_standalone_pm.ipynb` are also retained pending an approved decision
about portable notebook delivery. New research should import `qusa` directly; it
must not copy numerical or Monte Carlo implementations out of the package.
