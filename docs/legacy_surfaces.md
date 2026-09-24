---
title: Legacy surfaces
nav_exclude: true
---

# Legacy Surfaces

The supported runtime is `flask --app app run`, backed by `qusa.services` and the
local SQLite ledger.

The former Streamlit dashboards, alternate Flask prototype, standalone notebook,
and notebook generator were removed after review approval in Sprint 6. Git history
retains those artifacts; they are not supported runtimes and must not be restored as
parallel application paths.

New research should import `qusa` directly. Numerical and Monte Carlo logic must
remain in the package rather than being copied into a portable notebook.
