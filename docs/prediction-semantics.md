---
title: Prediction Semantics
nav_order: 3
---

# Prediction semantics

A prediction uses the latest row in the processed feature CSV and a saved model bundle. Its `date` is the **feature session**, not the date when the next market move has been observed. `readiness.target_session` is the next NYSE session after that feature session, when the calendar can determine it. A stale feature row still has a target session; that does not make the signal current.

`probability_up` is the model's estimated probability for class `1` on that row. `direction` comes from the model's predicted class, and `confidence` is `HIGH` when `probability_up` is at least the saved threshold or at most one minus it; otherwise it is `LOW`. These are model outputs, not calibrated odds, investment advice, or verified trading performance.

## Volatility state

The ATR filter is separate from the model direction and confidence. `disabled` means it was not enabled; `unavailable` means the requested check lacked a valid non-negative `atr_pct`; `pass` means ATR did not exceed the configured threshold; `blocked` means it did. `volatility_filter_triggered` only signals the blocked condition, so inspect `volatility_state` as well. The API's optional `volatility` number is a non-negative percent threshold override for that request.

## Readiness and freshness

`readiness` is saved at **generation time** using the most recently completed NYSE session then. `ready` means the feature row matched that completed session; `stale` means it did not; `unavailable` means the comparison could not be made. The API's `freshness` field on a new or latest result recalculates against the **current** completed session. A saved result can therefore have `readiness.status: ready` and `freshness.status: stale` later. The dashboard displays both concepts separately. Neither status verifies that the underlying provider data was available when a historical backtest made a decision.

The history endpoint stores the feature date and generation-time readiness. The latest endpoint adds current freshness; treat saved history as a record of what was generated, not a live quote. See the [API reference]({{ '/api/' | relative_url }}) for endpoint behavior.
