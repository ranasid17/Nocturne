"""Shared temporal dataset contract for overnight-direction models."""

import pandas as pd


TARGET_VERSION = "next_session_overnight_direction_v1"
FEATURE_DATE_COLUMN = "feature_date"
OUTCOME_DATE_COLUMN = "outcome_date"
OUTCOME_DELTA_COLUMN = "outcome_overnight_delta"


class DatasetContractError(ValueError):
    """Raised when input data cannot support a temporally valid dataset."""


def build_supervised_dataset(data):
    """Label each feature row with the following session's overnight move.

    Rows without a known following-session outcome are deliberately excluded. The
    function rejects ambiguous ordering instead of silently changing the meaning
    of a persisted feature dataset.
    """

    required_columns = {"date", "overnight_delta"}
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise DatasetContractError(
            "Dataset is missing required columns: " + ", ".join(missing_columns)
        )

    dataset = data.copy()
    dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
    if dataset["date"].isna().any():
        raise DatasetContractError("Dataset contains an invalid date.")
    if dataset["date"].duplicated().any():
        raise DatasetContractError("Dataset contains duplicate dates.")
    if not dataset["date"].is_monotonic_increasing:
        raise DatasetContractError("Dataset dates must be sorted in ascending order.")

    dataset[FEATURE_DATE_COLUMN] = dataset["date"]
    dataset[OUTCOME_DATE_COLUMN] = dataset["date"].shift(-1)
    dataset[OUTCOME_DELTA_COLUMN] = pd.to_numeric(
        dataset["overnight_delta"].shift(-1), errors="coerce"
    )
    dataset["overnight_delta"] = pd.to_numeric(
        dataset["overnight_delta"], errors="coerce"
    )
    dataset = dataset.dropna(
        subset=["overnight_delta", OUTCOME_DATE_COLUMN, OUTCOME_DELTA_COLUMN]
    ).copy()
    dataset["target"] = (dataset[OUTCOME_DELTA_COLUMN] > 0).astype(int)

    if dataset.empty:
        raise DatasetContractError("Dataset has no rows with known next-session outcomes.")

    return dataset
