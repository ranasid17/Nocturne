"""Atomic publication helpers for local CSV artifacts."""

import os
import tempfile
from pathlib import Path

import pandas as pd


def atomic_write_csv(data, destination):
    """Validate a CSV written beside its destination before replacing it."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        data.to_csv(temporary_path, index=False)
        published = pd.read_csv(temporary_path)
        if list(published.columns) != list(data.columns) or len(published) != len(data):
            raise ValueError("Temporary CSV validation failed.")
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return destination
