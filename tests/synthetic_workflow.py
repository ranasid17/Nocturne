"""Credential-free fixtures shared by installed and source-tree verification."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from qusa.model.train import get_safe_features
from qusa.utils.settings import load_settings


def prepare_fixture(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = load_settings(environ={})
    for key in config["data"]["paths"]:
        path = root / key.removesuffix("_dir")
        path.mkdir(parents=True, exist_ok=True)
        config["data"]["paths"][key] = str(path)
    config["model"]["output"]["model_output_path"] = str(root / "models")
    config["model"]["parameters"].update(cv=2, min_samples_leaf=2, min_samples_split=4)
    config["monte_carlo"]["enabled"] = False
    config["storage"] = {"database_path": str(root / "history.sqlite3")}
    rng = np.random.default_rng(42)
    data = pd.DataFrame({name: rng.normal(size=160) for name in get_safe_features(False)})
    data["date"] = pd.bdate_range("2025-01-02", periods=160)
    data["close"] = 100 + rng.random(160)
    data["overnight_delta"] = rng.choice([-1., 1.], size=160)
    data["atr_pct"] = 1.5
    data["overnight_delta_pct"] = data["overnight_delta"]
    data["volume_ratio"] = rng.uniform(0.5, 2, 160)
    data["rsi"] = rng.uniform(15, 85, 160)
    for ticker in ("UPRO", "AAPL"):
        data.to_csv(Path(config["data"]["paths"]["processed_data_dir"]) / f"{ticker}_processed.csv", index=False)
    close = 100 + np.cumsum(rng.normal(0, 0.3, 400))
    raw = pd.DataFrame({
        "date": pd.bdate_range("2024-01-02", periods=400),
        "open": close - .1, "high": close + 1, "low": close - 1, "close": close,
        "volume": rng.integers(100000, 1000000, size=400),
    })
    for ticker in ("UPRO", "FEATURE"):
        raw.to_csv(Path(config["data"]["paths"]["raw_data_dir"]) / f"{ticker}_history.csv", index=False)
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config, config_path
