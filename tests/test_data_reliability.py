from datetime import datetime

import pandas as pd
import pytest

from qusa.data.loader import DataLoader
from qusa.data.sessions import EASTERN, NyseSessionCalendar
from qusa.storage.locks import TickerBusyError, ticker_lock
from qusa.utils.settings import load_settings


def _bars(dates, close=100.0):
    return pd.DataFrame(
        {
            "date": dates,
            "open": [close] * len(dates),
            "high": [close + 1] * len(dates),
            "low": [close - 1] * len(dates),
            "close": [close] * len(dates),
            "volume": [1000] * len(dates),
        }
    )


@pytest.mark.parametrize(
    "now, expected",
    [
        (datetime(2026, 3, 9, 15, 0, tzinfo=EASTERN), "2026-03-06"),
        (datetime(2026, 3, 9, 17, 0, tzinfo=EASTERN), "2026-03-09"),
        (datetime(2026, 11, 26, 12, 0, tzinfo=EASTERN), "2026-11-25"),
        (datetime(2026, 11, 27, 14, 0, tzinfo=EASTERN), "2026-11-27"),
    ],
)
def test_nyse_sessions_handle_close_holiday_early_close_and_dst(now, expected):
    session = NyseSessionCalendar().most_recent_completed_session(now)
    assert session["date"] == expected


def test_nyse_readiness_marks_an_older_feature_bar_stale():
    readiness = NyseSessionCalendar().readiness_for_bar(
        "2026-03-06", datetime(2026, 3, 9, 17, 0, tzinfo=EASTERN)
    )

    assert readiness["status"] == "stale"
    assert readiness["expected_session"] == "2026-03-09"
    assert readiness["target_session"] == "2026-03-10"


def test_local_history_consolidation_never_requires_a_provider_key(monkeypatch, tmp_path):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    _bars(["2026-01-02"]).to_csv(tmp_path / "UPRO_history.csv", index=False)

    history, skipped = DataLoader(tmp_path).consolidate_history("UPRO")

    assert len(history) == 1
    assert skipped == []


def test_newer_fragment_revises_history_and_bad_input_is_not_archived(tmp_path):
    _bars(["2026-01-02"], close=100.0).to_csv(tmp_path / "UPRO_history.csv", index=False)
    _bars(["2026-01-02"], close=101.0).to_csv(tmp_path / "UPRO_revision.csv", index=False)
    pd.DataFrame({"date": ["bad-date"]}).to_csv(tmp_path / "UPRO_bad.csv", index=False)

    history, skipped = DataLoader(tmp_path).consolidate_history("UPRO")

    assert history.loc[0, "close"] == 101.0
    assert str(tmp_path / "UPRO_bad.csv") in skipped
    assert (tmp_path / "UPRO_bad.csv").exists()
    assert (tmp_path / "archive" / "UPRO_revision.csv").exists()


def test_failed_history_publication_preserves_the_existing_file(monkeypatch, tmp_path):
    original = _bars(["2026-01-02"], close=100.0)
    original.to_csv(tmp_path / "UPRO_history.csv", index=False)
    _bars(["2026-01-03"], close=101.0).to_csv(tmp_path / "UPRO_new.csv", index=False)

    monkeypatch.setattr("qusa.data.loader.atomic_write_csv", lambda *_args: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        DataLoader(tmp_path).consolidate_history("UPRO")

    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "UPRO_history.csv"), original)
    assert (tmp_path / "UPRO_new.csv").exists()


def test_ticker_lock_returns_a_defined_busy_error(tmp_path):
    with ticker_lock(tmp_path, "UPRO"):
        with pytest.raises(TickerBusyError, match="UPRO is busy"):
            with ticker_lock(tmp_path, "UPRO", timeout_seconds=0):
                pass


def test_settings_data_root_override_is_portable_and_does_not_rewrite_values(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "data:\n  paths:\n    raw_data_dir: raw\n"
        "model:\n  parameters:\n    class_weight: balanced\n"
    )

    config = load_settings(config_path, {"QUSA_DATA_ROOT": str(tmp_path / "data")})

    assert config["data"]["paths"]["raw_data_dir"] == str(tmp_path / "data" / "raw")
    assert config["model"]["parameters"]["class_weight"] == "balanced"
