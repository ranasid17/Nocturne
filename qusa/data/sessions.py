"""NYSE session and bar-availability policy."""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd


EASTERN = ZoneInfo("America/New_York")


class SessionCalendarUnavailableError(RuntimeError):
    """Raised when the configured exchange calendar dependency is unavailable."""


class NyseSessionCalendar:
    """Resolve completed NYSE sessions using exchange-provided close times."""

    def __init__(self, calendar=None, clock=None):
        self._calendar = calendar
        self._clock = clock or (lambda: datetime.now(EASTERN))

    @property
    def calendar(self):
        if self._calendar is None:
            try:
                import pandas_market_calendars as market_calendars
            except ImportError as exc:
                raise SessionCalendarUnavailableError(
                    "pandas_market_calendars is required for NYSE session checks."
                ) from exc
            self._calendar = market_calendars.get_calendar("NYSE")
        return self._calendar

    def _now(self, now=None):
        timestamp = pd.Timestamp(now or self._clock())
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(EASTERN)
        return timestamp.tz_convert(EASTERN)

    def schedule(self, start, end):
        return self.calendar.schedule(
            start_date=pd.Timestamp(start).date(),
            end_date=pd.Timestamp(end).date(),
            tz="America/New_York",
        )

    def most_recent_completed_session(self, now=None):
        """Return the latest session whose exchange close has passed, or None."""

        current = self._now(now)
        schedule = self.schedule(current.date() - timedelta(days=14), current.date())
        completed = schedule.loc[schedule["market_close"] <= current]
        if completed.empty:
            return None
        row = completed.iloc[-1]
        return {
            "date": pd.Timestamp(completed.index[-1]).date().isoformat(),
            "market_close": pd.Timestamp(row["market_close"]),
        }

    def next_session_after(self, session_date):
        start = pd.Timestamp(session_date).date() + timedelta(days=1)
        schedule = self.schedule(start, start + timedelta(days=14))
        if schedule.empty:
            return None
        return pd.Timestamp(schedule.index[0]).date().isoformat()

    def readiness_for_bar(self, bar_date, now=None):
        """Describe whether a bar is current for the latest completed session."""

        latest = self.most_recent_completed_session(now)
        if latest is None:
            return {
                "status": "unavailable",
                "feature_as_of": str(bar_date) if bar_date is not None else None,
                "target_session": None,
                "expected_session": None,
            }
        try:
            parsed = pd.Timestamp(bar_date)
            observed = None if pd.isna(parsed) else parsed.date().isoformat()
        except (ValueError, TypeError):
            observed = None
        if observed is None:
            return {"status": "unavailable", "feature_as_of": None,
                    "target_session": None, "expected_session": latest["date"]}
        if observed != latest["date"]:
            return {
                "status": "stale",
                "feature_as_of": observed,
                "target_session": self.next_session_after(observed),
                "expected_session": latest["date"],
            }
        return {
            "status": "ready",
            "feature_as_of": observed,
            "target_session": self.next_session_after(observed),
            "expected_session": latest["date"],
        }
