#!/usr/bin/env python3
"""
News filter for live scalp trading.

Pauses the bot during high-impact economic news windows that cause extreme
gold volatility (FOMC, NFP, CPI, PPI, GDP, etc.).

Two layers:
  1. Recurring schedule  — weekly/monthly events at fixed UTC times
  2. Manual blacklist    — load specific dates from news_blacklist.json

Usage:
  filter = NewsFilter(pause_minutes_before=10, pause_minutes_after=20)
  if filter.is_news_window():
      # skip this bar
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# ── High-impact recurring events (UTC times) ──────────────────────────────────
#
# Format: list of dicts with:
#   name       : event name
#   weekday    : 0=Mon … 6=Sun (None = any day)
#   week       : 1=first, 2=second … -1=last week of month (None = every week)
#   hour / min : UTC time of release
#
RECURRING_EVENTS = [
    # US Non-Farm Payrolls — 1st Friday of month, 13:30 UTC
    {'name': 'NFP',       'weekday': 4, 'week': 1,    'hour': 13, 'min': 30},
    # US CPI — usually 2nd or 3rd Tuesday/Wednesday, 13:30 UTC
    {'name': 'CPI',       'weekday': 1, 'week': None, 'hour': 13, 'min': 30},
    # US PPI — usually day after CPI, 13:30 UTC
    {'name': 'PPI',       'weekday': 2, 'week': None, 'hour': 13, 'min': 30},
    # US GDP — last Wednesday of month, 13:30 UTC (quarterly but safe to block)
    {'name': 'GDP',       'weekday': 2, 'week': -1,   'hour': 13, 'min': 30},
    # FOMC Rate Decision — ~8 times/year, Wednesday 19:00 UTC (2pm ET)
    {'name': 'FOMC',      'weekday': 2, 'week': None, 'hour': 19, 'min': 0},
    # FOMC Press Conference — same day, 19:30 UTC
    {'name': 'FOMC_PC',   'weekday': 2, 'week': None, 'hour': 19, 'min': 30},
    # US Retail Sales — mid-month, 13:30 UTC
    {'name': 'Retail',    'weekday': 0, 'week': None, 'hour': 13, 'min': 30},
    # ISM Manufacturing — 1st business day, 15:00 UTC
    {'name': 'ISM_Mfg',   'weekday': 0, 'week': 1,   'hour': 15, 'min': 0},
    # US Initial Jobless Claims — every Thursday 13:30 UTC
    {'name': 'Jobless',   'weekday': 3, 'week': None, 'hour': 13, 'min': 30},
    # Fed Chair speeches — Wednesdays variable time, block 18:00-20:00 UTC
    {'name': 'Fed_Speech','weekday': 2, 'week': None, 'hour': 18, 'min': 0},
]

BLACKLIST_FILE = Path(__file__).parent / 'news_blacklist.json'


class NewsFilter:
    def __init__(
        self,
        pause_minutes_before: int = 15,
        pause_minutes_after: int  = 30,
        enabled: bool = True,
    ):
        self.before  = timedelta(minutes=pause_minutes_before)
        self.after   = timedelta(minutes=pause_minutes_after)
        self.enabled = enabled
        self._blacklist = self._load_blacklist()

    def _load_blacklist(self) -> list:
        """Load specific news datetimes from news_blacklist.json if it exists."""
        if not BLACKLIST_FILE.exists():
            return []
        try:
            with open(BLACKLIST_FILE) as f:
                raw = json.load(f)
            # Expect list of ISO strings: ["2026-01-29T19:00:00Z", ...]
            return [datetime.fromisoformat(s.replace('Z', '+00:00')) for s in raw]
        except Exception as e:
            logger.warning(f"[NewsFilter] Could not load blacklist: {e}")
            return []

    def is_news_window(self, now: datetime = None) -> tuple:
        """
        Returns (is_blocked: bool, reason: str).
        Pass now= for testing; defaults to current UTC time.
        """
        if not self.enabled:
            return False, ''

        now = now or datetime.now(timezone.utc)

        # Check manual blacklist first
        for event_dt in self._blacklist:
            if (event_dt - self.before) <= now <= (event_dt + self.after):
                reason = f"blacklist event @ {event_dt.strftime('%Y-%m-%d %H:%M')} UTC"
                return True, reason

        # Check recurring schedule
        for event in RECURRING_EVENTS:
            if self._matches_recurring(now, event):
                window_start = now.replace(
                    hour=event['hour'], minute=event['min'],
                    second=0, microsecond=0
                )
                if (window_start - self.before) <= now <= (window_start + self.after):
                    reason = f"{event['name']} window ({event['hour']:02d}:{event['min']:02d} UTC)"
                    return True, reason

        return False, ''

    def _matches_recurring(self, now: datetime, event: dict) -> bool:
        """Check if today matches the event's weekday and week-of-month."""
        if event['weekday'] is not None and now.weekday() != event['weekday']:
            return False
        if event['week'] is not None:
            week = event['week']
            day  = now.day
            if week > 0:
                # nth occurrence: day 1-7 = week 1, 8-14 = week 2, etc.
                actual_week = (day - 1) // 7 + 1
                if actual_week != week:
                    return False
            elif week == -1:
                # last occurrence: day must be in last 7 days of month
                import calendar
                last_day = calendar.monthrange(now.year, now.month)[1]
                if day < last_day - 6:
                    return False
        return True


# ── Convenience function ──────────────────────────────────────────────────────

_default_filter = None

def get_filter(**kwargs) -> NewsFilter:
    global _default_filter
    if _default_filter is None:
        _default_filter = NewsFilter(**kwargs)
    return _default_filter
