from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

import requests

from live_service import calculate_ratings
from prediction_engine import Team, predict

STANDINGS_URL = "https://api.football-data.org/v4/competitions/WC/standings?stage=GROUP_STAGE"
MATCHES_URL = "https://api.football-data.org/v4/competitions/WC/matches?status=SCHEDULED"
FINISHED_URL = "https://api.football-data.org/v4/competitions/WC/matches?status=FINISHED"
_cache: tuple[float, list[dict[str, Any]]] | None = None


def _token() -> str:
    token = os.getenv("FOOTBALL_DATA_API_TOKEN") or os.getenv("FOOTBALL_DATA_API_KEY")
    if not token:
        raise RuntimeError("API_TOKEN_MISSING")
    return token


def _get(url: str, token: str) -> dict[str, Any]:
    response = requests.get(url, headers={"X-Auth-Token": token}, timeout=20)
    response.raise_for_status()
    return response.json()


def _standing_adjustments(payload: dict[str, Any]) -> dict[int, float]:
    """Convert group performance into a conservative Elo adjustment."""
    adjustments: dict[int, float] = {}
    for standing in payload.get("standings", []):
        if standing.get("type") not in (None, "TOTAL"):
            continue
        for row in standing.get("table", []):
            team_id = row.get("team", {}).get("id")
            played = row.get("playedGames") or 0
            if team_id is None or not played:
                continue
            points_per_game = (row.get("points") or 0) / played
            goal_difference_per_game = (row.get("goalDifference") or 0) / played
            # Group tables are tiny samples, so cap their influence strongly.
            adjustment = (points_per_game - 1.5) * 16 + goal_difference_per_game * 5
            adjustments[team_id] = max(-32.0, min(32.0, adjustment))
    return adjustments


def _local_date(value: str) -> tuple[str, str]:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone()
    return parsed.strftime("%a, %d.%m."), parsed.strftime("%H:%M")


def next_world_cup_predictions(ttl_seconds: int = 900) -> list[dict[str, Any]]:
    global _cache
    if _cache and time.monotonic() - _cache[0] < ttl_seconds:
        return _cache[1]

    token = _token()
    standings = _get(STANDINGS_URL, token)
    scheduled = _get(MATCHES_URL, token).get("matches", [])
    finished = _get(FINISHED_URL, token).get("matches", [])
    ratings, form = calculate_ratings(finished, "WC")
    table_adjustment = _standing_adjustments(standings)
    now = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []

    for match in sorted(scheduled, key=lambda item: item.get("utcDate", ""))[:24]:
        utc_date = match.get("utcDate")
        home, away = match.get("homeTeam", {}), match.get("awayTeam", {})
        home_id, away_id = home.get("id"), away.get("id")
        if not utc_date or home_id is None or away_id is None:
            continue
        match_time = datetime.fromisoformat(utc_date.replace("Z", "+00:00"))
        if match_time < now:
            continue
        home_form, away_form = sum(form.get(home_id, [8])), sum(form.get(away_id, [8]))
        home_rating = ratings.get(home_id, 1700.0) + table_adjustment.get(home_id, 0)
        away_rating = ratings.get(away_id, 1700.0) + table_adjustment.get(away_id, 0)
        home_rating += max(-24, min(24, (home_form - 8) * 4))
        away_rating += max(-24, min(24, (away_form - 8) * 4))
        forecast = predict(Team(home.get("name") or "Team A", round(home_rating)),
                           Team(away.get("name") or "Team B", round(away_rating)), neutral=True)
        day, kickoff = _local_date(utc_date)
        rows.append({**forecast, "id": match.get("id"), "day": day, "kickoff": kickoff,
            "home_name": home.get("shortName") or home.get("name"),
            "away_name": away.get("shortName") or away.get("name"),
            "home_crest": home.get("crest"), "away_crest": away.get("crest"),
            "home_form": home_form, "away_form": away_form})

    _cache = (time.monotonic(), rows)
    return rows
