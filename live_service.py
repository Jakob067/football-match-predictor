from __future__ import annotations

import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import requests

from prediction_engine import Team, predict, teams_for

API_ROOT = "https://api.football-data.org/v4"
_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def api_token() -> str | None:
    return os.getenv("FOOTBALL_DATA_API_TOKEN") or os.getenv("FOOTBALL_DATA_API_KEY")


def _get_matches(token: str, competition: str, status: str) -> list[dict[str, Any]]:
    response = requests.get(
        f"{API_ROOT}/competitions/{competition}/matches",
        headers={"X-Auth-Token": token},
        params={"status": status},
        timeout=20,
    )
    response.raise_for_status()
    return response.json().get("matches", [])


def _display_date(value: str) -> tuple[str, str]:
    date = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone()
    return date.strftime("%a, %d.%m."), date.strftime("%H:%M")


def _initial_rating(name: str, competition: str) -> float:
    normalized = "".join(ch for ch in name.lower() if ch.isalnum())
    best_rating, best_overlap = 1500.0, 0
    for team in teams_for(competition):
        candidate = "".join(ch for ch in team.name.lower() if ch.isalnum())
        overlap = len(set(normalized) & set(candidate))
        if normalized in candidate or candidate in normalized:
            return float(team.rating)
        if overlap > best_overlap and overlap >= min(len(set(normalized)), len(set(candidate))) * .8:
            best_rating, best_overlap = float(team.rating), overlap
    return best_rating


def calculate_ratings(matches: list[dict[str, Any]], competition: str) -> tuple[dict[int, float], dict[int, list[int]]]:
    """Chronological Elo with goal-margin weighting and recent form tracking."""
    ratings: dict[int, float] = {}
    form: dict[int, list[int]] = defaultdict(list)
    ordered = sorted(matches, key=lambda match: match.get("utcDate", ""))
    for match in ordered:
        home, away = match.get("homeTeam", {}), match.get("awayTeam", {})
        home_id, away_id = home.get("id"), away.get("id")
        score = match.get("score", {}).get("fullTime", {})
        home_goals, away_goals = score.get("home"), score.get("away")
        if None in (home_id, away_id, home_goals, away_goals):
            continue
        ratings.setdefault(home_id, _initial_rating(home.get("name", ""), competition))
        ratings.setdefault(away_id, _initial_rating(away.get("name", ""), competition))
        expected = 1 / (1 + 10 ** ((ratings[away_id] - ratings[home_id] - 55) / 400))
        actual = 1.0 if home_goals > away_goals else .5 if home_goals == away_goals else 0.0
        margin = abs(home_goals - away_goals)
        k = 22 * (1 + min(margin, 4) * .18)
        change = k * (actual - expected)
        ratings[home_id] += change
        ratings[away_id] -= change
        form[home_id].append(3 if actual == 1 else 1 if actual == .5 else 0)
        form[away_id].append(3 if actual == 0 else 1 if actual == .5 else 0)
        form[home_id] = form[home_id][-5:]
        form[away_id] = form[away_id][-5:]
    return ratings, form


def build_predictions(finished: list[dict[str, Any]], scheduled: list[dict[str, Any]], competition: str) -> list[dict[str, Any]]:
    ratings, form = calculate_ratings(finished, competition)
    rows = []
    now = datetime.now(timezone.utc)
    scheduled = sorted(scheduled, key=lambda match: match.get("utcDate", ""))[:16]
    for match in scheduled:
        home, away = match.get("homeTeam", {}), match.get("awayTeam", {})
        home_id, away_id = home.get("id"), away.get("id")
        if home_id is None or away_id is None or not match.get("utcDate"):
            continue
        match_date = datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00"))
        if match_date < now:
            continue
        home_rating = ratings.get(home_id, _initial_rating(home.get("name", ""), competition))
        away_rating = ratings.get(away_id, _initial_rating(away.get("name", ""), competition))
        # Recent form is regressed heavily: maximum adjustment is only 28 Elo points.
        home_points, away_points = sum(form.get(home_id, [8])), sum(form.get(away_id, [8]))
        home_rating += max(-28, min(28, (home_points - 8) * 4))
        away_rating += max(-28, min(28, (away_points - 8) * 4))
        forecast = predict(Team(home.get("name", "Home"), round(home_rating)),
                           Team(away.get("name", "Away"), round(away_rating)))
        day, kickoff = _display_date(match["utcDate"])
        rows.append({**forecast, "id": match.get("id"), "day": day, "kickoff": kickoff,
            "home_name": home.get("shortName") or home.get("name"),
            "away_name": away.get("shortName") or away.get("name"),
            "home_crest": home.get("crest"), "away_crest": away.get("crest"),
            "home_form": home_points, "away_form": away_points})
    return rows


def next_predictions(competition: str, ttl_seconds: int = 900) -> list[dict[str, Any]]:
    token = api_token()
    if not token:
        raise RuntimeError("API_TOKEN_MISSING")
    cached = _CACHE.get(competition)
    if cached and time.monotonic() - cached[0] < ttl_seconds:
        return cached[1]
    finished = _get_matches(token, competition, "FINISHED")
    scheduled = _get_matches(token, competition, "SCHEDULED,TIMED")
    result = build_predictions(finished, scheduled, competition)
    _CACHE[competition] = (time.monotonic(), result)
    return result
