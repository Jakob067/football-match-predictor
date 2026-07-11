from __future__ import annotations

import math
import time
from typing import Any

import requests

from live_service import API_ROOT, api_token

_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def _normalize(value: str | None) -> str:
    if not value:
        return ""
    replacements = {"footballclub":"", "futbolclub":"", "fc":"", "cf":"", "afc":""}
    result = "".join(character for character in value.lower() if character.isalnum())
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def _same_team(left: str | None, right: str | None) -> bool:
    a, b = _normalize(left), _normalize(right)
    return bool(a and b and (a in b or b in a))


def current_scorers(competition: str, ttl_seconds: int = 1800) -> list[dict[str, Any]]:
    cached = _CACHE.get(competition)
    if cached and time.monotonic() - cached[0] < ttl_seconds:
        return cached[1]
    token = api_token()
    if not token:
        return []
    response = requests.get(
        f"{API_ROOT}/competitions/{competition}/scorers",
        headers={"X-Auth-Token": token},
        params={"limit": 100},
        timeout=20,
    )
    response.raise_for_status()
    rows = response.json().get("scorers", [])
    _CACHE[competition] = (time.monotonic(), rows)
    return rows


def _player_candidates(
    scorers: list[dict[str, Any]], team_name: str, team_xg: float, side: str
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in scorers:
        team = row.get("team", {})
        if not (_same_team(team_name, team.get("name")) or _same_team(team_name, team.get("shortName"))):
            continue
        goals = int(row.get("goals") or 0)
        appearances = int(row.get("playedMatches") or 0)
        if appearances < 1:
            continue
        rate = goals / appearances
        # Scale the player's current scoring rate by this match's expected team attack.
        expected_player_goals = min(1.6, rate * max(.45, team_xg / 1.35))
        probability = (1 - math.exp(-expected_player_goals)) * 100
        candidates.append({
            "name": row.get("player", {}).get("name") or "Unbekannt",
            "team": team_name,
            "side": side,
            "goals": goals,
            "appearances": appearances,
            "probability": round(probability),
        })
    return sorted(candidates, key=lambda player: (player["probability"], player["goals"]), reverse=True)


def add_scorer_predictions(
    matches: list[dict[str, Any]], competition: str
) -> list[dict[str, Any]]:
    try:
        scorers = current_scorers(competition)
    except requests.RequestException:
        scorers = []
    for match in matches:
        home = _player_candidates(scorers, str(match["home_name"]), float(match["home_xg"]), "home")
        away = _player_candidates(scorers, str(match["away_name"]), float(match["away_xg"]), "away")
        # Keep both teams represented where data exists, then fill the third slot by probability.
        selected = home[:1] + away[:1]
        remaining = [player for player in home[1:] + away[1:] if player not in selected]
        selected.extend(sorted(remaining, key=lambda player: player["probability"], reverse=True)[:1])
        match["scorers"] = sorted(selected, key=lambda player: player["probability"], reverse=True)
    return matches
