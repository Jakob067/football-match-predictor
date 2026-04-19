from __future__ import annotations

from typing import Any

import pandas as pd
import requests


COLUMN_CANDIDATES: dict[str, list[str]] = {
    "date": ["Date", "date", "utcDate", "match_date"],
    "home_team": ["HomeTeam", "home_team", "homeTeam", "home"],
    "away_team": ["AwayTeam", "away_team", "awayTeam", "away"],
    "home_goals": ["FTHG", "home_goals", "homeGoals", "home_score"],
    "away_goals": ["FTAG", "away_goals", "awayGoals", "away_score"],
}

PLAYER_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "team": ["team", "Team", "team_name"],
    "player": ["player", "Player", "player_name"],
    "goals": ["goals", "Goals"],
    "assists": ["assists", "Assists"],
}

TEAM_STATS_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "team": ["team", "Team", "team_name"],
    "xg": ["xg", "xG", "expected_goals"],
    "possession": ["possession", "Possession", "ball_possession"],
    "shots": ["shots", "Shots", "shots_per_match"],
    "unavailable_players": ["unavailable_players", "injuries", "injured_players", "missing_players"],
}

def _to_canonical_dataframe(df: pd.DataFrame, override_map: dict[str, str] | None = None) -> pd.DataFrame:
    date_col = _resolve_column(df, "date", override_map)
    home_team_col = _resolve_column(df, "home_team", override_map)
    away_team_col = _resolve_column(df, "away_team", override_map)
    home_goals_col = _resolve_column(df, "home_goals", override_map)
    away_goals_col = _resolve_column(df, "away_goals", override_map)

    canonical = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce", utc=True),
            "home_team": df[home_team_col].astype(str).str.strip(),
            "away_team": df[away_team_col].astype(str).str.strip(),
            "home_goals": pd.to_numeric(df[home_goals_col], errors="coerce"),
            "away_goals": pd.to_numeric(df[away_goals_col], errors="coerce"),
        }
    )

    canonical = canonical.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"]).copy()
    canonical["home_goals"] = canonical["home_goals"].astype(int)
    canonical["away_goals"] = canonical["away_goals"].astype(int)
    canonical = canonical.sort_values("date").reset_index(drop=True)
    return canonical


def _to_player_stats_dataframe(
    df: pd.DataFrame,
    override_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    team_col = _resolve_column(df, "team", override_map)
    player_col = _resolve_column(df, "player", override_map)
    goals_col = _resolve_column(df, "goals", override_map)

    assists_col: str | None = None
    if override_map and "assists" in override_map:
        assists_col = override_map["assists"]
    else:
        for candidate in PLAYER_COLUMN_CANDIDATES["assists"]:
            if candidate in df.columns:
                assists_col = candidate
                break

    player_df = pd.DataFrame(
        {
            "team": df[team_col].astype(str).str.strip(),
            "player": df[player_col].astype(str).str.strip(),
            "goals": pd.to_numeric(df[goals_col], errors="coerce").fillna(0),
            "assists": 0 if assists_col is None else pd.to_numeric(df[assists_col], errors="coerce").fillna(0),
        }
    )
    player_df = player_df.dropna(subset=["team", "player"]).copy()
    player_df["goals"] = player_df["goals"].astype(float)
    player_df["assists"] = player_df["assists"].astype(float)
    return player_df


def _resolve_column(df: pd.DataFrame, logical_name: str, override_map: dict[str, str] | None = None) -> str:
    # Spaltenname aus Override oder Kandidatenliste bestimmen.
    if override_map and logical_name in override_map:
        return override_map[logical_name]

    candidates = COLUMN_CANDIDATES.get(logical_name, PLAYER_COLUMN_CANDIDATES.get(logical_name))
    if not candidates:
        raise ValueError(f"Unbekanntes logisches Feld: {logical_name}")

    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Keine passende Spalte für '{logical_name}' gefunden.")


def _resolve_optional_column(
    df: pd.DataFrame,
    logical_name: str,
    override_map: dict[str, str] | None = None,
) -> str | None:
    if override_map and logical_name in override_map:
        return override_map[logical_name]

    for candidate in TEAM_STATS_COLUMN_CANDIDATES.get(logical_name, []):
        if candidate in df.columns:
            return candidate
    return None


def load_matches_from_csv(csv_path: str, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    # CSV-Datei laden und in ein einheitliches Schema bringen.
    raw_df = pd.read_csv(csv_path)
    return _to_canonical_dataframe(raw_df, column_map)


def load_player_stats_from_csv(csv_path: str, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    # Optionales Spieler-CSV einlesen (z. B. Topscorer-Daten).
    raw_df = pd.read_csv(csv_path)
    return _to_player_stats_dataframe(raw_df, column_map)


def load_team_stats_from_csv(csv_path: str, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    # Team-Metriken (xG, Ballbesitz, Schüsse, Ausfälle) aus CSV laden.
    raw_df = pd.read_csv(csv_path)
    team_col = _resolve_column(raw_df, "team", column_map)
    xg_col = _resolve_optional_column(raw_df, "xg", column_map)
    possession_col = _resolve_optional_column(raw_df, "possession", column_map)
    shots_col = _resolve_optional_column(raw_df, "shots", column_map)
    unavailable_col = _resolve_optional_column(raw_df, "unavailable_players", column_map)

    team_stats_df = pd.DataFrame(
        {
            "team": raw_df[team_col].astype(str).str.strip(),
            "xg": (
                0.0
                if xg_col is None
                else pd.to_numeric(raw_df[xg_col], errors="coerce").fillna(0.0)
            ),
            "possession": (
                0.0
                if possession_col is None
                else pd.to_numeric(raw_df[possession_col], errors="coerce").fillna(0.0)
            ),
            "shots": (
                0.0
                if shots_col is None
                else pd.to_numeric(raw_df[shots_col], errors="coerce").fillna(0.0)
            ),
            "unavailable_players": (
                0.0
                if unavailable_col is None
                else pd.to_numeric(raw_df[unavailable_col], errors="coerce").fillna(0.0)
            ),
        }
    )
    return team_stats_df.dropna(subset=["team"]).reset_index(drop=True)


def _load_single_season_matches_from_api(
    api_token: str,
    competition_code: str = "PL",
    season: int | None = None,
) -> pd.DataFrame:
    # Daten über football-data.org abrufen.
    url = f"https://api.football-data.org/v4/competitions/{competition_code}/matches"
    headers = {"X-Auth-Token": api_token}
    params: dict[str, Any] = {"status": "FINISHED"}
    if season is not None:
        params["season"] = season

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    matches = payload.get("matches", [])

    records: list[dict[str, Any]] = []
    for match in matches:
        home_goals = match.get("score", {}).get("fullTime", {}).get("home")
        away_goals = match.get("score", {}).get("fullTime", {}).get("away")
        if home_goals is None or away_goals is None:
            continue

        records.append(
            {
                "date": match.get("utcDate"),
                "home_team": match.get("homeTeam", {}).get("name"),
                "away_team": match.get("awayTeam", {}).get("name"),
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
        )

    if not records:
        return pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals", "away_goals"])

    return _to_canonical_dataframe(pd.DataFrame(records))


def load_matches_from_api(
    api_token: str,
    competition_code: str = "PL",
    season: int | None = None,
    seasons_back: int = 1,
) -> pd.DataFrame:
    # Optional mehrere Saisons laden (z. B. 5+) für stabilere Trainingsdaten.
    if seasons_back < 1:
        raise ValueError("seasons_back muss mindestens 1 sein.")
    if season is None and seasons_back > 1:
        raise ValueError("Für seasons_back > 1 muss ein season-Wert gesetzt werden.")

    season_values: list[int | None]
    if season is None:
        season_values = [None]
    else:
        start = season - seasons_back + 1
        season_values = list(range(start, season + 1))

    all_frames = [
        _load_single_season_matches_from_api(
            api_token=api_token,
            competition_code=competition_code,
            season=season_value,
        )
        for season_value in season_values
    ]
    combined = pd.concat(all_frames, ignore_index=True)
    if combined.empty:
        return combined

    return (
        combined.drop_duplicates(
            subset=["date", "home_team", "away_team", "home_goals", "away_goals"],
        )
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_top_scorers_from_api(
    api_token: str,
    competition_code: str = "PL",
    season: int | None = None,
) -> pd.DataFrame:
    # Topscorer aus football-data.org abrufen (für Schlüsselspieler-Prognose).
    url = f"https://api.football-data.org/v4/competitions/{competition_code}/scorers"
    headers = {"X-Auth-Token": api_token}
    params: dict[str, Any] = {}
    if season is not None:
        params["season"] = season

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    scorers = payload.get("scorers", [])

    records: list[dict[str, Any]] = []
    for scorer in scorers:
        player_name = scorer.get("player", {}).get("name")
        team_name = scorer.get("team", {}).get("name")
        goals = scorer.get("goals") or 0
        assists = scorer.get("assists") or 0
        if not player_name or not team_name:
            continue
        records.append(
            {
                "team": team_name,
                "player": player_name,
                "goals": goals,
                "assists": assists,
            }
        )

    if not records:
        return pd.DataFrame(columns=["team", "player", "goals", "assists"])
    return _to_player_stats_dataframe(pd.DataFrame(records))


def load_upcoming_matches_from_api(
    api_token: str,
    competition_code: str = "PL",
    season: int | None = None,
    days_ahead: int = 3,
    now_utc: pd.Timestamp | None = None,
) -> pd.DataFrame:
    # Kommende Spiele im gewählten Zeitraum (ab heute) aus football-data.org laden.
    if days_ahead < 1:
        raise ValueError("days_ahead muss mindestens 1 sein.")

    reference_time = now_utc if now_utc is not None else pd.Timestamp.now(tz="UTC")
    date_from = reference_time.date().isoformat()
    date_to = (reference_time + pd.Timedelta(days=days_ahead)).date().isoformat()

    url = f"https://api.football-data.org/v4/competitions/{competition_code}/matches"
    headers = {"X-Auth-Token": api_token}
    params: dict[str, Any] = {
        "status": "SCHEDULED",
        "dateFrom": date_from,
        "dateTo": date_to,
    }
    if season is not None:
        params["season"] = season

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    matches = payload.get("matches", [])

    records: list[dict[str, Any]] = []
    for match in matches:
        match_id = match.get("id")
        utc_date = pd.to_datetime(match.get("utcDate"), errors="coerce", utc=True)
        home_team = match.get("homeTeam", {}).get("name")
        away_team = match.get("awayTeam", {}).get("name")

        if pd.isna(utc_date) or match_id is None or not home_team or not away_team:
            continue

        records.append(
            {
                "match_id": int(match_id),
                "date": utc_date,
                "home_team": str(home_team).strip(),
                "away_team": str(away_team).strip(),
                "status": str(match.get("status", "SCHEDULED")),
            }
        )

    if not records:
        return pd.DataFrame(columns=["match_id", "date", "home_team", "away_team", "status"])

    return pd.DataFrame(records).sort_values("date").reset_index(drop=True)
