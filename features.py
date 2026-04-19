from __future__ import annotations

from collections import defaultdict

import pandas as pd


RESULT_TO_CLASS = {"team_a_win": 0, "draw": 1, "team_b_win": 2}
CLASS_TO_TEXT_DE = {
    0: "Team A gewinnt",
    1: "Unentschieden",
    2: "Team B gewinnt",
}

FEATURE_COLUMNS = [
    "h2h_win_rate_last10",
    "team_a_form_points_last5",
    "team_b_form_points_last5",
    "team_a_avg_goals_for",
    "team_a_avg_goals_against",
    "team_b_avg_goals_for",
    "team_b_avg_goals_against",
    "home_advantage",
    "team_a_avg_goal_diff",
    "team_b_avg_goal_diff",
    "team_a_xg",
    "team_b_xg",
    "team_a_possession",
    "team_b_possession",
    "team_a_shots",
    "team_b_shots",
    "team_a_unavailable_players",
    "team_b_unavailable_players",
]


def _points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _label_from_score(goals_a: int, goals_b: int) -> str:
    if goals_a > goals_b:
        return "team_a_win"
    if goals_a == goals_b:
        return "draw"
    return "team_b_win"


def _team_stats(history: list[dict[str, int]]) -> dict[str, float]:
    # Statistiken nur aus bereits gespielten Partien berechnen.
    if not history:
        return {
            "form_points_last5": 0.0,
            "avg_goals_for": 0.0,
            "avg_goals_against": 0.0,
            "avg_goal_diff": 0.0,
        }

    last5 = history[-5:]
    form_points_last5 = float(sum(item["points"] for item in last5))
    avg_goals_for = float(sum(item["goals_for"] for item in history) / len(history))
    avg_goals_against = float(sum(item["goals_against"] for item in history) / len(history))
    avg_goal_diff = avg_goals_for - avg_goals_against

    return {
        "form_points_last5": form_points_last5,
        "avg_goals_for": avg_goals_for,
        "avg_goals_against": avg_goals_against,
        "avg_goal_diff": avg_goal_diff,
    }


def _h2h_win_rate(team_a: str, team_b: str, h2h_history: dict[tuple[str, str], list[str]], last_n: int = 10) -> float:
    key = tuple(sorted((team_a, team_b)))
    recent_results = h2h_history.get(key, [])[-last_n:]
    if not recent_results:
        # Neutralwert, falls es noch kein direktes Duell gab.
        return 0.5
    wins_team_a = sum(1 for winner in recent_results if winner == team_a)
    return float(wins_team_a / len(recent_results))


def _team_metric_lookup(
    team_stats_df: pd.DataFrame | None,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    metric_columns = ["xg", "possession", "shots", "unavailable_players"]
    defaults = {column: 0.0 for column in metric_columns}
    if team_stats_df is None or team_stats_df.empty:
        return {}, defaults

    required = {"team", *metric_columns}
    missing = [column for column in required if column not in team_stats_df.columns]
    if missing:
        raise ValueError(f"Team-Statistikdaten fehlen: {', '.join(sorted(missing))}")

    normalized_df = team_stats_df.copy()
    normalized_df["team"] = normalized_df["team"].astype(str).str.strip()
    for column in metric_columns:
        normalized_df[column] = pd.to_numeric(normalized_df[column], errors="coerce").fillna(0.0)

    grouped_df = (
        normalized_df.groupby("team", as_index=False)[metric_columns]
        .mean()
        .reset_index(drop=True)
    )
    if not grouped_df.empty:
        defaults = {
            column: float(grouped_df[column].mean())
            for column in metric_columns
        }

    lookup = {
        str(row.team): {
            "xg": float(row.xg),
            "possession": float(row.possession),
            "shots": float(row.shots),
            "unavailable_players": float(row.unavailable_players),
        }
        for row in grouped_df.itertuples(index=False)
    }
    return lookup, defaults


def _metrics_for_team(
    team: str,
    metrics_lookup: dict[str, dict[str, float]],
    defaults: dict[str, float],
) -> dict[str, float]:
    return metrics_lookup.get(team, defaults)


def build_feature_dataset(
    matches_df: pd.DataFrame,
    team_stats_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    # Trainingsdaten aus Matchhistorie erzeugen (ohne Datenleck).
    sorted_df = matches_df.sort_values("date").reset_index(drop=True)
    metrics_lookup, metric_defaults = _team_metric_lookup(team_stats_df)

    team_history: dict[str, list[dict[str, int]]] = defaultdict(list)
    h2h_history: dict[tuple[str, str], list[str]] = defaultdict(list)
    rows: list[dict[str, float | int | str]] = []

    for row in sorted_df.itertuples(index=False):
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        home_goals = int(row.home_goals)
        away_goals = int(row.away_goals)
        date = row.date

        home_stats = _team_stats(team_history[home_team])
        away_stats = _team_stats(team_history[away_team])
        home_metrics = _metrics_for_team(home_team, metrics_lookup, metric_defaults)
        away_metrics = _metrics_for_team(away_team, metrics_lookup, metric_defaults)

        # Perspektive 1: Team A ist Heimteam.
        rows.append(
            {
                "date": date,
                "team_a": home_team,
                "team_b": away_team,
                "h2h_win_rate_last10": _h2h_win_rate(home_team, away_team, h2h_history),
                "team_a_form_points_last5": home_stats["form_points_last5"],
                "team_b_form_points_last5": away_stats["form_points_last5"],
                "team_a_avg_goals_for": home_stats["avg_goals_for"],
                "team_a_avg_goals_against": home_stats["avg_goals_against"],
                "team_b_avg_goals_for": away_stats["avg_goals_for"],
                "team_b_avg_goals_against": away_stats["avg_goals_against"],
                "home_advantage": 1,
                "team_a_avg_goal_diff": home_stats["avg_goal_diff"],
                "team_b_avg_goal_diff": away_stats["avg_goal_diff"],
                "team_a_xg": home_metrics["xg"],
                "team_b_xg": away_metrics["xg"],
                "team_a_possession": home_metrics["possession"],
                "team_b_possession": away_metrics["possession"],
                "team_a_shots": home_metrics["shots"],
                "team_b_shots": away_metrics["shots"],
                "team_a_unavailable_players": home_metrics["unavailable_players"],
                "team_b_unavailable_players": away_metrics["unavailable_players"],
                "target_label": _label_from_score(home_goals, away_goals),
            }
        )

        # Perspektive 2: Team A ist Auswärtsteam.
        rows.append(
            {
                "date": date,
                "team_a": away_team,
                "team_b": home_team,
                "h2h_win_rate_last10": _h2h_win_rate(away_team, home_team, h2h_history),
                "team_a_form_points_last5": away_stats["form_points_last5"],
                "team_b_form_points_last5": home_stats["form_points_last5"],
                "team_a_avg_goals_for": away_stats["avg_goals_for"],
                "team_a_avg_goals_against": away_stats["avg_goals_against"],
                "team_b_avg_goals_for": home_stats["avg_goals_for"],
                "team_b_avg_goals_against": home_stats["avg_goals_against"],
                "home_advantage": 0,
                "team_a_avg_goal_diff": away_stats["avg_goal_diff"],
                "team_b_avg_goal_diff": home_stats["avg_goal_diff"],
                "team_a_xg": away_metrics["xg"],
                "team_b_xg": home_metrics["xg"],
                "team_a_possession": away_metrics["possession"],
                "team_b_possession": home_metrics["possession"],
                "team_a_shots": away_metrics["shots"],
                "team_b_shots": home_metrics["shots"],
                "team_a_unavailable_players": away_metrics["unavailable_players"],
                "team_b_unavailable_players": home_metrics["unavailable_players"],
                "target_label": _label_from_score(away_goals, home_goals),
            }
        )

        home_points = _points(home_goals, away_goals)
        away_points = _points(away_goals, home_goals)
        team_history[home_team].append(
            {"goals_for": home_goals, "goals_against": away_goals, "points": home_points}
        )
        team_history[away_team].append(
            {"goals_for": away_goals, "goals_against": home_goals, "points": away_points}
        )

        h2h_key = tuple(sorted((home_team, away_team)))
        if home_goals > away_goals:
            winner = home_team
        elif away_goals > home_goals:
            winner = away_team
        else:
            winner = "draw"
        h2h_history[h2h_key].append(winner)

    feature_df = pd.DataFrame(rows)
    feature_df["target"] = feature_df["target_label"].map(RESULT_TO_CLASS).astype(int)
    return feature_df


def _team_history_from_matches(team: str, matches_df: pd.DataFrame) -> list[dict[str, int]]:
    team_matches = matches_df[
        (matches_df["home_team"] == team) | (matches_df["away_team"] == team)
    ].sort_values("date")

    history: list[dict[str, int]] = []
    for row in team_matches.itertuples(index=False):
        if row.home_team == team:
            goals_for = int(row.home_goals)
            goals_against = int(row.away_goals)
        else:
            goals_for = int(row.away_goals)
            goals_against = int(row.home_goals)
        history.append(
            {
                "goals_for": goals_for,
                "goals_against": goals_against,
                "points": _points(goals_for, goals_against),
            }
        )
    return history


def _h2h_history(team_a: str, team_b: str, matches_df: pd.DataFrame) -> list[str]:
    h2h_matches = matches_df[
        ((matches_df["home_team"] == team_a) & (matches_df["away_team"] == team_b))
        | ((matches_df["home_team"] == team_b) & (matches_df["away_team"] == team_a))
    ].sort_values("date")

    outcomes: list[str] = []
    for row in h2h_matches.itertuples(index=False):
        if row.home_goals == row.away_goals:
            outcomes.append("draw")
        elif (row.home_goals > row.away_goals and row.home_team == team_a) or (
            row.away_goals > row.home_goals and row.away_team == team_a
        ):
            outcomes.append(team_a)
        else:
            outcomes.append(team_b)
    return outcomes


def build_prediction_features(
    team_a: str,
    team_b: str,
    matches_df: pd.DataFrame,
    team_stats_df: pd.DataFrame | None = None,
    team_a_is_home: bool = True,
) -> pd.DataFrame:
    # Features für genau ein neues Match erzeugen.
    metrics_lookup, metric_defaults = _team_metric_lookup(team_stats_df)
    team_a_metrics = _metrics_for_team(team_a, metrics_lookup, metric_defaults)
    team_b_metrics = _metrics_for_team(team_b, metrics_lookup, metric_defaults)

    history_a = _team_history_from_matches(team_a, matches_df)
    history_b = _team_history_from_matches(team_b, matches_df)
    stats_a = _team_stats(history_a)
    stats_b = _team_stats(history_b)
    h2h_outcomes = _h2h_history(team_a, team_b, matches_df)[-10:]

    if not h2h_outcomes:
        h2h_rate = 0.5
    else:
        h2h_rate = float(sum(1 for winner in h2h_outcomes if winner == team_a) / len(h2h_outcomes))

    features = {
        "h2h_win_rate_last10": h2h_rate,
        "team_a_form_points_last5": stats_a["form_points_last5"],
        "team_b_form_points_last5": stats_b["form_points_last5"],
        "team_a_avg_goals_for": stats_a["avg_goals_for"],
        "team_a_avg_goals_against": stats_a["avg_goals_against"],
        "team_b_avg_goals_for": stats_b["avg_goals_for"],
        "team_b_avg_goals_against": stats_b["avg_goals_against"],
        "home_advantage": 1 if team_a_is_home else 0,
        "team_a_avg_goal_diff": stats_a["avg_goal_diff"],
        "team_b_avg_goal_diff": stats_b["avg_goal_diff"],
        "team_a_xg": team_a_metrics["xg"],
        "team_b_xg": team_b_metrics["xg"],
        "team_a_possession": team_a_metrics["possession"],
        "team_b_possession": team_b_metrics["possession"],
        "team_a_shots": team_a_metrics["shots"],
        "team_b_shots": team_b_metrics["shots"],
        "team_a_unavailable_players": team_a_metrics["unavailable_players"],
        "team_b_unavailable_players": team_b_metrics["unavailable_players"],
    }

    return pd.DataFrame([features], columns=FEATURE_COLUMNS)
