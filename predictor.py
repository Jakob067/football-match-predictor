from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from features import CLASS_TO_TEXT_DE, build_prediction_features


@dataclass
class PredictorContext:
    model: Any
    historical_matches: pd.DataFrame
    player_stats: pd.DataFrame
    team_stats: pd.DataFrame


_PREDICTOR_CONTEXT: PredictorContext | None = None


def initialize_predictor(
    model: Any,
    historical_matches: pd.DataFrame,
    player_stats: pd.DataFrame | None = None,
    team_stats: pd.DataFrame | None = None,
) -> None:
    # Trainiertes Modell und historische Daten für spätere Vorhersagen speichern.
    global _PREDICTOR_CONTEXT
    _PREDICTOR_CONTEXT = PredictorContext(
        model=model,
        historical_matches=historical_matches.copy(),
        player_stats=(
            player_stats.copy()
            if player_stats is not None
            else pd.DataFrame(columns=["team", "player", "goals", "assists"])
        ),
        team_stats=(
            team_stats.copy()
            if team_stats is not None
            else pd.DataFrame(columns=["team", "xg", "possession", "shots", "unavailable_players"])
        ),
    )


def _predict_key_player(team_a: str, team_b: str, player_stats: pd.DataFrame) -> str:
    if player_stats.empty:
        return "Keine Spielerstatistiken verfügbar"

    candidates = player_stats[player_stats["team"].isin([team_a, team_b])].copy()
    if candidates.empty:
        return "Keine Spielerstatistiken für diese Teams verfügbar"

    # Heuristik: Tore zählen stärker als Assists.
    candidates["impact_score"] = candidates["goals"] * 1.0 + candidates["assists"] * 0.7
    best = candidates.sort_values("impact_score", ascending=False).iloc[0]
    return f"{best['player']} ({best['team']})"


def predict_match(team_a: str, team_b: str) -> str:
    # Vorhersagefunktion mit gewünschter Signatur.
    if _PREDICTOR_CONTEXT is None:
        raise RuntimeError("Predictor wurde noch nicht initialisiert. Bitte zuerst trainieren.")

    feature_df = build_prediction_features(
        team_a,
        team_b,
        _PREDICTOR_CONTEXT.historical_matches,
        team_stats_df=_PREDICTOR_CONTEXT.team_stats,
        team_a_is_home=True,
    )
    probabilities = _PREDICTOR_CONTEXT.model.predict_proba(feature_df)[0]
    best_class = int(np.argmax(probabilities))
    best_probability = float(probabilities[best_class] * 100.0)
    result_text = CLASS_TO_TEXT_DE[best_class]
    key_player = _predict_key_player(team_a, team_b, _PREDICTOR_CONTEXT.player_stats)
    return (
        f"Vorhersage: {result_text} ({best_probability:.1f}% Wahrscheinlichkeit) | "
        f"Schlüsselspieler: {key_player}"
    )
