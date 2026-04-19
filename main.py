from __future__ import annotations

import argparse
import os
import unittest

import pandas as pd

from data_loader import (
    load_matches_from_api,
    load_matches_from_csv,
    load_player_stats_from_csv,
    load_team_stats_from_csv,
    load_top_scorers_from_api,
)
from features import CLASS_TO_TEXT_DE, FEATURE_COLUMNS, build_feature_dataset
from modeling import train_model
from predictor import initialize_predictor, predict_match
from visualization import plot_confusion_matrix_heatmap, plot_feature_importance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fußballspiel-Ausgang mit ML-Modellen vorhersagen")
    parser.add_argument("--source", choices=["csv", "api"], default="csv", help="Datenquelle")
    parser.add_argument("--csv-path", type=str, help="Pfad zur CSV-Datei")
    parser.add_argument("--api-token", type=str, help="football-data.org API Token")
    parser.add_argument("--competition", type=str, default="PL", help="Wettbewerbscode, z. B. PL")
    parser.add_argument("--season", type=int, help="Saisonjahr, z. B. 2024")
    parser.add_argument("--history-seasons", type=int, default=5, help="Anzahl Saisons für Trainingshistorie (API)")
    parser.add_argument(
        "--model",
        choices=["random_forest", "xgboost", "neural_net"],
        default="random_forest",
        help="Modelltyp für das Training",
    )
    parser.add_argument("--player-csv-path", type=str, help="Optionaler CSV-Pfad mit Spielerstatistiken")
    parser.add_argument(
        "--team-stats-csv-path",
        type=str,
        help="Optionaler CSV-Pfad mit Teammetriken (xG, Ballbesitz, Schüsse, Ausfälle)",
    )
    parser.add_argument("--team-a", type=str, help="Team A für eine Beispielvorhersage")
    parser.add_argument("--team-b", type=str, help="Team B für eine Beispielvorhersage")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Ordner für Plots")
    parser.add_argument("--auto-test", action="store_true", help="Automatische Tests vor dem Lauf ausführen")
    return parser.parse_args()


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if args.source == "csv":
        if not args.csv_path:
            raise ValueError("Bei --source csv muss --csv-path angegeben werden.")
        matches_df = load_matches_from_csv(args.csv_path)
        if args.player_csv_path:
            player_df = load_player_stats_from_csv(args.player_csv_path)
        else:
            player_df = pd.DataFrame(columns=["team", "player", "goals", "assists"])
        if args.team_stats_csv_path:
            team_stats_df = load_team_stats_from_csv(args.team_stats_csv_path)
        else:
            team_stats_df = pd.DataFrame(
                columns=["team", "xg", "possession", "shots", "unavailable_players"]
            )
        return matches_df, player_df, team_stats_df

    token = args.api_token or os.getenv("FOOTBALL_DATA_API_TOKEN") or os.getenv("FOOTBALL_DATA_API_KEY")
    if not token:
        raise ValueError("Für API-Nutzung wird --api-token, FOOTBALL_DATA_API_TOKEN oder FOOTBALL_DATA_API_KEY benötigt.")
    matches_df = load_matches_from_api(
        token,
        competition_code=args.competition,
        season=args.season,
        seasons_back=args.history_seasons,
    )
    player_df = load_top_scorers_from_api(token, competition_code=args.competition, season=args.season)
    if args.team_stats_csv_path:
        team_stats_df = load_team_stats_from_csv(args.team_stats_csv_path)
    else:
        team_stats_df = pd.DataFrame(columns=["team", "xg", "possession", "shots", "unavailable_players"])
    return matches_df, player_df, team_stats_df


def run_automatic_tests() -> None:
    # Projektweite Tests automatisch ausführen.
    suite = unittest.defaultTestLoader.discover("tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Automatische Tests sind fehlgeschlagen.")


def run_pipeline(args: argparse.Namespace) -> None:
    if args.auto_test:
        run_automatic_tests()

    matches_df, player_df, team_stats_df = load_data(args)
    feature_df = build_feature_dataset(matches_df, team_stats_df=team_stats_df)
    training_result = train_model(feature_df, model_type=args.model)

    print(f"Accuracy: {training_result['accuracy']:.4f}")

    class_labels = [CLASS_TO_TEXT_DE[idx] for idx in [0, 1, 2]]
    conf_matrix_df = pd.DataFrame(
        training_result["confusion_matrix"],
        index=class_labels,
        columns=class_labels,
    )
    print("\nConfusion Matrix:")
    print(conf_matrix_df)

    os.makedirs(args.output_dir, exist_ok=True)
    plot_feature_importance(
        training_result["model"],
        FEATURE_COLUMNS,
        X_reference=training_result["X_test"],
        y_reference=training_result["y_test"],
        output_path=os.path.join(args.output_dir, "feature_importance.png"),
    )
    plot_confusion_matrix_heatmap(
        training_result["confusion_matrix"],
        class_labels,
        output_path=os.path.join(args.output_dir, "confusion_matrix_heatmap.png"),
    )

    initialize_predictor(training_result["model"], matches_df, player_df, team_stats=team_stats_df)
    if args.team_a and args.team_b:
        print("\n" + predict_match(args.team_a, args.team_b))


if __name__ == "__main__":
    run_pipeline(parse_args())
