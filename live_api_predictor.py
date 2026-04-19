from __future__ import annotations

import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Basiskonfiguration für football-data.org
API_KEY = os.getenv("FOOTBALL_DATA_API_TOKEN") or os.getenv("FOOTBALL_DATA_API_KEY")
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}
LEAGUES = {"Bundesliga": "BL1", "Champions League": "CL", "Premier League": "PL"}

FEATURE_COLUMNS = [
    "h2h_home_winrate",
    "home_form",
    "away_form",
    "home_avg_goals_scored",
    "away_avg_goals_scored",
    "home_avg_goals_conceded",
    "away_avg_goals_conceded",
    "home_advantage",
]


def fetch_matches(league_code: str) -> pd.DataFrame:
    # Abgeschlossene Spiele je Liga live von der API laden.
    url = f"{BASE_URL}/competitions/{league_code}/matches"
    params = {"status": "FINISHED"}

    attempts = 0
    while attempts < 3:
        attempts += 1
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "6"))
                time.sleep(max(6, retry_after))
                continue
            response.raise_for_status()
            payload = response.json()
            matches = payload.get("matches", [])

            records: list[dict[str, object]] = []
            for match in matches:
                try:
                    full_time = match["score"]["fullTime"]
                    home_goals = full_time.get("home")
                    away_goals = full_time.get("away")
                    if home_goals is None or away_goals is None:
                        continue

                    if home_goals > away_goals:
                        result = 2
                    elif home_goals == away_goals:
                        result = 1
                    else:
                        result = 0

                    records.append(
                        {
                            "date": pd.to_datetime(match.get("utcDate"), utc=True, errors="coerce"),
                            "home_team": str(match["homeTeam"]["name"]).strip(),
                            "away_team": str(match["awayTeam"]["name"]).strip(),
                            "home_goals": int(home_goals),
                            "away_goals": int(away_goals),
                            "result": result,
                            "league_code": league_code,
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    # Unvollständige oder kaputte Matchdaten werden übersprungen.
                    continue

            df = pd.DataFrame(records)
            if df.empty:
                return pd.DataFrame(
                    columns=[
                        "date",
                        "home_team",
                        "away_team",
                        "home_goals",
                        "away_goals",
                        "result",
                        "league_code",
                    ]
                )
            return df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        except requests.RequestException:
            # Netz- oder API-Fehler: kurz warten und erneut versuchen.
            time.sleep(6)

    return pd.DataFrame(
        columns=[
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "result",
            "league_code",
        ]
    )


def fetch_all_leagues() -> pd.DataFrame:
    # Alle geforderten Ligen abrufen, mit Rate-Limit-Abstand zwischen Requests.
    all_frames: list[pd.DataFrame] = []
    items = list(LEAGUES.items())
    for idx, (_, code) in enumerate(items):
        all_frames.append(fetch_matches(code))
        if idx < len(items) - 1:
            time.sleep(6)

    combined = pd.concat(all_frames, ignore_index=True)
    if combined.empty:
        raise RuntimeError("Keine Spieldaten von der API erhalten.")
    return combined.sort_values("date").reset_index(drop=True)


def _team_points_for_match(team: str, home_team: str, away_team: str, home_goals: int, away_goals: int) -> int:
    # Punkte aus Sicht eines Teams berechnen.
    if team == home_team:
        if home_goals > away_goals:
            return 3
        if home_goals == away_goals:
            return 1
        return 0
    if away_goals > home_goals:
        return 3
    if away_goals == home_goals:
        return 1
    return 0


def _history_stats(history: list[dict[str, int]]) -> dict[str, float]:
    # Form- und Torstatistiken aus einer Teamhistorie berechnen.
    if not history:
        return {"form5": 0.0, "avg_scored10": 0.0, "avg_conceded10": 0.0}

    last5 = history[-5:]
    last10 = history[-10:]
    form5 = float(sum(item["points"] for item in last5))
    avg_scored10 = float(sum(item["goals_for"] for item in last10) / len(last10))
    avg_conceded10 = float(sum(item["goals_against"] for item in last10) / len(last10))
    return {"form5": form5, "avg_scored10": avg_scored10, "avg_conceded10": avg_conceded10}


def _h2h_home_winrate(home_team: str, away_team: str, h2h_history: dict[tuple[str, str], list[str]]) -> float:
    # Heimteam-Siegquote in den letzten 10 direkten Duellen.
    key = tuple(sorted((home_team, away_team)))
    recent = h2h_history.get(key, [])[-10:]
    if not recent:
        return 0.5
    home_wins = sum(1 for winner in recent if winner == home_team)
    return float(home_wins / len(recent))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Für jedes Match Trainingsfeatures aus der Historie aufbauen (ohne Datenleck).
    df_sorted = df.sort_values("date").reset_index(drop=True)
    team_history: dict[str, list[dict[str, int]]] = defaultdict(list)
    h2h_history: dict[tuple[str, str], list[str]] = defaultdict(list)
    rows: list[dict[str, object]] = []

    for match in df_sorted.itertuples(index=False):
        home_team = str(match.home_team)
        away_team = str(match.away_team)
        home_goals = int(match.home_goals)
        away_goals = int(match.away_goals)

        home_stats = _history_stats(team_history[home_team])
        away_stats = _history_stats(team_history[away_team])

        rows.append(
            {
                "date": match.date,
                "home_team": home_team,
                "away_team": away_team,
                "h2h_home_winrate": _h2h_home_winrate(home_team, away_team, h2h_history),
                "home_form": home_stats["form5"],
                "away_form": away_stats["form5"],
                "home_avg_goals_scored": home_stats["avg_scored10"],
                "away_avg_goals_scored": away_stats["avg_scored10"],
                "home_avg_goals_conceded": home_stats["avg_conceded10"],
                "away_avg_goals_conceded": away_stats["avg_conceded10"],
                "home_advantage": 1,
                "result": int(match.result),
            }
        )

        home_points = _team_points_for_match(home_team, home_team, away_team, home_goals, away_goals)
        away_points = _team_points_for_match(away_team, home_team, away_team, home_goals, away_goals)
        team_history[home_team].append(
            {"goals_for": home_goals, "goals_against": away_goals, "points": home_points}
        )
        team_history[away_team].append(
            {"goals_for": away_goals, "goals_against": home_goals, "points": away_points}
        )

        key = tuple(sorted((home_team, away_team)))
        if home_goals > away_goals:
            winner = home_team
        elif away_goals > home_goals:
            winner = away_team
        else:
            winner = "draw"
        h2h_history[key].append(winner)

    return pd.DataFrame(rows)


def train_model(feature_df: pd.DataFrame):
    # Random-Forest-Modell trainieren und Qualität bewerten.
    X = feature_df[FEATURE_COLUMNS]
    y = feature_df["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Away win", "Draw", "Home win"],
        yticklabels=["Away win", "Draw", "Home win"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Vorhergesagt")
    plt.ylabel("Tatsächlich")
    plt.tight_layout()
    plt.show()

    return model, X_test, y_test, y_pred


def plot_feature_importance(model: RandomForestClassifier) -> None:
    # Wichtigste Features als Balkendiagramm anzeigen.
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    top10 = importances.head(10)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=top10.values, y=top10.index, hue=top10.index, legend=False)
    plt.title("Top 10 Feature-Importances")
    plt.xlabel("Wichtigkeit")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def _team_history_from_df(df: pd.DataFrame, team: str, last_n: int) -> pd.DataFrame:
    # Letzte N Spiele eines Teams aus der Matchhistorie holen.
    team_df = df[(df["home_team"] == team) | (df["away_team"] == team)].sort_values("date")
    return team_df.tail(last_n)


def _points_from_row(team: str, row: pd.Series) -> int:
    # Punkte eines Teams aus einer Matchzeile berechnen.
    return _team_points_for_match(
        team,
        str(row["home_team"]),
        str(row["away_team"]),
        int(row["home_goals"]),
        int(row["away_goals"]),
    )


def _avg_scored_conceded(team: str, team_matches: pd.DataFrame) -> tuple[float, float]:
    # Durchschnittlich geschossene und kassierte Tore berechnen.
    if team_matches.empty:
        return 0.0, 0.0

    scored: list[int] = []
    conceded: list[int] = []
    for _, row in team_matches.iterrows():
        home_goals = int(row["home_goals"])
        away_goals = int(row["away_goals"])
        if row["home_team"] == team:
            scored.append(home_goals)
            conceded.append(away_goals)
        else:
            scored.append(away_goals)
            conceded.append(home_goals)

    return float(sum(scored) / len(scored)), float(sum(conceded) / len(conceded))


def build_prediction_vector(home_team: str, away_team: str, df: pd.DataFrame) -> pd.DataFrame:
    # Featurevektor für ein neues Match aus vorhandener Historie erzeugen.
    home_last5 = _team_history_from_df(df, home_team, 5)
    away_last5 = _team_history_from_df(df, away_team, 5)
    home_last10 = _team_history_from_df(df, home_team, 10)
    away_last10 = _team_history_from_df(df, away_team, 10)

    home_form = float(sum(_points_from_row(home_team, row) for _, row in home_last5.iterrows()))
    away_form = float(sum(_points_from_row(away_team, row) for _, row in away_last5.iterrows()))

    home_avg_scored, home_avg_conceded = _avg_scored_conceded(home_team, home_last10)
    away_avg_scored, away_avg_conceded = _avg_scored_conceded(away_team, away_last10)

    h2h = df[
        ((df["home_team"] == home_team) & (df["away_team"] == away_team))
        | ((df["home_team"] == away_team) & (df["away_team"] == home_team))
    ].sort_values("date")

    h2h_recent = h2h.tail(10)
    if h2h_recent.empty:
        h2h_home_winrate = 0.5
    else:
        home_wins = 0
        for _, row in h2h_recent.iterrows():
            home_goals = int(row["home_goals"])
            away_goals = int(row["away_goals"])
            if (row["home_team"] == home_team and home_goals > away_goals) or (
                row["away_team"] == home_team and away_goals > home_goals
            ):
                home_wins += 1
        h2h_home_winrate = float(home_wins / len(h2h_recent))

    vector = pd.DataFrame(
        [
            {
                "h2h_home_winrate": h2h_home_winrate,
                "home_form": home_form,
                "away_form": away_form,
                "home_avg_goals_scored": home_avg_scored,
                "away_avg_goals_scored": away_avg_scored,
                "home_avg_goals_conceded": home_avg_conceded,
                "away_avg_goals_conceded": away_avg_conceded,
                "home_advantage": 1,
            }
        ],
        columns=FEATURE_COLUMNS,
    )
    return vector


def _prediction_text(label: int, home_team: str, away_team: str) -> str:
    # Textausgabe zum vorhergesagten Ergebnis.
    if label == 2:
        return f"{home_team} wins"
    if label == 1:
        return "Draw"
    return f"{away_team} wins"


def plot_prediction_probabilities(home_team: str, away_team: str, prob_map: dict[int, float]) -> None:
    # Wahrscheinlichkeiten der drei Ausgänge als Balkendiagramm anzeigen.
    labels = ["Home win", "Draw", "Away win"]
    values = [prob_map.get(2, 0.0) * 100, prob_map.get(1, 0.0) * 100, prob_map.get(0, 0.0) * 100]

    plt.figure(figsize=(7, 4))
    sns.barplot(x=labels, y=values, hue=labels, legend=False)
    plt.title(f"Ausgangswahrscheinlichkeiten: {home_team} vs {away_team}")
    plt.ylabel("Wahrscheinlichkeit (%)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()


def predict_match(home_team: str, away_team: str, df: pd.DataFrame, model: RandomForestClassifier) -> None:
    # Matchausgang vorhersagen und Wahrscheinlichkeiten ausgeben.
    vector = build_prediction_vector(home_team, away_team, df)
    probs = model.predict_proba(vector)[0]
    prob_map = {int(label): float(prob) for label, prob in zip(model.classes_, probs)}

    home_pct = prob_map.get(2, 0.0) * 100
    draw_pct = prob_map.get(1, 0.0) * 100
    away_pct = prob_map.get(0, 0.0) * 100

    predicted_label = max(prob_map, key=prob_map.get)
    pred_text = _prediction_text(predicted_label, home_team, away_team)

    print(f"\n⚽ {home_team} vs {away_team}")
    print(f"→ Home win:  {home_pct:5.1f}%")
    print(f"→ Draw:      {draw_pct:5.1f}%")
    print(f"→ Away win:  {away_pct:5.1f}%")
    print(f"→ Prediction: {pred_text}")

    plot_prediction_probabilities(home_team, away_team, prob_map)


def main() -> None:
    # Kompletter Ablauf: Daten holen, Features bauen, Modell trainieren, Vorhersage zeigen.
    if not API_KEY:
        raise RuntimeError(
            "Bitte API-Key setzen: FOOTBALL_DATA_API_TOKEN oder FOOTBALL_DATA_API_KEY."
        )

    print("Lade Live-Daten aus BL1, CL und PL ...")
    matches_df = fetch_all_leagues()
    print(f"Geladene Spiele: {len(matches_df)}")

    print("Baue Features ...")
    feature_df = build_features(matches_df)
    print(f"Feature-Zeilen: {len(feature_df)}")

    print("Trainiere Modell ...")
    model, _, _, _ = train_model(feature_df)

    print("Zeige Feature-Importances ...")
    plot_feature_importance(model)

    # Beispielvorhersage (anpassbar)
    predict_match("FC Bayern München", "Arsenal FC", matches_df, model)


if __name__ == "__main__":
    main()
