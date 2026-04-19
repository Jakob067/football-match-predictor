from __future__ import annotations

import argparse
import os
from functools import lru_cache

import pandas as pd
from flask import Flask, render_template_string, request
from requests import RequestException

from data_loader import (
    load_matches_from_api,
    load_team_stats_from_csv,
    load_upcoming_matches_from_api,
)
from features import build_feature_dataset, build_prediction_features
from modeling import train_model


PAGE_TEMPLATE = """
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Football Predictor</title>
  <style>
    :root {
      --bg: #0b1220;
      --panel: #121a2b;
      --panel-2: #182238;
      --line: #293651;
      --text: #f4f7ff;
      --muted: #a8b4d1;
      --accent: #00d084;
      --accent-2: #18a0fb;
      --warn: #3b2b11;
      --warn-line: #d9a441;
      --chip: #0e2f26;
      --chip-line: #1a5746;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif;
      background:
        radial-gradient(1200px 500px at 20% -10%, #12325e 0%, transparent 60%),
        radial-gradient(1000px 500px at 100% 0%, #124731 0%, transparent 55%),
        var(--bg);
      color: var(--text);
      min-height: 100vh;
    }
    .container { max-width: 1160px; margin: 0 auto; padding: 1.5rem; }
    .hero {
      background: linear-gradient(130deg, #10233f, #0e2d28);
      border: 1px solid #284265;
      border-radius: 18px;
      padding: 1.3rem 1.4rem;
      box-shadow: 0 22px 45px rgba(3, 8, 18, 0.4);
    }
    .hero h1 { margin: 0; font-size: 1.45rem; letter-spacing: 0.2px; }
    .hero p { margin: 0.45rem 0 0; color: #d3def6; }
    .row {
      margin-top: 1rem;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 0.85rem;
    }
    .card {
      background: linear-gradient(180deg, var(--panel), var(--panel-2));
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 1rem;
      box-shadow: 0 12px 24px rgba(5, 12, 24, 0.25);
    }
    .card h2 { margin: 0 0 0.75rem; font-size: 1rem; }
    .field { margin-bottom: 0.75rem; }
    .field:last-child { margin-bottom: 0; }
    label {
      display: block;
      margin-bottom: 0.33rem;
      font-size: 0.83rem;
      color: var(--muted);
      font-weight: 600;
    }
    input, select, button {
      width: 100%;
      border-radius: 10px;
      border: 1px solid #3a4a68;
      background: #0f1728;
      color: var(--text);
      padding: 0.58rem 0.65rem;
      font-size: 0.93rem;
    }
    input::placeholder { color: #7d8baa; }
    button {
      border: 0;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      color: #051322;
      font-weight: 700;
      cursor: pointer;
      margin-top: 0.2rem;
    }
    .hint {
      margin-top: 0.85rem;
      padding: 0.65rem 0.7rem;
      border-radius: 10px;
      border: 1px solid var(--chip-line);
      background: var(--chip);
      color: #b9f7de;
      font-size: 0.82rem;
    }
    .warning {
      margin-top: 1rem;
      border: 1px solid var(--warn-line);
      background: var(--warn);
      color: #ffd889;
      border-radius: 12px;
      padding: 0.75rem 0.85rem;
      font-size: 0.91rem;
    }
    .fixtures-grid {
      margin-top: 0.5rem;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
      gap: 0.7rem;
    }
    .fixture {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 0.65rem;
      background: #0f1728;
      display: block;
    }
    .fixture-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.6rem;
      margin-bottom: 0.3rem;
    }
    .fixture-time { color: var(--muted); font-size: 0.78rem; }
    .fixture-teams { font-weight: 600; line-height: 1.35; }
    .fixture input { width: auto; transform: scale(1.05); }
    .pred-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 0.2rem;
      font-size: 0.91rem;
    }
    .pred-table th, .pred-table td {
      border-bottom: 1px solid #2a3650;
      text-align: left;
      padding: 0.58rem 0.25rem;
      vertical-align: top;
    }
    .bar-wrap {
      position: relative;
      height: 8px;
      border-radius: 99px;
      background: #0e1524;
      border: 1px solid #2d3b56;
      overflow: hidden;
      min-width: 90px;
    }
    .bar {
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      border-radius: 99px;
      background: linear-gradient(90deg, #00d084, #18a0fb);
    }
    .small-note { color: var(--muted); font-size: 0.78rem; margin-top: 0.55rem; }
  </style>
</head>
<body>
  <main class="container">
    <section class="hero">
      <h1>Football Predictor - FotMob inspired</h1>
      <p>Schnelle Match-Auswahl, dunkles Live-Look-and-Feel und praxistaugliche ML-Modelle.</p>
    </section>

    {% if error_message %}
      <div class="warning">{{ error_message }}</div>
    {% endif %}

    <form method="post" class="row">
      <section class="card">
        <h2>Liga</h2>
        <div class="field">
          <label for="competition">Competition</label>
          <input id="competition" name="competition" value="{{ competition }}" />
        </div>
        <div class="field">
          <label for="season">Season (z. B. 2025)</label>
          <input id="season" name="season" value="{{ season_text }}" />
        </div>
        <div class="hint">API-Token wird global geladen: --api-token oder FOOTBALL_DATA_API_TOKEN / FOOTBALL_DATA_API_KEY.</div>
      </section>

      <section class="card">
        <h2>Accuracy Setup</h2>
        <div class="field">
          <label for="history_seasons">Historische Saisons</label>
          <select id="history_seasons" name="history_seasons">
            {% for season_count in [1, 3, 5, 7] %}
              <option value="{{ season_count }}" {% if history_seasons == season_count %}selected{% endif %}>{{ season_count }}</option>
            {% endfor %}
          </select>
        </div>
        <div class="field">
          <label for="model_type">Modell</label>
          <select id="model_type" name="model_type">
            {% for value, label in model_choices %}
              <option value="{{ value }}" {% if model_type == value %}selected{% endif %}>{{ label }}</option>
            {% endfor %}
          </select>
        </div>
        <div class="field">
          <label for="team_stats_csv_path">Team-Stats CSV (xG, Ballbesitz, Schuesse, Ausfaelle)</label>
          <input id="team_stats_csv_path" name="team_stats_csv_path" value="{{ team_stats_csv_path }}" placeholder="./data/team_stats.csv" />
        </div>
      </section>

      <section class="card">
        <h2>Spiele</h2>
        <div class="field">
          <label for="days">Tage im Voraus</label>
          <select id="days" name="days">
            {% for day in [2, 3] %}
              <option value="{{ day }}" {% if days_ahead == day %}selected{% endif %}>{{ day }}</option>
            {% endfor %}
          </select>
        </div>
        <button type="submit">Vorhersagen erstellen</button>
        <div class="small-note">Fuer 5+ Saisons bitte "Season" setzen.</div>
      </section>

      <section class="card" style="grid-column: 1 / -1;">
        <h2>Verfuegbare Fixtures</h2>
        {% if fixtures %}
          <div class="fixtures-grid">
            {% for fixture in fixtures %}
              <label class="fixture">
                <div class="fixture-head">
                  <span class="fixture-time">{{ fixture.date.strftime("%Y-%m-%d %H:%M") }} UTC</span>
                  <input
                    type="radio"
                    name="match_id"
                    value="{{ fixture.match_id }}"
                    {% if fixture.match_id|string == selected_match_id %}checked{% endif %}
                  />
                </div>
                <div class="fixture-teams">{{ fixture.home_team }}<br>vs<br>{{ fixture.away_team }}</div>
              </label>
            {% endfor %}
          </div>
        {% else %}
          <div class="small-note">Keine geplanten Spiele im gewaehlten Zeitraum gefunden.</div>
        {% endif %}
      </section>
    </form>

    {% if predictions %}
      <section class="card" style="margin-top: 1rem;">
        <h2>Vorhersagen</h2>
        <table class="pred-table">
          <thead>
            <tr>
              <th>Spiel</th>
              <th>Tipp</th>
              <th>Home</th>
              <th>Draw</th>
              <th>Away</th>
            </tr>
          </thead>
          <tbody>
            {% for row in predictions %}
              <tr>
                <td>{{ row.home_team }} vs {{ row.away_team }}</td>
                <td>{{ row.prediction }}</td>
                <td>
                  {{ row.home_win_pct }}%
                  <div class="bar-wrap"><div class="bar" style="width: {{ row.home_win_pct }}%;"></div></div>
                </td>
                <td>
                  {{ row.draw_pct }}%
                  <div class="bar-wrap"><div class="bar" style="width: {{ row.draw_pct }}%;"></div></div>
                </td>
                <td>
                  {{ row.away_win_pct }}%
                  <div class="bar-wrap"><div class="bar" style="width: {{ row.away_win_pct }}%;"></div></div>
                </td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      </section>
    {% endif %}
  </main>
</body>
</html>
"""


EMPTY_TEAM_STATS_DF = pd.DataFrame(columns=["team", "xg", "possession", "shots", "unavailable_players"])
GLOBAL_API_TOKEN = os.getenv("FOOTBALL_DATA_API_TOKEN") or os.getenv("FOOTBALL_DATA_API_KEY")
MODEL_CHOICES = [
    ("random_forest", "Random Forest"),
    ("xgboost", "XGBoost"),
    ("neural_net", "Neural Net (MLP)"),
]


def _resolve_api_token(cli_token: str | None) -> str | None:
    for candidate in [
        cli_token,
        GLOBAL_API_TOKEN,
    ]:
        if candidate and candidate.strip():
            return candidate.strip()
    return None


def _parse_season(raw_value: str) -> int | None:
    text = raw_value.strip()
    if not text:
        return None
    return int(text)


@lru_cache(maxsize=16)
def _load_team_stats(team_stats_csv_path: str) -> pd.DataFrame:
    if not team_stats_csv_path:
        return EMPTY_TEAM_STATS_DF.copy()
    return load_team_stats_from_csv(team_stats_csv_path)


@lru_cache(maxsize=12)
def _load_trained_model(
    api_token: str,
    competition_code: str,
    season: int | None,
    history_seasons: int,
    model_type: str,
    team_stats_csv_path: str,
):
    historical_matches = load_matches_from_api(
        api_token=api_token,
        competition_code=competition_code,
        season=season,
        seasons_back=history_seasons,
    )
    team_stats_df = _load_team_stats(team_stats_csv_path)
    feature_df = build_feature_dataset(historical_matches, team_stats_df=team_stats_df)
    training_result = train_model(feature_df, model_type=model_type)
    return training_result["model"], historical_matches, team_stats_df


def _prediction_from_model(
    model,
    historical_matches: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    home_team: str,
    away_team: str,
) -> dict[str, str]:
    feature_vector = build_prediction_features(
        team_a=home_team,
        team_b=away_team,
        matches_df=historical_matches,
        team_stats_df=team_stats_df,
        team_a_is_home=True,
    )
    probabilities = model.predict_proba(feature_vector)[0]
    probability_by_class = {
        int(label): float(prob * 100.0) for label, prob in zip(model.classes_, probabilities)
    }

    home_win_pct = round(probability_by_class.get(0, 0.0), 1)
    draw_pct = round(probability_by_class.get(1, 0.0), 1)
    away_win_pct = round(probability_by_class.get(2, 0.0), 1)

    prediction_class = max(probability_by_class, key=probability_by_class.get)
    if prediction_class == 0:
        prediction = f"{home_team} gewinnt"
    elif prediction_class == 1:
        prediction = "Unentschieden"
    else:
        prediction = f"{away_team} gewinnt"

    return {
        "home_team": home_team,
        "away_team": away_team,
        "prediction": prediction,
        "home_win_pct": f"{home_win_pct:.1f}",
        "draw_pct": f"{draw_pct:.1f}",
        "away_win_pct": f"{away_win_pct:.1f}",
    }


def create_app(
    default_api_token: str | None = GLOBAL_API_TOKEN,
    default_competition: str = "PL",
    default_season: int | None = None,
    default_days_ahead: int = 3,
    default_history_seasons: int = 5,
    default_model_type: str = "random_forest",
    default_team_stats_csv_path: str = "",
) -> Flask:
    app = Flask(__name__)
    app.config["DEFAULT_API_TOKEN"] = default_api_token
    app.config["DEFAULT_COMPETITION"] = default_competition
    app.config["DEFAULT_SEASON"] = default_season
    app.config["DEFAULT_DAYS_AHEAD"] = default_days_ahead
    app.config["DEFAULT_HISTORY_SEASONS"] = default_history_seasons
    app.config["DEFAULT_MODEL_TYPE"] = default_model_type
    app.config["DEFAULT_TEAM_STATS_CSV_PATH"] = default_team_stats_csv_path

    @app.route("/", methods=["GET", "POST"])
    def index() -> str:
        competition = request.values.get("competition", app.config["DEFAULT_COMPETITION"]).strip() or "PL"
        season_text = request.values.get(
            "season",
            "" if app.config["DEFAULT_SEASON"] is None else str(app.config["DEFAULT_SEASON"]),
        )
        days_raw = request.values.get("days", str(app.config["DEFAULT_DAYS_AHEAD"]))
        history_raw = request.values.get("history_seasons", str(app.config["DEFAULT_HISTORY_SEASONS"]))
        model_type = request.values.get("model_type", app.config["DEFAULT_MODEL_TYPE"]).strip() or "random_forest"
        team_stats_csv_path = request.values.get(
            "team_stats_csv_path",
            app.config["DEFAULT_TEAM_STATS_CSV_PATH"],
        ).strip()

        fixtures: list[dict[str, object]] = []
        selected_match_id = request.form.get("match_id", "").strip()
        selected_ids = set(request.form.getlist("match_ids"))
        if not selected_match_id and selected_ids:
            selected_match_id = sorted(selected_ids)[0]
        predictions: list[dict[str, str]] = []
        error_message: str | None = None

        try:
            season = _parse_season(season_text)
            days_ahead = int(days_raw)
            history_seasons = int(history_raw)
            if history_seasons < 1:
                raise ValueError("Historische Saisons muessen mindestens 1 sein.")
            if season is None and history_seasons > 1:
                raise ValueError("Bitte Season setzen, wenn historische Saisons > 1 genutzt werden.")
        except ValueError as exc:
            return render_template_string(
                PAGE_TEMPLATE,
                competition=competition,
                season_text=season_text,
                days_ahead=app.config["DEFAULT_DAYS_AHEAD"],
                history_seasons=app.config["DEFAULT_HISTORY_SEASONS"],
                model_type=model_type,
                model_choices=MODEL_CHOICES,
                team_stats_csv_path=team_stats_csv_path,
                fixtures=[],
                selected_match_id=selected_match_id,
                predictions=[],
                error_message=str(exc),
            )

        token = _resolve_api_token(app.config["DEFAULT_API_TOKEN"])
        if token is None:
            error_message = (
                "Kein API Token gefunden. Bitte in web_app.py GLOBAL_API_TOKEN setzen "
                "oder --api-token / FOOTBALL_DATA_API_TOKEN / FOOTBALL_DATA_API_KEY nutzen."
            )
        else:
            try:
                upcoming_matches = load_upcoming_matches_from_api(
                    api_token=token,
                    competition_code=competition,
                    season=season,
                    days_ahead=days_ahead,
                )
                fixtures = upcoming_matches.to_dict(orient="records")
            except RequestException as exc:
                error_message = f"API-Fehler beim Laden der Fixtures: {exc}"

            if request.method == "POST" and error_message is None:
                if not selected_match_id:
                    error_message = "Bitte ein Spiel auswaehlen."
                else:
                    try:
                        model, historical_matches, team_stats_df = _load_trained_model(
                            token,
                            competition,
                            season,
                            history_seasons,
                            model_type,
                            team_stats_csv_path,
                        )
                        selected_df = upcoming_matches[
                            upcoming_matches["match_id"].astype(str) == selected_match_id
                        ].sort_values("date")
                        predictions = [
                            _prediction_from_model(
                                model=model,
                                historical_matches=historical_matches,
                                team_stats_df=team_stats_df,
                                home_team=str(row.home_team),
                                away_team=str(row.away_team),
                            )
                            for row in selected_df.itertuples(index=False)
                        ]
                        if not predictions:
                            error_message = "Das gewaehlte Spiel wurde nicht gefunden."
                    except (RequestException, ValueError, FileNotFoundError, ImportError) as exc:
                        error_message = str(exc)

        return render_template_string(
            PAGE_TEMPLATE,
            competition=competition,
            season_text=season_text,
            days_ahead=days_ahead,
            history_seasons=history_seasons,
            model_type=model_type,
            model_choices=MODEL_CHOICES,
            team_stats_csv_path=team_stats_csv_path,
            fixtures=fixtures,
            selected_match_id=selected_match_id,
            predictions=predictions,
            error_message=error_message,
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weboberflaeche fuer Spielauswahl und Vorhersagen")
    parser.add_argument("--api-token", type=str, help="football-data.org API Token")
    parser.add_argument("--competition", type=str, default="PL", help="Wettbewerbscode, z. B. PL")
    parser.add_argument("--season", type=int, help="Saisonjahr, z. B. 2025")
    parser.add_argument("--days-ahead", type=int, default=3, choices=[2, 3], help="Spielauswahl in 2 oder 3 Tagen")
    parser.add_argument("--history-seasons", type=int, default=5, help="Anzahl historischer Saisons fuer Training")
    parser.add_argument(
        "--model",
        choices=["random_forest", "xgboost", "neural_net"],
        default="random_forest",
        help="Modelltyp fuer Vorhersagen",
    )
    parser.add_argument(
        "--team-stats-csv-path",
        type=str,
        default="",
        help="Optionaler CSV-Pfad mit xG/Ballbesitz/Schuesse/Ausfaelle pro Team",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host fuer den Webserver")
    parser.add_argument("--port", type=int, default=8000, help="Port fuer den Webserver")
    parser.add_argument("--debug", action="store_true", help="Flask Debug Modus aktivieren")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        default_api_token=args.api_token or GLOBAL_API_TOKEN,
        default_competition=args.competition,
        default_season=args.season,
        default_days_ahead=args.days_ahead,
        default_history_seasons=args.history_seasons,
        default_model_type=args.model,
        default_team_stats_csv_path=args.team_stats_csv_path,
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
