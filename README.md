# football-match-predictor

Ein ML-gestütztes System zur Vorhersage von Fußballspielergebnissen. Unterstützt CSV-Dateien und die football-data.org API als Datenquellen, drei Modelltypen sowie eine Web-Oberfläche für interaktive Prognosen.

---

## Funktionsübersicht

- Vorhersage von Spielausgängen (Heimsieg / Unentschieden / Auswärtssieg) mit Wahrscheinlichkeiten
- Schlüsselspieler-Prognose basierend auf Tor- und Assist-Statistiken
- Live-Vorhersagen für kommende Spieltage via API
- Visualisierungen: Feature-Wichtigkeit und Confusion Matrix
- Drei Modelltypen: Random Forest, Neural Network, XGBoost
- Web-Interface (Flask) für browserbasierte Nutzung

---

## Projektstruktur

```
football-match-predictor/
├── main.py                 # CLI-Einstiegspunkt und Pipeline-Orchestrierung
├── data_loader.py          # CSV- und API-Datenlader (football-data.org)
├── features.py             # Feature Engineering und Datensatz-Aufbau
├── modeling.py             # Modelltraining (Random Forest, Neural Net, XGBoost)
├── predictor.py            # Vorhersage-Interface mit Schlüsselspieler-Heuristik
├── live_api_predictor.py   # Live-Vorhersagen für anstehende Spieltage
├── visualization.py        # Feature-Importance-Plot und Confusion-Matrix-Heatmap
├── web_app.py              # Flask-Webanwendung
└── requirements.txt        # Python-Abhängigkeiten
```

### Modulübersicht

| Modul | Beschreibung |
|---|---|
| `data_loader.py` | Lädt Spieldaten aus CSV oder der football-data.org API. Unterstützt flexible Spaltennamen via Mapping und mehrere Saisons gleichzeitig. |
| `features.py` | Baut aus der Spielhistorie einen Trainingsdatensatz ohne Datenleck. Berechnet Form, Head-to-Head-Quote, Expected Goals, Ballbesitz, Schüsse und Ausfälle. |
| `modeling.py` | Trainiert einen der drei Klassifikatoren auf einem 80/20-Split mit Stratifizierung. Gibt Accuracy, Confusion Matrix und das fertige Modell zurück. |
| `predictor.py` | Hält das trainierte Modell im Speicher und berechnet für ein beliebiges Duell Ausgangwahrscheinlichkeiten sowie den wahrscheinlichsten Schlüsselspieler. |
| `live_api_predictor.py` | Ruft kommende Spiele aus der API ab und erstellt automatisch Vorhersagen für alle Begegnungen innerhalb eines konfigurierbaren Zeitfensters. |
| `visualization.py` | Erzeugt einen Feature-Importance-Barplot und eine Confusion-Matrix-Heatmap als PNG-Dateien. |
| `web_app.py` | Flask-App für browserbasierte Eingabe von Team A vs. Team B mit direkter Anzeige der Prognose. |

---

## Features (ML-Eingaben)

Das Modell nutzt 18 Features pro Spiel:

| Feature | Beschreibung |
|---|---|
| `h2h_win_rate_last10` | Siegquote Team A in den letzten 10 direkten Duellen |
| `team_a/b_form_points_last5` | Punkte aus den letzten 5 Spielen |
| `team_a/b_avg_goals_for/against` | Durchschnittliche Tore pro Spiel (gesamt) |
| `team_a/b_avg_goal_diff` | Durchschnittliche Tordifferenz |
| `home_advantage` | 1 = Team A spielt Heimspiel, 0 = Auswärtsspiel |
| `team_a/b_xg` | Expected Goals (aus Team-Stats-CSV oder API) |
| `team_a/b_possession` | Ballbesitzanteil in % |
| `team_a/b_shots` | Schüsse pro Spiel |
| `team_a/b_unavailable_players` | Anzahl verletzter / gesperrter Spieler |

Jedes Spiel wird aus **beiden Perspektiven** als Trainingsbeispiel aufgenommen (Team A = Heim, Team A = Auswärts), um die Datenbasis zu verdoppeln.

---

## Datenquellen

### CSV

```bash
python main.py --source csv --csv-path matches.csv
```

Pflichtfelder (flexible Spaltennamen, z. B. `FTHG` oder `home_goals`): Datum, Heimteam, Auswärtsteam, Tore Heim, Tore Auswärts.

Optionale Zusatz-CSVs:

```bash
--player-csv-path players.csv         # Spielerstatistiken (Tore, Assists)
--team-stats-csv-path team_stats.csv  # xG, Ballbesitz, Schüsse, Ausfälle
```

### football-data.org API

```bash
python main.py --source api --api-token DEIN_TOKEN --competition PL --season 2024 --history-seasons 5
```

Der Token kann alternativ als Umgebungsvariable `FOOTBALL_DATA_API_TOKEN` gesetzt werden. Mit `--history-seasons` lassen sich mehrere Saisons für ein robusteres Training kombinieren.

---

## CLI-Nutzung

```bash
# Training + Vorhersage für ein konkretes Duell
python main.py \
  --source csv \
  --csv-path matches.csv \
  --model random_forest \
  --team-a "Arsenal FC" \
  --team-b "Chelsea FC" \
  --output-dir outputs/

# XGBoost-Modell mit API-Daten und automatischen Tests
python main.py \
  --source api \
  --api-token DEIN_TOKEN \
  --model xgboost \
  --auto-test
```

### Alle CLI-Optionen

| Flag | Standard | Beschreibung |
|---|---|---|
| `--source` | `csv` | Datenquelle: `csv` oder `api` |
| `--csv-path` | – | Pfad zur Spiel-CSV (bei `--source csv`) |
| `--api-token` | – | API-Token für football-data.org |
| `--competition` | `PL` | Wettbewerbscode (z. B. `BL1`, `SA`, `PD`) |
| `--season` | – | Saisonjahr (z. B. `2024`) |
| `--history-seasons` | `5` | Anzahl Saisons für Training (API) |
| `--model` | `random_forest` | `random_forest`, `xgboost`, `neural_net` |
| `--team-a` / `--team-b` | – | Beispielvorhersage nach Training |
| `--output-dir` | `outputs/` | Zielordner für Plots |
| `--auto-test` | `false` | Führt Tests aus `tests/` vor dem Start aus |

---

## Ausgaben

Nach einem Lauf befinden sich im `--output-dir`:

```
outputs/
├── feature_importance.png        # Balkendiagramm der Feature-Wichtigkeiten
└── confusion_matrix_heatmap.png  # Heatmap der Klassifikationsgüte
```

Konsolenausgabe (Beispiel):

```
Accuracy: 0.5312

Confusion Matrix:
                  Team A gewinnt  Unentschieden  Team B gewinnt
Team A gewinnt               210             48              62
Unentschieden                 71             89              74
Team B gewinnt                55             43             198

Vorhersage: Team A gewinnt (61.3% Wahrscheinlichkeit) | Schlüsselspieler: Erling Haaland (Manchester City)
```

---

## Modelle

| Modell | Stärken | Hinweis |
|---|---|---|
| `random_forest` | Robust, interpretierbar, keine Zusatzinstallation | Standardwahl |
| `neural_net` | Flexible nichtlineare Zusammenhänge | Längere Trainingszeit |
| `xgboost` | Häufig höchste Genauigkeit bei tabellarischen Daten | Benötigt `pip install xgboost` |

---

## Wettbewerbscodes (football-data.org)

| Code | Liga |
|---|---|
| `PL` | Premier League |
| `BL1` | Bundesliga |
| `SA` | Serie A |
| `PD` | La Liga |
| `FL1` | Ligue 1 |
| `CL` | UEFA Champions League |

---

## Lizenz

MIT License – siehe [LICENSE](LICENSE) für Details.
