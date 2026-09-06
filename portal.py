from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests
from flask import Flask, render_template_string, request

from live_service import next_predictions
from prediction_engine import COMPETITIONS, competition_choices
from wc_live_service import next_world_cup_predictions


def _load_env_token() -> None:
    """Load the API token from the project's local .env file."""
    if os.getenv("FOOTBALL_DATA_API_TOKEN"):
        return
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    try:
        with open(env_path, encoding="utf-8") as env_file:
            for line in env_file:
                key, separator, value = line.strip().partition("=")
                if separator and key == "FOOTBALL_DATA_API_TOKEN":
                    os.environ[key] = value.strip().strip('"').strip("'")
                    return
    except FileNotFoundError:
        pass


_load_env_token()

PAGE = (Path(__file__).parent / 'templates' / 'portal.html').read_text(encoding='utf-8')


def _fetch(code: str) -> list[dict[str, object]]:
    return next_world_cup_predictions() if code == "WC" else next_predictions(code)


def create_app(default_competition: str = "PL", *, details_enabled: bool = False) -> Flask:
    app = Flask(__name__)
    app.config["DEFAULT_COMPETITION"] = default_competition if default_competition in COMPETITIONS else "PL"

    @app.get("/")
    def index() -> str:
        view = request.args.get("view", "home")
        view = view if view in {"home", "matches", "predictions"} else "home"
        code = request.args.get("competition", app.config["DEFAULT_COMPETITION"])
        code = code if code in COMPETITIONS else "PL"
        matches: list[dict[str, object]] = []
        error: str | None = None
        if view != "home":
            try:
                matches = _fetch(code)
            except RuntimeError as exc:
                error = "missing" if str(exc) == "API_TOKEN_MISSING" else str(exc)
            except requests.RequestException as exc:
                error = f"HTTP {exc.response.status_code}" if exc.response is not None else "Netzwerkfehler"
        return render_template_string(PAGE, view=view, competition=code,
            competitions=competition_choices(), competition_name=COMPETITIONS[code][0], matches=matches, error=error, details_enabled=details_enabled)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Modernes Fußball-Liveportal")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    create_app().run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
