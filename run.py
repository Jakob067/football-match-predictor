"""Stable entry point for the complete Matchday portal."""
from portal_players import app


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
