"""World Cup live site using the configured standings and scheduled-match URLs."""
from __future__ import annotations

import live_website
from wc_live_service import next_world_cup_predictions


def _world_cup_predictions(_competition: str) -> list[dict[str, object]]:
    return next_world_cup_predictions()


live_website.next_predictions = _world_cup_predictions
app = live_website.create_app("WC")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
