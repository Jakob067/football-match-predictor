from __future__ import annotations

import portal
from scorer_service import add_scorer_predictions

_original_fetch = portal._fetch


def _fetch_with_players(code: str) -> list[dict[str, object]]:
    return add_scorer_predictions(_original_fetch(code), code)


portal._fetch = _fetch_with_players
# Player rendering is part of the shared template, without fragile HTML replacements.
app = portal.create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
