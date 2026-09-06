from __future__ import annotations

import portal_players

portal = portal_players.portal
# Native disclosure controls and the shared template provide match analysis.
app = portal.create_app(details_enabled=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
