"""Single launcher for the complete portal with player and match details."""
from portal_details import app


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
