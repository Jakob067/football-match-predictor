from __future__ import annotations

import argparse
import requests
from flask import Flask, render_template_string, request

from live_service import next_predictions
from prediction_engine import COMPETITIONS, competition_choices

PAGE = r'''<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#07120e"><title>Matchday Live</title><style>
:root{--bg:#06100c;--panel:#10241a;--line:#29483a;--green:#4bf29a;--text:#f4fff8;--muted:#8fa99b}*{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(circle at 50% -15%,#17643c 0,transparent 38%),var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:28px 18px 70px}.nav{display:flex;align-items:center;justify-content:space-between;animation:down .55s ease}.logo{font-weight:900;font-size:1.15rem}.logo i{font-style:normal;background:var(--green);color:#062014;padding:9px;border-radius:12px;margin-right:9px}.live{display:flex;align-items:center;gap:7px;color:var(--green);font-size:.72rem;text-transform:uppercase;letter-spacing:.1em}.live:before{content:"";width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 12px var(--green);animation:pulse 1.8s infinite}.hero{padding:64px 0 34px;animation:up .65s ease}.hero h1{font-size:clamp(2.5rem,7vw,5rem);letter-spacing:-.06em;line-height:1;margin:0 0 18px}.hero h1 span{color:var(--green)}.hero p{color:var(--muted);max-width:620px;line-height:1.6}.toolbar{display:flex;gap:12px;align-items:end;margin-bottom:22px;animation:up .65s .08s both}.field{flex:1}.field label{display:block;color:var(--muted);font-size:.7rem;font-weight:700;text-transform:uppercase;margin-bottom:7px}.field select{width:100%;height:50px;background:#081810;color:var(--text);border:1px solid var(--line);border-radius:14px;padding:0 14px;font-size:.95rem;outline:none;transition:.2s}.field select:focus{border-color:var(--green);box-shadow:0 0 0 4px #4bf29a14}.refresh{height:50px;border:0;border-radius:14px;background:var(--green);color:#062014;font-weight:800;padding:0 22px;cursor:pointer;transition:.2s}.refresh:hover{transform:translateY(-2px);box-shadow:0 12px 30px #4bf29a30}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:15px}.match{position:relative;background:linear-gradient(145deg,#132c20,#0a1a12);border:1px solid var(--line);border-radius:20px;padding:20px;overflow:hidden;animation:reveal .55s both;transition:.25s}.match:hover{transform:translateY(-4px);border-color:#4bf29a66;box-shadow:0 18px 55px #0006}.match:after{content:"";position:absolute;width:130px;height:130px;border-radius:50%;background:#4bf29a0a;right:-50px;top:-55px}.time{display:flex;justify-content:space-between;color:var(--muted);font-size:.72rem;margin-bottom:20px}.teams{display:grid;grid-template-columns:1fr 36px 1fr;align-items:center;text-align:center;gap:8px}.team{min-width:0;font-weight:750}.team img{display:block;width:44px;height:44px;object-fit:contain;margin:0 auto 9px}.team span{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.vs{color:#648072;font-size:.72rem}.tip{text-align:center;margin:20px 0 15px;padding:12px;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}.tip small{display:block;color:var(--muted);font-size:.65rem;text-transform:uppercase;letter-spacing:.08em}.tip b{display:block;color:var(--green);font-size:1.1rem;margin-top:4px}.prob{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.prob div{background:#06150e;border-radius:10px;padding:8px;text-align:center}.prob b{display:block;font-size:.9rem}.prob span{color:var(--muted);font-size:.6rem}.meta{text-align:center;color:#6f8b7c;font-size:.62rem;margin-top:12px}.notice{background:#10241a;border:1px solid var(--line);border-radius:20px;padding:24px;line-height:1.6;color:var(--muted)}code{color:var(--green);background:#06150e;padding:4px 7px;border-radius:6px}.empty{text-align:center;padding:50px;color:var(--muted)}@keyframes up{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:none}}@keyframes down{from{opacity:0;transform:translateY(-10px)}to{opacity:1;transform:none}}@keyframes reveal{from{opacity:0;transform:translateY(20px) scale(.98)}to{opacity:1;transform:none}}@keyframes pulse{50%{opacity:.35}}@media(max-width:700px){.grid{grid-template-columns:1fr}.toolbar{align-items:stretch;flex-direction:column}.refresh{width:100%}.hero{padding:48px 0 28px}}@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style></head><body><main><nav class="nav"><div class="logo"><i>⚽</i>Matchday</div><div class="live">Live-Daten</div></nav><header class="hero"><h1>Die nächsten Spiele.<br><span>Sofort prognostiziert.</span></h1><p>Aktuelle Resultate aktualisieren automatisch Teamstärke und Form. Du musst nur noch den Wettbewerb wählen.</p></header><form class="toolbar"><div class="field"><label>Wettbewerb</label><select name="competition" onchange="this.form.submit()">{% for code,label in competitions %}<option value="{{code}}" {% if code==competition %}selected{% endif %}>{{label}}</option>{% endfor %}</select></div><button class="refresh">Aktualisieren ↻</button></form>
{% if error=='missing' %}<div class="notice"><b>Einmalige Live-Daten-Einrichtung</b><br>Setze deinen kostenlosen football-data.org Token und starte die Seite neu:<br><br><code>$env:FOOTBALL_DATA_API_TOKEN="DEIN_TOKEN"</code><br><code>py live_website.py</code><br><br>Danach erscheinen die nächsten Spiele automatisch. Es werden niemals erfundene Begegnungen angezeigt.</div>{% elif error %}<div class="notice">Live-Daten konnten gerade nicht geladen werden: {{error}}. Bitte später erneut versuchen oder API-Berechtigung für diesen Wettbewerb prüfen.</div>{% elif matches %}<section class="grid">{% for m in matches %}<article class="match" style="animation-delay:{{loop.index0 * 45}}ms"><div class="time"><span>{{m.day}}</span><b>{{m.kickoff}} Uhr</b></div><div class="teams"><div class="team">{% if m.home_crest %}<img src="{{m.home_crest}}" alt="">{% endif %}<span>{{m.home_name}}</span></div><div class="vs">VS</div><div class="team">{% if m.away_crest %}<img src="{{m.away_crest}}" alt="">{% endif %}<span>{{m.away_name}}</span></div></div><div class="tip"><small>Modell-Tipp · {{m.score}}</small><b>{{m.prediction}}</b></div><div class="prob"><div><b>{{m.home}}%</b><span>HEIM</span></div><div><b>{{m.draw}}%</b><span>REMIS</span></div><div><b>{{m.away}}%</b><span>AUSWÄRTS</span></div></div><div class="meta">xG {{m.home_xg}} : {{m.away_xg}} · Form {{m.home_form}} : {{m.away_form}}</div></article>{% endfor %}</section>{% else %}<div class="empty">In diesem Wettbewerb sind aktuell keine kommenden Spiele angesetzt.</div>{% endif %}</main></body></html>'''


def create_app(default_competition: str = "PL") -> Flask:
    app = Flask(__name__)
    app.config["DEFAULT_COMPETITION"] = default_competition if default_competition in COMPETITIONS else "PL"

    @app.get("/")
    def index() -> str:
        code = request.args.get("competition", app.config["DEFAULT_COMPETITION"])
        code = code if code in COMPETITIONS else "PL"
        matches, error = [], None
        try:
            matches = next_predictions(code)
        except RuntimeError as exc:
            error = "missing" if str(exc) == "API_TOKEN_MISSING" else str(exc)
        except requests.RequestException as exc:
            error = f"{exc.response.status_code} {exc.response.reason}" if exc.response is not None else "Netzwerkfehler"
        return render_template_string(PAGE, competitions=competition_choices(), competition=code, matches=matches, error=error)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatische Live-Fußballprognosen")
    parser.add_argument("--competition", default="PL", choices=COMPETITIONS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    create_app(args.competition).run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
