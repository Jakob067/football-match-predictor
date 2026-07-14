from __future__ import annotations

import argparse
import os

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

PAGE = r'''<!doctype html><html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#07142f"><title>Matchday Blue</title><style>
:root{--bg:#030916;--bg2:#07142f;--card:rgba(10,25,55,.72);--card2:#0b1a38;--line:rgba(125,174,255,.18);--blue:#4e8cff;--cyan:#53d8ff;--text:#f5f8ff;--muted:#8ea5ca;--nav:72px}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;min-height:100vh;background:var(--bg);color:var(--text);font-family:Inter,"Segoe UI",system-ui,sans-serif;overflow-x:hidden}.aurora{position:fixed;inset:0;pointer-events:none;background:radial-gradient(700px 500px at 15% -5%,#174cbd55,transparent 65%),radial-gradient(600px 500px at 95% 40%,#00a9e933,transparent 62%),linear-gradient(180deg,var(--bg2),var(--bg));z-index:-2}.grid-bg{position:fixed;inset:0;pointer-events:none;z-index:-1;opacity:.22;background-image:linear-gradient(#74a5ff11 1px,transparent 1px),linear-gradient(90deg,#74a5ff11 1px,transparent 1px);background-size:52px 52px;mask-image:linear-gradient(to bottom,#000,transparent 80%)}.blob{position:fixed;width:420px;height:420px;border-radius:50%;filter:blur(110px);background:#176bff30;left:35%;top:15%;z-index:-1;animation:drift 14s ease-in-out infinite}.nav{position:sticky;top:0;z-index:20;height:var(--nav);display:flex;align-items:center;justify-content:space-between;padding:0 max(22px,calc((100vw - 1180px)/2));background:#030a18aa;border-bottom:1px solid var(--line);backdrop-filter:blur(22px)}.brand{display:flex;align-items:center;gap:11px;font-weight:850;letter-spacing:-.03em}.brand-icon{display:grid;place-items:center;width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,var(--blue),var(--cyan));box-shadow:0 8px 30px #438cff50}.links{display:flex;gap:8px}.links a{color:var(--muted);text-decoration:none;font-size:.85rem;font-weight:700;padding:10px 14px;border-radius:11px;transition:.25s}.links a:hover,.links a.active{color:white;background:#4c8cff18}.status{display:flex;align-items:center;gap:7px;color:#69dcff;font-size:.68rem;text-transform:uppercase;letter-spacing:.1em}.status i{width:7px;height:7px;border-radius:50%;background:#53d8ff;box-shadow:0 0 14px #53d8ff;animation:pulse 1.8s infinite}.shell{max-width:1180px;margin:auto;padding:0 22px 80px}.hero{min-height:calc(88vh - var(--nav));display:grid;place-items:center;text-align:center;padding:60px 0}.hero-inner{animation:heroIn .9s cubic-bezier(.2,.8,.2,1) both}.kicker{display:inline-flex;align-items:center;gap:8px;color:#8ec3ff;font-size:.73rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;border:1px solid var(--line);background:#0d244c88;padding:8px 13px;border-radius:999px}.kicker:before{content:"";width:6px;height:6px;border-radius:50%;background:var(--cyan);box-shadow:0 0 10px var(--cyan)}h1{font-size:clamp(3rem,8vw,6.8rem);line-height:.92;letter-spacing:-.075em;margin:22px 0}.shine{background:linear-gradient(110deg,#fff 10%,#79adff 45%,#67e3ff 70%,#fff);background-size:200%;background-clip:text;color:transparent;animation:shine 6s linear infinite}.lead{max-width:670px;color:var(--muted);font-size:clamp(.95rem,2vw,1.12rem);line-height:1.7;margin:0 auto 30px}.actions{display:flex;justify-content:center;gap:12px;flex-wrap:wrap}.btn{display:inline-flex;align-items:center;justify-content:center;gap:9px;height:52px;padding:0 22px;border-radius:14px;text-decoration:none;font-weight:800;font-size:.9rem;transition:.25s;border:1px solid var(--line)}.btn-primary{background:linear-gradient(110deg,var(--blue),#3cbfff);color:white;box-shadow:0 15px 45px #327cff38}.btn-secondary{background:#0c1d3caa;color:#bad0f5}.btn:hover{transform:translateY(-3px);box-shadow:0 18px 48px #327cff50}.feature-row{display:grid;grid-template-columns:repeat(3,1fr);gap:13px;margin-top:-55px;padding-bottom:60px}.feature{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:20px;backdrop-filter:blur(18px);animation:rise .65s both}.feature:nth-child(2){animation-delay:.08s}.feature:nth-child(3){animation-delay:.16s}.feature b{display:block;font-size:.95rem;margin:9px 0 5px}.feature span{color:var(--muted);font-size:.78rem;line-height:1.5}.feature-icon{font-size:1.25rem}.page-head{padding:58px 0 30px;animation:rise .65s both}.page-head h1{font-size:clamp(2.4rem,6vw,4.5rem);margin:12px 0}.page-head p{color:var(--muted)}.toolbar{display:flex;align-items:end;gap:12px;padding:16px;background:var(--card);border:1px solid var(--line);border-radius:18px;margin-bottom:18px;backdrop-filter:blur(18px);animation:rise .65s .06s both}.field{flex:1}.field label{display:block;color:var(--muted);font-size:.65rem;font-weight:800;text-transform:uppercase;letter-spacing:.08em;margin-bottom:7px}.field select{width:100%;height:48px;border:1px solid var(--line);border-radius:12px;background:#07152e;color:white;padding:0 14px;font-size:.9rem;outline:none;transition:.2s}.field select:focus{border-color:#528fff;box-shadow:0 0 0 4px #528fff18}.refresh{height:48px;padding:0 20px;border:0;border-radius:12px;background:linear-gradient(120deg,var(--blue),#48c5ff);color:white;font-weight:800;cursor:pointer;transition:.2s}.refresh:hover{transform:translateY(-2px)}.matches{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}.match{position:relative;background:linear-gradient(145deg,#0d2146dd,#08152ddd);border:1px solid var(--line);border-radius:20px;padding:20px;overflow:hidden;animation:cardIn .55s both;transition:.27s}.match:before{content:"";position:absolute;width:130px;height:130px;border-radius:50%;right:-60px;top:-65px;background:#51a0ff12}.match:hover{transform:translateY(-5px);border-color:#65a2ff70;box-shadow:0 22px 60px #0008}.match-head{display:flex;justify-content:space-between;color:var(--muted);font-size:.7rem;margin-bottom:20px}.match-head b{color:#b6ceef}.teams{display:grid;grid-template-columns:1fr 34px 1fr;align-items:center;gap:6px;text-align:center}.team{min-width:0;font-weight:800;font-size:.88rem}.team img{display:block;width:45px;height:45px;object-fit:contain;margin:0 auto 9px;filter:drop-shadow(0 8px 12px #0007)}.team span{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.versus{color:#607ca8;font-size:.66rem}.prediction{margin:18px 0 13px;padding:12px;text-align:center;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}.prediction small{display:block;color:var(--muted);font-size:.62rem;text-transform:uppercase;letter-spacing:.09em}.prediction b{display:block;color:#70c9ff;margin-top:4px}.odds{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.odd{text-align:center;background:#06132a;border:1px solid #6d9cff0d;border-radius:10px;padding:8px}.odd b{display:block;font-size:.9rem}.odd span{font-size:.55rem;color:var(--muted)}.formline{height:4px;border-radius:99px;background:#142a4d;margin-top:6px;overflow:hidden}.formline i{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--cyan));width:var(--w);animation:grow 1s .25s both}.technical{color:#6681aa;text-align:center;font-size:.6rem;margin-top:11px}.notice,.empty{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:28px;color:var(--muted);line-height:1.65}.notice code{color:#75cfff;background:#06132a;padding:4px 7px;border-radius:6px}.empty{text-align:center;padding:60px}.footer{color:#526b91;text-align:center;font-size:.68rem;margin-top:35px}@keyframes heroIn{from{opacity:0;transform:translateY(30px) scale(.98)}to{opacity:1;transform:none}}@keyframes rise{from{opacity:0;transform:translateY(22px)}to{opacity:1;transform:none}}@keyframes cardIn{from{opacity:0;transform:translateY(16px) scale(.98)}to{opacity:1;transform:none}}@keyframes drift{50%{transform:translate(70px,45px) scale(1.15)}}@keyframes shine{to{background-position:-200% center}}@keyframes pulse{50%{opacity:.3}}@keyframes grow{from{width:0}}@media(max-width:720px){.nav{padding:0 14px}.status{display:none}.links a{padding:9px;font-size:.75rem}.brand span:last-child{display:none}.shell{padding:0 14px 55px}.hero{min-height:75vh}.feature-row{grid-template-columns:1fr;margin-top:0}.matches{grid-template-columns:1fr}.toolbar{align-items:stretch;flex-direction:column}.refresh{width:100%}}@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important;scroll-behavior:auto!important}}
</style></head><body><div class="aurora"></div><div class="grid-bg"></div><div class="blob"></div><nav class="nav"><a class="brand" href="/?view=home" style="color:inherit;text-decoration:none"><span class="brand-icon">⚽</span><span>Matchday Blue</span></a><div class="links"><a href="/?view=home" class="{{'active' if view=='home'}}">Start</a><a href="/?view=matches" class="{{'active' if view=='matches'}}">Nächste Spiele</a><a href="/?view=predictions" class="{{'active' if view=='predictions'}}">Prognosen</a></div><div class="status"><i></i>API verbunden</div></nav><main class="shell">
{% if view=='home' %}<section class="hero"><div class="hero-inner"><span class="kicker">Automatische Live-Prognosen</span><h1>Fußball.<br><span class="shine">Neu berechnet.</span></h1><p class="lead">Alle kommenden Top-Spiele auf einen Blick. Aktuelle Resultate, Form, Teamstärke und Torwahrscheinlichkeiten werden automatisch verarbeitet.</p><div class="actions"><a class="btn btn-primary" href="/?view=matches">Nächste Spiele ansehen <span>→</span></a><a class="btn btn-secondary" href="/?view=predictions">Prognosen öffnen</a></div></div></section><section class="feature-row"><article class="feature"><div class="feature-icon">◉</div><b>Live-Spielplan</b><span>Echte kommende Begegnungen direkt aus der Fußball-API.</span></article><article class="feature"><div class="feature-icon">⌁</div><b>Dynamische Stärke</b><span>Elo, Form, Ergebnisse und Tordifferenz werden laufend gewichtet.</span></article><article class="feature"><div class="feature-icon">✦</div><b>Klare Prognosen</b><span>1-X-2-Wahrscheinlichkeit, xG und wahrscheinlichstes Resultat.</span></article></section>
{% else %}<header class="page-head"><span class="kicker">{{'Live-Spielplan' if view=='matches' else 'Prediction Center'}}</span><h1>{{'Die nächsten Matches.' if view=='matches' else 'Alle Prognosen.'}}</h1><p>{{'Kommende Begegnungen mit lokaler Anstoßzeit.' if view=='matches' else 'Automatisch berechnet aus aktueller Stärke und Form.'}}</p></header><form class="toolbar"><input type="hidden" name="view" value="{{view}}"><div class="field"><label>Wettbewerb wählen</label><select name="competition" onchange="this.form.submit()">{% for code,label in competitions %}<option value="{{code}}" {% if code==competition %}selected{% endif %}>{{label}}</option>{% endfor %}</select></div><button class="refresh">Aktualisieren ↻</button></form>
{% if error=='missing' %}<div class="notice"><b>API-Token fehlt.</b><br>Trage <code>FOOTBALL_DATA_API_TOKEN</code> in die lokale <code>.env</code>-Datei ein und starte das Portal neu.</div>{% elif error %}<div class="notice">Die Live-Daten konnten gerade nicht geladen werden: {{error}}. Prüfe die API-Berechtigung für diesen Wettbewerb.</div>{% elif matches %}<section class="matches">{% for m in matches %}<article class="match" style="animation-delay:{{loop.index0*45}}ms"><div class="match-head"><span>{{m.day}}</span><b>{{m.kickoff}} Uhr</b></div><div class="teams"><div class="team">{% if m.home_crest %}<img src="{{m.home_crest}}" alt="">{% endif %}<span>{{m.home_name}}</span></div><div class="versus">VS</div><div class="team">{% if m.away_crest %}<img src="{{m.away_crest}}" alt="">{% endif %}<span>{{m.away_name}}</span></div></div>{% if view=='predictions' %}<div class="prediction"><small>Modell-Tipp · {{m.score}}</small><b>{{m.prediction}}</b></div><div class="odds"><div class="odd"><b>{{m.home}}%</b><span>HEIM</span><div class="formline"><i style="--w:{{m.home}}%"></i></div></div><div class="odd"><b>{{m.draw}}%</b><span>REMIS</span><div class="formline"><i style="--w:{{m.draw}}%"></i></div></div><div class="odd"><b>{{m.away}}%</b><span>AUSWÄRTS</span><div class="formline"><i style="--w:{{m.away}}%"></i></div></div></div><div class="technical">xG {{m.home_xg}} : {{m.away_xg}} · Form {{m.home_form}} : {{m.away_form}}</div>{% endif %}</article>{% endfor %}</section>{% else %}<div class="empty">Für diesen Wettbewerb sind derzeit keine kommenden Begegnungen angesetzt.</div>{% endif %}<div class="footer">Prognosen sind Wahrscheinlichkeiten und keine Ergebnisgarantie.</div>{% endif %}</main></body></html>'''


def _fetch(code: str) -> list[dict[str, object]]:
    return next_world_cup_predictions() if code == "WC" else next_predictions(code)


def create_app(default_competition: str = "PL") -> Flask:
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
            competitions=competition_choices(), matches=matches, error=error)
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
