from __future__ import annotations

import argparse
from flask import Flask, render_template_string, request
from prediction_engine import COMPETITIONS, competition_choices, predict, teams_for

PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Matchday AI</title><style>
:root{--bg:#07120e;--panel:#10231b;--line:#284639;--green:#37e18c;--text:#f4fff8;--muted:#9db8aa}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% -10%,#174b34,transparent 40%),var(--bg);color:var(--text);font-family:Inter,system-ui,sans-serif}main{max-width:880px;margin:auto;padding:32px 18px}.brand{color:var(--green);font-weight:800;letter-spacing:.12em;text-transform:uppercase}h1{font-size:clamp(2rem,6vw,4rem);line-height:1;margin:.4rem 0}.sub{color:var(--muted);max-width:650px;line-height:1.55}.card{margin-top:28px;background:linear-gradient(145deg,#132a20,#0c1c16);border:1px solid var(--line);border-radius:22px;padding:clamp(18px,4vw,34px);box-shadow:0 25px 80px #0008}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.full{grid-column:1/-1}label{display:block;color:var(--muted);font-size:.8rem;font-weight:700;margin-bottom:7px;text-transform:uppercase;letter-spacing:.06em}select,button{width:100%;padding:14px;border-radius:12px;border:1px solid #385b4b;background:#091711;color:var(--text);font-size:1rem}button{background:var(--green);color:#052014;border:0;font-weight:800;cursor:pointer}.check{display:flex;align-items:center;gap:10px;color:var(--muted);margin:5px 0 18px}.check input{width:18px;height:18px}.result{text-align:center;border-bottom:1px solid var(--line);padding-bottom:24px}.result small{color:var(--muted);text-transform:uppercase}.result h2{font-size:2rem;margin:.4rem}.score{font-size:3.3rem;font-weight:900;color:var(--green)}.pills{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:22px}.pill{background:#091711;border:1px solid var(--line);padding:14px;border-radius:14px}.pill b{display:block;font-size:1.5rem}.pill span{color:var(--muted);font-size:.8rem}.meta{display:flex;justify-content:center;gap:30px;color:var(--muted);margin-top:20px;font-size:.9rem}.note{font-size:.78rem;color:#789486;line-height:1.5;margin-top:24px}@media(max-width:600px){.grid{grid-template-columns:1fr}.full{grid-column:auto}.pills{gap:6px}.pill{padding:10px}.pill b{font-size:1.2rem}.meta{display:block;text-align:center}}
</style></head><body><main><div class="brand">⚽ Matchday AI</div><h1>Football predictions,<br>without the fuss.</h1><p class="sub">Choose a major league or international tournament, select the teams, and get an instant data-informed match forecast.</p>
<form class="card" method="post"><div class="grid"><div class="full"><label>Competition</label><select name="competition" onchange="this.form.submit()">{% for code,label in competitions %}<option value="{{code}}" {% if code==competition %}selected{% endif %}>{{label}}</option>{% endfor %}</select></div><div><label>Home / Team A</label><select name="home_team">{% for t in teams %}<option {% if t.name==home_name %}selected{% endif %}>{{t.name}}</option>{% endfor %}</select></div><div><label>Away / Team B</label><select name="away_team">{% for t in teams %}<option {% if t.name==away_name %}selected{% endif %}>{{t.name}}</option>{% endfor %}</select></div><div class="full"><label class="check"><input type="checkbox" name="neutral" {% if neutral %}checked{% endif %}> Neutral venue (tournaments / finals)</label><button name="action" value="predict">Predict match</button></div></div></form>
{% if error %}<div class="card">{{error}}</div>{% endif %}{% if result %}<section class="card"><div class="result"><small>Most likely outcome · {{result.confidence}} confidence</small><h2>{{result.prediction}}</h2><div class="score">{{result.score}}</div><small>Most likely score</small></div><div class="pills"><div class="pill"><b>{{result.home}}%</b><span>{{home_name}} win</span></div><div class="pill"><b>{{result.draw}}%</b><span>Draw</span></div><div class="pill"><b>{{result.away}}%</b><span>{{away_name}} win</span></div></div><div class="meta"><span>Expected goals: {{result.home_xg}} – {{result.away_xg}}</span><span>{{"Neutral venue" if neutral else "Home advantage included"}}</span></div><p class="note">Forecasts are estimates, not guarantees. This model combines team-strength ratings, home advantage and expected-goal distributions. Injuries, line-ups and late news are not included; never treat predictions as betting advice.</p></section>{% endif %}</main></body></html>'''


def create_app(default_competition: str = "PL") -> Flask:
    app = Flask(__name__)
    app.config["DEFAULT_COMPETITION"] = default_competition if default_competition in COMPETITIONS else "PL"

    @app.route("/", methods=["GET", "POST"])
    def index() -> str:
        code = request.values.get("competition", app.config["DEFAULT_COMPETITION"])
        code = code if code in COMPETITIONS else "PL"
        teams = teams_for(code)
        home_name = request.form.get("home_team", teams[0].name)
        away_name = request.form.get("away_team", teams[1].name)
        neutral = request.form.get("neutral") == "on"
        result = error = None
        if request.form.get("action") == "predict":
            available = {team.name: team for team in teams}
            if home_name == away_name:
                error = "Please choose two different teams."
            elif home_name not in available or away_name not in available:
                error = "A selected team is not in this competition."
            else:
                result = predict(available[home_name], available[away_name], neutral)
        return render_template_string(PAGE, competitions=competition_choices(), competition=code,
            teams=teams, home_name=home_name, away_name=away_name, neutral=neutral, result=result, error=error)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Football prediction web interface")
    parser.add_argument("--competition", default="PL", choices=COMPETITIONS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    create_app(args.competition).run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
