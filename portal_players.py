from __future__ import annotations

import portal
from scorer_service import add_scorer_predictions

_original_fetch = portal._fetch


def _fetch_with_players(code: str) -> list[dict[str, object]]:
    return add_scorer_predictions(_original_fetch(code), code)


portal._fetch = _fetch_with_players
portal.PAGE = portal.PAGE.replace(
    "</style></head>",
    r'''.scorers{margin-top:14px;padding-top:13px;border-top:1px solid var(--line)}.scorers-title{color:var(--muted);font-size:.6rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px}.scorer-list{display:flex;gap:6px}.scorer{flex:1;min-width:0;background:#06132a;border:1px solid #629dff1c;border-radius:10px;padding:8px}.scorer-name{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:.7rem;font-weight:750}.scorer-meta{display:flex;justify-content:space-between;color:var(--muted);font-size:.55rem;margin-top:4px}.scorer-prob{color:#69d3ff;font-weight:800}@media(max-width:420px){.scorer-list{display:grid;grid-template-columns:1fr}}
</style></head>''',
).replace(
    '<div class="technical">xG {{m.home_xg}} : {{m.away_xg}} · Form {{m.home_form}} : {{m.away_form}}</div>',
    '''<div class="technical">xG {{m.home_xg}} : {{m.away_xg}} · Form {{m.home_form}} : {{m.away_form}}</div>{% if m.scorers %}<div class="scorers"><div class="scorers-title">Wahrscheinlichste Torschützen · aktuelle Saison</div><div class="scorer-list">{% for p in m.scorers %}<div class="scorer"><span class="scorer-name">{{p.name}}</span><span class="scorer-meta"><span>{{p.goals}} Tore / {{p.appearances}} Sp.</span><span class="scorer-prob">{{p.probability}}%</span></span></div>{% endfor %}</div></div>{% endif %}''',
)

app = portal.create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
