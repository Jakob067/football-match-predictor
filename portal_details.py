from __future__ import annotations

import portal_players

portal = portal_players.portal

portal.PAGE = portal.PAGE.replace(
    "</style></head>",
    r'''.match{cursor:pointer}.match:focus-visible{outline:2px solid var(--cyan);outline-offset:3px}.details-toggle{display:flex;align-items:center;justify-content:center;gap:8px;width:100%;height:38px;margin-top:14px;border:1px solid var(--line);border-radius:11px;background:#0b2349;color:#a9caff;font-size:.7rem;font-weight:800;cursor:pointer;transition:.22s}.details-toggle:hover{background:#123366;color:white}.chevron{display:inline-block;transition:transform .35s}.match.open .chevron{transform:rotate(180deg)}.match-details{display:grid;grid-template-rows:0fr;opacity:0;transition:grid-template-rows .48s cubic-bezier(.2,.8,.2,1),opacity .35s,margin .35s;margin-top:0}.match-details-inner{min-height:0;overflow:hidden}.match.open .match-details{grid-template-rows:1fr;opacity:1;margin-top:4px}.match.open{border-color:#67a6ff78;box-shadow:0 24px 70px #0009}.detail-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-top:11px}.detail-stat{padding:9px 6px;text-align:center;border-radius:10px;background:#06132a;border:1px solid #639fff14}.detail-stat b{display:block;color:#dbe9ff;font-size:.8rem}.detail-stat span{display:block;color:var(--muted);font-size:.54rem;text-transform:uppercase;margin-top:2px}.explanation{margin-top:11px;padding:10px;border-left:2px solid #56b9ff;background:#06132a;color:#7895bf;font-size:.62rem;line-height:1.55;border-radius:0 8px 8px 0}.click-hint{text-align:center;color:#58739d;font-size:.56rem;margin-top:8px}.match.open .click-hint{display:none}@media(prefers-reduced-motion:reduce){.match-details{transition:none}}
</style></head>''',
).replace(
    '<article class="match" style="animation-delay:{{loop.index0*45}}ms">',
    '<article class="match" style="animation-delay:{{loop.index0*45}}ms" tabindex="0" role="button" aria-expanded="false" onclick="toggleMatch(this,event)" onkeydown="keyMatch(this,event)">',
).replace(
    "{% if view=='predictions' %}<div class=\"prediction\">",
    '<button type="button" class="details-toggle"><span>Genaue Analyse anzeigen</span><i class="chevron">⌄</i></button><div class="match-details"><div class="match-details-inner"><div class="prediction">',
).replace(
    '<div class="technical">xG {{m.home_xg}} : {{m.away_xg}} · Form {{m.home_form}} : {{m.away_form}}</div>',
    '<div class="detail-grid"><div class="detail-stat"><b>{{m.home_xg}} : {{m.away_xg}}</b><span>Expected Goals</span></div><div class="detail-stat"><b>{{m.home_form}} : {{m.away_form}}</b><span>Formpunkte</span></div><div class="detail-stat"><b>{{m.confidence}}</b><span>Modellvertrauen</span></div></div><div class="explanation">Die Prozentwerte kombinieren aktuelle Elo-Stärke, letzte Ergebnisse, Tordifferenz, Heimvorteil und eine Poisson-Torverteilung. Torschützen werden anhand ihrer aktuellen Torquote an das erwartete Team-xG angepasst.</div>',
).replace(
    '{% endif %}{% endif %}</article>{% endfor %}',
    '{% endif %}</div></div><div class="click-hint">Karte anklicken für Details</div></article>{% endfor %}',
).replace(
    "</body></html>",
    r'''<script>
function toggleMatch(card,event){
  if(event.target.closest('a,select'))return;
  const open=card.classList.toggle('open');
  card.setAttribute('aria-expanded',String(open));
  const label=card.querySelector('.details-toggle span');
  if(label)label.textContent=open?'Analyse schließen':'Genaue Analyse anzeigen';
  if(open)setTimeout(()=>card.scrollIntoView({behavior:'smooth',block:'nearest'}),180);
}
function keyMatch(card,event){if(event.key==='Enter'||event.key===' '){event.preventDefault();toggleMatch(card,event)}}
</script></body></html>''',
)

app = portal.create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
