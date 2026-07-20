from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial


@dataclass(frozen=True)
class Team:
    name: str
    rating: int


def _teams(values: dict[str, int]) -> list[Team]:
    return [Team(name, rating) for name, rating in values.items()]


NATIONS = {
    "Argentina": 1940, "Spain": 1935, "France": 1925, "England": 1900, "Brazil": 1895,
    "Portugal": 1890, "Netherlands": 1855, "Germany": 1850, "Italy": 1835, "Belgium": 1820,
    "Uruguay": 1815, "Colombia": 1810, "Croatia": 1795, "Morocco": 1785, "Japan": 1775,
    "Switzerland": 1765, "Norway": 1760, "Denmark": 1755, "Austria": 1745, "Turkey": 1740,
    "USA": 1735, "Senegal": 1730, "Ecuador": 1725, "Mexico": 1725, "South Korea": 1710,
    "Ukraine": 1700, "Canada": 1695, "Iran": 1690, "Czechia": 1690, "Egypt": 1685,
    "Nigeria": 1680, "Serbia": 1680, "Algeria": 1675, "Ivory Coast": 1670,
    "Australia": 1665, "Cameroon": 1655, "Chile": 1655, "Paraguay": 1650,
    "Tunisia": 1645, "Venezuela": 1645, "Mali": 1640, "Qatar": 1630, "Peru": 1625,
    "South Africa": 1625, "DR Congo": 1620, "Burkina Faso": 1615, "Panama": 1610,
    "Costa Rica": 1605, "Ghana": 1605, "Saudi Arabia": 1600, "Uzbekistan": 1595,
    "Iraq": 1585, "Jamaica": 1570, "United Arab Emirates": 1570, "Jordan": 1565,
    "Bolivia": 1530, "China": 1510,
}

COMPETITIONS: dict[str, tuple[str, list[Team]]] = {
    "PL": ("Premier League", _teams({"Arsenal":1900,"Manchester City":1910,"Liverpool":1900,"Chelsea":1815,"Manchester United":1780,"Tottenham Hotspur":1790,"Newcastle United":1810,"Aston Villa":1790,"Brighton":1750,"West Ham United":1720,"Crystal Palace":1740,"Brentford":1725,"Fulham":1715,"Everton":1695,"Wolverhampton":1685,"Bournemouth":1710,"Nottingham Forest":1740,"Leeds United":1660,"Burnley":1620,"Sunderland":1610})),
    "BL1": ("Bundesliga", _teams({"Bayern Munich":1920,"Bayer Leverkusen":1865,"Borussia Dortmund":1840,"RB Leipzig":1815,"Eintracht Frankfurt":1790,"VfB Stuttgart":1780,"SC Freiburg":1740,"Mainz 05":1730,"Borussia Monchengladbach":1710,"Werder Bremen":1695,"Wolfsburg":1700,"Augsburg":1680,"Union Berlin":1680,"Hoffenheim":1670,"St. Pauli":1640,"Heidenheim":1630,"Hamburg":1660,"Cologne":1650})),
    "PD": ("La Liga", _teams({"Real Madrid":1940,"Barcelona":1930,"Atletico Madrid":1850,"Athletic Club":1810,"Villarreal":1790,"Real Betis":1770,"Real Sociedad":1765,"Sevilla":1710,"Valencia":1700,"Girona":1720,"Celta Vigo":1710,"Osasuna":1695,"Getafe":1675,"Rayo Vallecano":1685,"Mallorca":1680,"Alaves":1650,"Espanyol":1645,"Levante":1615,"Elche":1605,"Real Oviedo":1595})),
    "SA": ("Serie A", _teams({"Inter Milan":1900,"Napoli":1870,"AC Milan":1840,"Juventus":1840,"Atalanta":1835,"Roma":1810,"Lazio":1780,"Fiorentina":1765,"Bologna":1760,"Torino":1700,"Genoa":1685,"Udinese":1675,"Como":1710,"Parma":1640,"Cagliari":1640,"Lecce":1625,"Verona":1620,"Sassuolo":1640,"Pisa":1580,"Cremonese":1570})),
    "FL1": ("Ligue 1", _teams({"Paris Saint-Germain":1940,"Marseille":1810,"Monaco":1805,"Lille":1795,"Lyon":1775,"Nice":1765,"Lens":1755,"Strasbourg":1735,"Rennes":1725,"Brest":1710,"Toulouse":1690,"Auxerre":1645,"Nantes":1640,"Angers":1625,"Le Havre":1605,"Lorient":1610,"Metz":1595,"Paris FC":1600})),
    "CL": ("UEFA Champions League", _teams({"Real Madrid":1940,"Paris Saint-Germain":1940,"Barcelona":1930,"Bayern Munich":1920,"Manchester City":1910,"Arsenal":1900,"Liverpool":1900,"Inter Milan":1900,"Bayer Leverkusen":1865,"Napoli":1870,"Atletico Madrid":1850,"Borussia Dortmund":1840,"Juventus":1840,"AC Milan":1840,"Atalanta":1835,"Chelsea":1815,"RB Leipzig":1815,"Marseille":1810,"Monaco":1805,"Benfica":1790,"Sporting CP":1785,"PSV":1775,"Ajax":1775,"Galatasaray":1760,"Club Brugge":1725,"Olympiacos":1710,"Celtic":1690,"Slavia Prague":1680,"Bodo/Glimt":1670,"Copenhagen":1670,"Red Star Belgrade":1660,"Dynamo Kyiv":1650,"Qarabag":1600})),
    "WC": ("FIFA World Cup", _teams(NATIONS)),
    "EC": ("UEFA European Championship", _teams({k:v for k,v in NATIONS.items() if k in {"Spain","France","England","Portugal","Netherlands","Germany","Italy","Belgium","Croatia","Switzerland","Denmark","Austria","Turkey","Norway","Ukraine","Czechia","Serbia"}})),
    "CA": ("Copa America", _teams({k:v for k,v in NATIONS.items() if k in {"Argentina","Brazil","Uruguay","Colombia","Ecuador","USA","Mexico","Canada","Paraguay","Venezuela","Chile","Peru","Costa Rica","Panama","Jamaica","Bolivia"}})),
    "AFCON": ("Africa Cup of Nations", _teams({k:v for k,v in NATIONS.items() if k in {"Morocco","Senegal","Egypt","Nigeria","Algeria","Ivory Coast","Cameroon","Tunisia","Mali","South Africa","DR Congo","Burkina Faso","Ghana"}})),
    "AC": ("AFC Asian Cup", _teams({k:v for k,v in NATIONS.items() if k in {"Japan","South Korea","Iran","Australia","Qatar","Saudi Arabia","Uzbekistan","Iraq","United Arab Emirates","Jordan","China"}})),
}


def competition_choices() -> list[tuple[str, str]]:
    return [(code, value[0]) for code, value in COMPETITIONS.items()]


def teams_for(code: str) -> list[Team]:
    return sorted(COMPETITIONS.get(code, COMPETITIONS["PL"])[1], key=lambda team: team.name)


def predict(
    home: Team,
    away: Team,
    neutral: bool = False,
    must_decide: bool = False,
) -> dict[str, object]:
    difference = home.rating + (0 if neutral else 65) - away.rating
    home_xg = max(.35, min(3.4, 1.34 * 10 ** (difference / 800)))
    away_xg = max(.30, min(3.1, 1.12 * 10 ** (-difference / 800)))
    hp = [exp(-home_xg) * home_xg**i / factorial(i) for i in range(11)]
    ap = [exp(-away_xg) * away_xg**i / factorial(i) for i in range(11)]
    matrix = [[h * a for a in ap] for h in hp]
    outcomes = [sum(matrix[i][j] for i in range(1, 11) for j in range(i)),
                sum(matrix[i][i] for i in range(11)),
                sum(matrix[i][j] for i in range(11) for j in range(i + 1, 11))]
    total = sum(outcomes)
    probs = [value / total for value in outcomes]
    score_candidates = (
        (i, j)
        for i in range(7)
        for j in range(7)
        if not must_decide or i != j
    )
    score = max(score_candidates, key=lambda s: matrix[s[0]][s[1]])

    if must_decide:
        # A knockout match needs a winner. Split the regulation-time draw chance
        # between both teams according to their underlying win strength.
        decisive_total = probs[0] + probs[2]
        home_advance = probs[0] / decisive_total
        away_advance = probs[2] / decisive_total
        best = 0 if home_advance >= away_advance else 2
        labels = [f"{home.name} setzt sich durch", "", f"{away.name} setzt sich durch"]
        confidence_value = max(home_advance, away_advance)
    else:
        home_advance, away_advance = probs[0], probs[2]
        labels = [f"{home.name} gewinnt", "Unentschieden", f"{away.name} gewinnt"]
        best = max(range(3), key=probs.__getitem__)
        confidence_value = probs[best]

    confidence = "Hoch" if confidence_value >= .60 else "Mittel" if confidence_value >= .45 else "Offen"
    return {"home":round(probs[0]*100,1),"draw":round(probs[1]*100,1),"away":round(probs[2]*100,1),
            "prediction":labels[best],"score":f"{score[0]}–{score[1]}","home_xg":round(home_xg,2),
            "away_xg":round(away_xg,2),"confidence":confidence,
            "must_decide":must_decide,
            "home_advance":round(home_advance*100, 1),
            "away_advance":round(away_advance*100, 1)}
