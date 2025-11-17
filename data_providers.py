import os
from datetime import datetime
from typing import Optional
import requests
from dotenv import load_dotenv

from multisport_models import FootballTeamStats, FootballMatchStats

# Charger les variables d'environnement
load_dotenv()

SPORTRADAR_API_KEY = os.getenv("SPORTRADAR_API_KEY")
SPORTRADAR_SOCCER_ACCESS_LEVEL = os.getenv("SPORTRADAR_SOCCER_ACCESS_LEVEL", "trial")
SPORTRADAR_SOCCER_LANG = os.getenv("SPORTRADAR_SOCCER_LANG", "en")
SPORTRADAR_SOCCER_BASE_URL = "https://api.sportradar.com/soccer"

# --- API-FOOTBALL / API-SPORTS ---
APISPORTS_API_KEY = os.getenv("APISPORTS_API_KEY")
APISPORTS_BASE_URL = os.getenv("APISPORTS_BASE_URL", "https://v3.football.api-sports.io")

def normalize_user_date_dmy(date_str: str) -> str:
    """
    L'utilisateur tape JJ/MM/AAAA ou JJ-MM-AAAA.
    On renvoie YYYY-MM-DD pour Sportradar.
    """
    s = date_str.strip().replace("-", "/")
    # Exemple attendu : 15/11/2025
    try:
        dt = datetime.strptime(s, "%d/%m/%Y")
    except ValueError:
        raise ValueError(
            f"Format de date invalide : '{date_str}'. Utilise JJ/MM/AAAA, ex: 15/11/2025."
        )
    return dt.strftime("%Y-%m-%d")

def get_apisports_session() -> requests.Session:
    """
    Crée une session pour API-FOOTBALL (API-Sports).
    """
    if not APISPORTS_API_KEY:
        raise RuntimeError("APISPORTS_API_KEY manquante dans .env")

    session = requests.Session()
    session.headers.update({
        "x-apisports-key": APISPORTS_API_KEY
    })
    return session

def get_sportradar_session() -> requests.Session:
    if not SPORTRADAR_API_KEY:
        raise RuntimeError("SPORTRADAR_API_KEY manquante dans .env")

    session = requests.Session()
    # Sportradar Soccer v4 utilise généralement api_key en query string
    # donc pas besoin de header spécial, mais on pourrait en ajouter si ton contrat le demande.
    return session


def find_soccer_event_by_teams_and_date(
    home_name: str,
    away_name: str,
    date_user: str
) -> str:
    """
    Utilise le Daily Schedules de Sportradar pour trouver un sport_event_id
    correspondant à home_name vs away_name à une date donnée.

    ⚠️ On fait du matching "souple" sur les noms :
    - minuscules
    - on tolère les sous-chaînes
    - ex : 'Allemagne' ne matchera sûrement pas 'Germany',
      donc pour les sélections utilise plutôt les noms en anglais
      (Germany, Slovakia, France, etc.)
    """
    session = get_sportradar_session()
    iso_date = normalize_user_date_dmy(date_user)

    url = f"{SPORTRADAR_SOCCER_BASE_URL}/{SPORTRADAR_SOCCER_ACCESS_LEVEL}/v4/{SPORTRADAR_SOCCER_LANG}/schedules/{iso_date}/schedules.json"
    params = {"api_key": SPORTRADAR_API_KEY}

    resp = session.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    schedules = data.get("schedules", [])
    print(f"[DEBUG] {len(schedules)} matchs trouvés pour le {iso_date} (Sportradar)")

    if not schedules:
        raise RuntimeError(
            f"Aucun match trouvé pour la date {iso_date} via Sportradar. "
            f"Vérifie que ton plan inclut des compétitions ce jour-là."
        )

    def norm(s: str) -> str:
        return s.strip().lower()

    def team_match(user_input: str, provider_name: str) -> bool:
        """
        Matching souple :
        - on compare en minuscules
        - on accepte égalité exacte
        - ou user_input inclus dans provider_name
        - ou provider_name inclus dans user_input
        ex : 'new mexico' ≈ 'new mexico united'
        """
        u = norm(user_input)
        p = norm(provider_name)
        if not u or not p:
            return False
        if u == p:
            return True
        if u in p:
            return True
        if p in u:
            return True
        return False

    target_home = home_name
    target_away = away_name

    best_event_id: Optional[str] = None
    best_score: float = 0.0

    for sch in schedules:
        sport_event = sch.get("sport_event", {})
        competitors = sport_event.get("competitors", [])

        home_team_name: Optional[str] = None
        away_team_name: Optional[str] = None

        for comp in competitors:
            q = comp.get("qualifier")
            name = comp.get("name", "")
            if q == "home":
                home_team_name = name
            elif q == "away":
                away_team_name = name

        if not home_team_name or not away_team_name:
            continue

        # Score simple : 1 si ça matche, 0 sinon
        home_ok = team_match(target_home, home_team_name)
        away_ok = team_match(target_away, away_team_name)

        # On peut aussi afficher quelques matches pour debug
        # print(f"[DEBUG-MATCHES] {home_team_name} vs {away_team_name}")

        score = (1.0 if home_ok else 0.0) + (1.0 if away_ok else 0.0)

        if score > best_score:
            best_score = score
            best_event_id = sport_event.get("id")

    # On considère qu'il faut au minimum 2/2 pour être sûr (home et away matchent)
    if best_event_id is None or best_score < 2.0:
        raise RuntimeError(
            f"Aucun match {home_name} vs {away_name} trouvé le {iso_date} dans les daily schedules "
            f"(score de matching max = {best_score}). "
            f"Essaye avec les noms exacts en anglais tels qu'ils apparaissent chez Sportradar."
        )

    print(f"[DEBUG] Match choisi (score={best_score}) → event_id = {best_event_id}")
    return best_event_id              

def find_apisports_fixture_by_teams_and_date(
    home_name: str,
    away_name: str,
    date_user: str
) -> int:
    """
    Utilise API-FOOTBALL pour trouver un fixture_id à partir de :
    - home_name
    - away_name
    - date (JJ/MM/AAAA ou JJ-MM-AAAA)

    Matching souple sur les noms d'équipes.
    """
    session = get_apisports_session()
    iso_date = normalize_user_date_dmy(date_user)

    url = f"{APISPORTS_BASE_URL}/fixtures"
    params = {"date": iso_date}

    resp = session.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    fixtures = data.get("response", [])
    print(f"[DEBUG API-FOOTBALL] {len(fixtures)} fixtures trouvés pour le {iso_date}")

    if not fixtures:
        raise RuntimeError(
            f"Aucun match trouvé pour la date {iso_date} via API-FOOTBALL. "
            f"Vérifie que des matchs existent ce jour-là et que ton plan y a accès."
        )

    def norm(s: str) -> str:
        return s.strip().lower()

    def team_match(user_input: str, provider_name: str) -> bool:
        u = norm(user_input)
        p = norm(provider_name)
        if not u or not p:
            return False
        if u == p:
            return True
        if u in p:
            return True
        if p in u:
            return True
        return False

    best_fixture_id = None
    best_score = 0.0

    for item in fixtures:
        teams = item.get("teams", {})
        home = teams.get("home", {}).get("name", "")
        away = teams.get("away", {}).get("name", "")

        score = 0.0
        if team_match(home_name, home):
            score += 1.0
        if team_match(away_name, away):
            score += 1.0

        if score > best_score:
            best_score = score
            best_fixture_id = item.get("fixture", {}).get("id")

    if best_fixture_id is None or best_score < 2.0:
        raise RuntimeError(
            f"Aucun match {home_name} vs {away_name} trouvé le {iso_date} dans API-FOOTBALL "
            f"(score de matching max = {best_score}). "
            f"Essaye avec les noms tels qu'ils apparaissent dans la doc ou le dashboard API."
        )

    print(f"[DEBUG API-FOOTBALL] Fixture choisi (score={best_score}) → id = {best_fixture_id}")
    return int(best_fixture_id)

def fetch_soccer_match_stats_from_sportradar(event_id: str) -> FootballMatchStats:
    """
    Appelle l'endpoint Sport Event Summary pour récupérer les stats match.
    NB : la structure exacte dépend de ton package et de la couverture.
    Ici on fait une version GENERIQUE, à adapter après avoir vu ton JSON réel.
    """
    session = get_sportradar_session()
    url = f"{SPORTRADAR_SOCCER_BASE_URL}/{SPORTRADAR_SOCCER_ACCESS_LEVEL}/v4/{SPORTRADAR_SOCCER_LANG}/sport_events/{event_id}/summary.json"
    params = {"api_key": SPORTRADAR_API_KEY}

    resp = session.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # 👉 IMPORTANT :
    # La structure exacte de 'data' dépend de ton plan/couverture.
    # Typiquement tu auras quelque chose comme :
    # data["statistics"]["totals"]["competitors"] ou data["statistics"]["teams"]
    # avec pour chaque équipe : buts, tirs, corners, etc.
    #
    # Comme je ne peux pas voir ton JSON exact, on va :
    # 1) afficher les clés principales pour que tu vérifies
    # 2) mettre un mapping minimal à adapter ensuite.

    print("[DEBUG] Clés racine summary:", list(data.keys()))

    # --- À ADAPTER APRÈS INSPECTION ---
    # Pour l'instant, on suppose une structure "totals" avec "competitors"
    statistics = data.get("statistics", {})
    totals = statistics.get("totals", {})
    competitors_stats = totals.get("competitors", [])

    if len(competitors_stats) != 2:
        raise RuntimeError(
            "Structure inattendue dans le summary Sportradar. "
            "Vérifie la structure JSON exacte dans la doc ou en print complet."
        )

    # On crée un petit helper
    def build_team_stats(comp_stat) -> FootballTeamStats:
        team_info = comp_stat.get("team") or comp_stat.get("competitor") or {}
        team_name = team_info.get("name", "Unknown")

        # Ces champs sont À ADAPTER selon ton JSON réel.
        # Je mets des clés plausibles pour te donner le squelette.
        team_values = comp_stat.get("statistics", {})

        def get_val(key: str, default: float = 0.0) -> float:
            val = team_values.get(key, default)
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        goals_for = get_val("goals_scored", 0.0)
        shots_total = get_val("shots_total", 0.0)
        shots_on = get_val("shots_on_target", 0.0)
        corners = get_val("corner_kicks", 0.0)
        yellow_cards = get_val("yellow_cards", 0.0)

        return FootballTeamStats(
            name=team_name,
            goals_for_avg=goals_for,
            goals_against_avg=0.0,  # on fixe après avec l'autre équipe
            shots_for_avg=shots_total,
            shots_on_target_for_avg=shots_on,
            corners_for_avg=corners,
            cards_for_avg=yellow_cards,
            touches_for_avg=None,
            win_rate=0.0,
            draw_rate=0.0,
            loss_rate=0.0,
        )

    team1_stats = build_team_stats(competitors_stats[0])
    team2_stats = build_team_stats(competitors_stats[1])

    # buts encaissés = buts marqués par l'adversaire sur ce match
    team1_stats.goals_against_avg = team2_stats.goals_for_avg
    team2_stats.goals_against_avg = team1_stats.goals_for_avg

    # Qualifier pour savoir qui est HOME / AWAY, si dispo
    # (au besoin, on pourra les inverser selon 'qualifier')
    sport_event = data.get("sport_event", {})
    competitors = sport_event.get("competitors", [])
    if len(competitors) == 2:
        # on essaie de recaler team1_stats = home, team2_stats = away
        home_name = next((c["name"] for c in competitors if c.get("qualifier") == "home"), None)
        away_name = next((c["name"] for c in competitors if c.get("qualifier") == "away"), None)

        if home_name and away_name:
            # on ré-ordonne team1/team2 si besoin
            if team1_stats.name == away_name and team2_stats.name == home_name:
                team1_stats, team2_stats = team2_stats, team1_stats

    match_stats = FootballMatchStats(
        home=team1_stats,
        away=team2_stats
    )
    return match_stats

def fetch_football_match_stats_from_apisports(fixture_id: int) -> FootballMatchStats:
    """
    Récupère les stats d'un match de foot via API-FOOTBALL et les convertit
    en FootballMatchStats utilisables par notre moteur.

    Version robuste :
    - si /fixtures/statistics est vide, on construit quand même des stats minimales
      à partir du fixture (buts, noms d'équipes).
    """
    session = get_apisports_session()

    # 1) Récupérer les infos du match (home/away, score si dispo)
    url_fixture = f"{APISPORTS_BASE_URL}/fixtures"
    resp_fix = session.get(url_fixture, params={"id": fixture_id}, timeout=15)
    resp_fix.raise_for_status()
    data_fix = resp_fix.json()
    resp_list = data_fix.get("response", [])
    if not resp_list:
        raise RuntimeError(f"Aucun fixture trouvé pour id={fixture_id} dans API-FOOTBALL.")

    fixture = resp_list[0]
    teams = fixture.get("teams", {})
    goals = fixture.get("goals", {}) or {}

    home_name = teams.get("home", {}).get("name", "Home")
    away_name = teams.get("away", {}).get("name", "Away")

    home_goals = goals.get("home", 0)
    away_goals = goals.get("away", 0)

    # Certains matchs futurs auront home_goals/away_goals = None
    try:
        home_goals = float(home_goals) if home_goals is not None else 0.0
    except (TypeError, ValueError):
        home_goals = 0.0

    try:
        away_goals = float(away_goals) if away_goals is not None else 0.0
    except (TypeError, ValueError):
        away_goals = 0.0

    # 2) Récupérer les stats détaillées (si dispo)
    url_stats = f"{APISPORTS_BASE_URL}/fixtures/statistics"
    resp_stats = session.get(url_stats, params={"fixture": fixture_id}, timeout=15)
    resp_stats.raise_for_status()
    data_stats = resp_stats.json()
    stats_by_team = data_stats.get("response", [])

    # --- CAS 1 : PAS DE STATS DÉTAILLÉES (match futur ou non couvert) ---
    if len(stats_by_team) == 0:
        print(f"[DEBUG API-FOOTBALL] Pas de statistics pour fixture {fixture_id}, "
              f"on construit des stats minimales.")

        home_stats = FootballTeamStats(
            name=home_name,
            goals_for_avg=home_goals,
            goals_against_avg=away_goals,
            shots_for_avg=0.0,
            shots_on_target_for_avg=0.0,
            corners_for_avg=0.0,
            cards_for_avg=0.0,
            touches_for_avg=None,
            win_rate=0.0,
            draw_rate=0.0,
            loss_rate=0.0,
        )

        away_stats = FootballTeamStats(
            name=away_name,
            goals_for_avg=away_goals,
            goals_against_avg=home_goals,
            shots_for_avg=0.0,
            shots_on_target_for_avg=0.0,
            corners_for_avg=0.0,
            cards_for_avg=0.0,
            touches_for_avg=None,
            win_rate=0.0,
            draw_rate=0.0,
            loss_rate=0.0,
        )

        return FootballMatchStats(
            home=home_stats,
            away=away_stats
        )

    # --- CAS 2 : STATS COMPLÈTES DISPONIBLES (2 équipes) ---
    if len(stats_by_team) != 2:
        raise RuntimeError(
            f"Réponse statistics inattendue pour fixture {fixture_id}: {stats_by_team}"
        )

    def extract_team_stats(team_block) -> FootballTeamStats:
        team_name = team_block["team"]["name"]
        statistics = team_block.get("statistics", [])

        def get_value(stat_type: str, default=0.0):
            for item in statistics:
                if item["type"].lower() == stat_type.lower():
                    val = item["value"]
                    if val is None:
                        return default
                    if isinstance(val, str) and val.endswith("%"):
                        try:
                            return float(val.strip("%")) / 100.0
                        except ValueError:
                            return default
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        return default
            return default

        goals_for = get_value("Goals", 0.0)
        shots_total = get_value("Total Shots", 0.0)
        shots_on = get_value("Shots on Goal", 0.0)
        corners = get_value("Corner Kicks", 0.0)
        yellow_cards = get_value("Yellow Cards", 0.0)

        return FootballTeamStats(
            name=team_name,
            goals_for_avg=goals_for,
            goals_against_avg=0.0,  # mis à jour après
            shots_for_avg=shots_total,
            shots_on_target_for_avg=shots_on,
            corners_for_avg=corners,
            cards_for_avg=yellow_cards,
            touches_for_avg=None,
            win_rate=0.0,
            draw_rate=0.0,
            loss_rate=0.0,
        )

    team_stats_list = [extract_team_stats(x) for x in stats_by_team]

    # Recalage sur home / away
    home_stats = None
    away_stats = None
    for ts in team_stats_list:
        if ts.name == home_name:
            home_stats = ts
        elif ts.name == away_name:
            away_stats = ts

    # si pas trouvé exactement, fallback sur l'ordre
    if home_stats is None or away_stats is None:
        home_stats = team_stats_list[0]
        away_stats = team_stats_list[1]

    # buts encaissés = buts de l'adversaire sur ce match
    home_stats.goals_against_avg = away_stats.goals_for_avg
    away_stats.goals_against_avg = home_stats.goals_for_avg

    return FootballMatchStats(
        home=home_stats,
        away=away_stats
    )

if __name__ == "__main__":
    print("Test API-FOOTBALL (API-Sports) - FOOTBALL")
    home = input("Nom équipe à domicile (home) : ").strip()
    away = input("Nom équipe à l'extérieur (away) : ").strip()
    date_str = input("Date du match (JJ/MM/AAAA ou JJ-MM-AAAA) : ").strip()

    try:
        fixture_id = find_apisports_fixture_by_teams_and_date(home, away, date_str)
        print(f"[OK] fixture_id trouvé : {fixture_id}")
        match_stats = fetch_football_match_stats_from_apisports(fixture_id)
        print("\n=== HOME ===")
        print(match_stats.home)
        print("\n=== AWAY ===")
        print(match_stats.away)
    except Exception as e:
        print("Erreur :", e)
