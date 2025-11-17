from pprint import pprint

import numpy as np

from multisport_models import (
    MultiSportModels,
    FootballFeatureBuilder,
    FootballPlayerStats,
)
from data_providers import (
    find_apisports_fixture_by_teams_and_date,
    fetch_football_match_stats_from_apisports,
)


def main():
    print("=== Analyse FOOT (API-FOOTBALL + moteur KNN) ===")

    home = input("Équipe à domicile (home) : ").strip()
    away = input("Équipe à l'extérieur (away) : ").strip()
    date_str = input("Date du match (JJ/MM/AAAA ou JJ-MM-AAAA) : ").strip()

    # 1) Récupérer le match via API-FOOTBALL
    fixture_id = find_apisports_fixture_by_teams_and_date(home, away, date_str)
    print(f"[INFO] fixture_id sélectionné : {fixture_id}")

    match_stats = fetch_football_match_stats_from_apisports(fixture_id)
    print(f"[INFO] Match récupéré : {match_stats.home.name} vs {match_stats.away.name}")

    # 2) Créer le moteur multi-sport
    engine = MultiSportModels()
    fb = FootballFeatureBuilder()

    # 3) Entraînement DEMO du modèle foot
    #    (pour l'instant on "surentraîne" sur le match lui-même, juste pour que KNN ait quelque chose)
    X_match = np.vstack([fb.match_features(match_stats) for _ in range(30)])

    # Résultat : on met des étiquettes fictives pour l'exemple
    # (plus tard on utilisera les historiques réels)
    y_result = ["HOME"] * 15 + ["DRAW"] * 5 + ["AWAY"] * 10
    engine.football.fit_result(X_match, y_result)

    # Totaux de buts : pour l'instant valeur fixe 2.5
    y_goals = [2.5] * 30
    engine.football.fit_goals(X_match, y_goals)

    # 4) Option : buteur
    want_player = input("Analyser un buteur (probabilité de marquer) ? (o/n) : ").strip().lower()
    player_stats = None

    if want_player == "o":
        player_name = input("Nom du joueur : ").strip()
        team_for_player = input("Son équipe (telle qu'API la renvoie, ex: Germany ou Slovakia) : ").strip()

        # On choisit les stats d'équipe correspondantes
        if team_for_player.lower() == match_stats.home.name.lower():
            base_team = match_stats.home
        else:
            base_team = match_stats.away

        # Pour l'instant : statistiques joueur FICTIVES (plus tard : API joueur)
        player_stats = FootballPlayerStats(
            name=player_name,
            team_name=base_team.name,
            goals_per90=0.5,
            assists_per90=0.2,
            shots_per90=3.0,
            xg_per90=0.6,
            xa_per90=0.3,
            minutes_last5=400,
            is_injured=False
        )

        # Entraînement démo du modèle "buteur"
        X_player = np.vstack([fb.player_features(player_stats, base_team) for _ in range(50)])
        y_scored = [1] * 30 + [0] * 20
        engine.football.fit_player_scoring(X_player, y_scored)

    # 5) Analyse finale
    result = engine.analyze_football(match_stats, player_stats)

    print("\n=== RÉSULTAT ANALYSE ===")
    pprint(result)


if __name__ == "__main__":
    main()
