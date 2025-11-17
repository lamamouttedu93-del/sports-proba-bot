import os
import asyncio
from dotenv import load_dotenv

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

from multisport_models import (
    MultiSportModels,
    FootballFeatureBuilder,
    FootballPlayerStats,
)
from data_providers import (
    find_apisports_fixture_by_teams_and_date,
    fetch_football_match_stats_from_apisports,
)

# Charger les variables d'environnement (.env)
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Commande /start
    """
    user = update.effective_user
    msg = (
        f"Salut {user.first_name or '👤'} !\n\n"
        "Je suis ton bot d'analyse sportive perso.\n"
        "Pour l'instant je sais juste répondre à /ping,\n"
        "mais on va bientôt me connecter à l'IA de pronostics. ⚽📊"
    )
    await update.message.reply_text(msg)


async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Commande /ping → /pong
    """
    await update.message.reply_text("pong 🏓")

async def analyze_foot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Commande /analyze_foot
    Format attendu dans le même message :
    /analyze_foot Germany Slovakia 17/11/2025 L.Sane Germany

    - 3 premiers paramètres OBLIGATOIRES :
        home away date(JJ/MM/AAAA ou JJ-MM-AAAA)
    - 2 derniers paramètres OPTIONNELS :
        nom_du_joueur équipe_du_joueur

    Exemple simple (sans joueur) :
        /analyze_foot Germany Slovakia 17/11/2025

    Exemple avec joueur :
        /analyze_foot Germany Slovakia 17/11/2025 L.Sane Germany
    """
    msg = update.message
    if msg is None:
        return

    args = context.args  # liste des paramètres après la commande
    if len(args) < 3:
        await msg.reply_text(
            "❌ Format incorrect.\n\n"
            "Exemples :\n"
            "/analyze_foot Germany Slovakia 17/11/2025\n"
            "/analyze_foot Germany Slovakia 17/11/2025 L.Sane Germany"
        )
        return

    home = args[0]
    away = args[1]
    date_str = args[2]

    player_name = None
    player_team = None
    if len(args) >= 5:
        # on concatène au cas où le nom du joueur ou de l'équipe contient un espace
        player_name = args[3]
        player_team = " ".join(args[4:])  # tout ce qui reste

    await msg.reply_text(
        f"🔎 Analyse du match : {home} vs {away} le {date_str}...\n"
        f"(les résultats sont informatifs, sans garantie de gain 😉)"
    )

    try:
        # 1) Récupérer les stats du match via API-FOOTBALL
        fixture_id = find_apisports_fixture_by_teams_and_date(home, away, date_str)
        match_stats = fetch_football_match_stats_from_apisports(fixture_id)

        # 2) Créer le moteur et feature builder
        engine = MultiSportModels()
        fb = FootballFeatureBuilder()

        # 3) Entraînement DEMO (comme dans test_engine.py)
        import numpy as np
        X_match = np.vstack([fb.match_features(match_stats) for _ in range(30)])
        y_result = ["HOME"] * 15 + ["DRAW"] * 5 + ["AWAY"] * 10
        engine.football.fit_result(X_match, y_result)

        y_goals = [2.5] * 30
        engine.football.fit_goals(X_match, y_goals)

        # 4) Joueur optionnel
        player_stats = None
        if player_name and player_team:
            # choisir l'équipe correspondante
            if player_team.lower() == match_stats.home.name.lower():
                base_team = match_stats.home
            else:
                base_team = match_stats.away

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

            X_player = np.vstack([fb.player_features(player_stats, base_team) for _ in range(50)])
            y_scored = [1] * 30 + [0] * 20
            engine.football.fit_player_scoring(X_player, y_scored)

        # 5) Analyse
        result = engine.analyze_football(match_stats, player_stats)

        # 6) Mise en forme pour Telegram
        winner = result["winner"]
        scorelines = result["scoreline_candidates"]
        markets_goals = result["markets"]["goals"]

        text = "✅ Analyse terminée :\n\n"

        # Vainqueur
        text += (
            f"🏆 Vainqueur probable : {winner['team_name']} "
            f"({winner['label']})\n"
            f"   - HOME : {winner['probabilities']['HOME']:.2f}\n"
            f"   - DRAW : {winner['probabilities']['DRAW']:.2f}\n"
            f"   - AWAY : {winner['probabilities']['AWAY']:.2f}\n\n"
        )

        # Score exact (2 candidats)
        text += "🎯 Scores exacts les plus probables :\n"
        for s in scorelines:
            text += f"   - {s['score']} (p ~ {s['probability']:.2f})\n"
        text += "\n"

        # Over/Under buts
        text += "⚽ Nombre de buts (ligne principale) :\n"
        for line_str, choice in markets_goals.items():
            line_clean = line_str.replace("_", ".")
            text += f"   - {line_clean} : {choice}\n"
        text += "\n"

        # Joueur
        if result["player_scoring"] is not None:
            ps = result["player_scoring"]
            text += (
                f"👤 Buteur potentiel : {ps['player']} ({ps['team']})\n"
                f"   → Probabilité (modèle démo) : {ps['probability_to_score']:.2f}\n\n"
            )

        text += "ℹ️ Rappel : analyse purement informative, aucune garantie de résultat."

        await msg.reply_text(text)

    except Exception as e:
        await msg.reply_text(f"❌ Erreur pendant l'analyse : {e}")

def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN manquant dans .env")

    # Création de l'application Telegram (python-telegram-bot v22)
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Déclaration des handlers de commandes
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("analyze_foot", analyze_foot))

    print("Bot Telegram démarré. Appuie sur CTRL+C pour arrêter.")
    app.run_polling()


if __name__ == "__main__":
    main()
