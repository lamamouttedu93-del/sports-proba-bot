from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Literal
import numpy as np
from math import exp, factorial
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


# ============================================================
# 1. DATACLASSES PAR SPORT
# ============================================================

# ---------- FOOTBALL ----------

@dataclass
class FootballTeamStats:
    name: str
    goals_for_avg: float
    goals_against_avg: float
    shots_for_avg: float
    shots_on_target_for_avg: float
    corners_for_avg: float
    cards_for_avg: float
    touches_for_avg: Optional[float] = None
    win_rate: float = 0.0
    draw_rate: float = 0.0
    loss_rate: float = 0.0


@dataclass
class FootballMatchStats:
    home: FootballTeamStats
    away: FootballTeamStats


@dataclass
class FootballPlayerStats:
    name: str
    team_name: str
    goals_per90: float
    assists_per90: float
    shots_per90: float
    xg_per90: Optional[float] = None
    xa_per90: Optional[float] = None
    minutes_last5: int = 0
    is_injured: bool = False


# ---------- HOCKEY ----------

@dataclass
class HockeyTeamStats:
    name: str
    goals_for_avg: float    # OT inclus
    goals_against_avg: float
    shots_for_avg: float
    win_rate: float
    loss_rate: float


@dataclass
class HockeyMatchStats:
    home: HockeyTeamStats
    away: HockeyTeamStats


@dataclass
class HockeyPlayerStats:
    name: str
    team_name: str
    points_per_game: float   # buts + assists par match
    goals_per_game: float
    assists_per_game: float
    is_injured: bool = False


# ---------- TENNIS ----------

@dataclass
class TennisPlayerStats:
    name: str
    aces_per_set_avg: float
    aces_per_match_avg: float
    first_serve_pct: float
    hold_pct: float          # % de jeux de service remportés (0-1)
    return_game_win_pct: float
    tiebreak_tendency: float # proportion de sets allant au TB (0-1)
    is_server_dominant: bool


@dataclass
class TennisMatchStats:
    player1: TennisPlayerStats
    player2: TennisPlayerStats
    best_of_sets: int        # 3 ou 5


# ============================================================
# 2. UTILITAIRES PROBA (Poisson & Over/Under)
# ============================================================

def poisson_prob(lmbda: float, k: int) -> float:
    return (lmbda ** k) * exp(-lmbda) / factorial(k)


def poisson_distribution(lmbda: float, max_k: int = 10) -> Dict[int, float]:
    probs = {k: poisson_prob(lmbda, k) for k in range(0, max_k + 1)}
    s = sum(probs.values())
    if s == 0:
        return probs
    for k in probs:
        probs[k] /= s
    return probs


def choose_over_under_from_expectation(expectation: float, line: float) -> str:
    return "OVER" if expectation > line else "UNDER"


# ============================================================
# 3. FEATURE BUILDERS
# ============================================================

class FootballFeatureBuilder:
    """
    Construit les vecteurs de features à partir des stats agrégées.
    Utilisé à la fois pour l'entraînement et pour la prédiction.
    """

    @staticmethod
    def match_features(match: FootballMatchStats) -> np.ndarray:
        h = match.home
        a = match.away
        home_vec = np.array([
            h.goals_for_avg,
            h.goals_against_avg,
            h.shots_for_avg,
            h.shots_on_target_for_avg,
            h.corners_for_avg,
            h.cards_for_avg,
            h.touches_for_avg if h.touches_for_avg is not None else 0.0,
            h.win_rate,
            h.draw_rate,
            h.loss_rate,
        ], dtype=float)

        away_vec = np.array([
            a.goals_for_avg,
            a.goals_against_avg,
            a.shots_for_avg,
            a.shots_on_target_for_avg,
            a.corners_for_avg,
            a.cards_for_avg,
            a.touches_for_avg if a.touches_for_avg is not None else 0.0,
            a.win_rate,
            a.draw_rate,
            a.loss_rate,
        ], dtype=float)

        diff_vec = home_vec - away_vec
        return np.concatenate([home_vec, away_vec, diff_vec], axis=0)

    @staticmethod
    def player_features(player: FootballPlayerStats, team: FootballTeamStats) -> np.ndarray:
        return np.array([
            player.goals_per90,
            player.assists_per90,
            player.shots_per90,
            player.xg_per90 if player.xg_per90 is not None else player.goals_per90,
            player.xa_per90 if player.xa_per90 is not None else player.assists_per90,
            player.minutes_last5,
            1.0 if player.is_injured else 0.0,
            team.goals_for_avg,
            team.shots_for_avg,
            team.win_rate
        ], dtype=float)


class HockeyFeatureBuilder:
    @staticmethod
    def match_features(match: HockeyMatchStats) -> np.ndarray:
        h = match.home
        a = match.away
        home_vec = np.array([
            h.goals_for_avg,
            h.goals_against_avg,
            h.shots_for_avg,
            h.win_rate,
            h.loss_rate,
        ], dtype=float)
        away_vec = np.array([
            a.goals_for_avg,
            a.goals_against_avg,
            a.shots_for_avg,
            a.win_rate,
            a.loss_rate,
        ], dtype=float)
        diff_vec = home_vec - away_vec
        return np.concatenate([home_vec, away_vec, diff_vec], axis=0)

    @staticmethod
    def player_features(player: HockeyPlayerStats, team: HockeyTeamStats) -> np.ndarray:
        return np.array([
            player.points_per_game,
            player.goals_per_game,
            player.assists_per_game,
            1.0 if player.is_injured else 0.0,
            team.goals_for_avg,
            team.win_rate
        ], dtype=float)


class TennisFeatureBuilder:
    @staticmethod
    def match_features(match: TennisMatchStats) -> np.ndarray:
        p1 = match.player1
        p2 = match.player2
        p1_vec = np.array([
            p1.aces_per_set_avg,
            p1.aces_per_match_avg,
            p1.first_serve_pct,
            p1.hold_pct,
            p1.return_game_win_pct,
            p1.tiebreak_tendency,
            1.0 if p1.is_server_dominant else 0.0
        ], dtype=float)
        p2_vec = np.array([
            p2.aces_per_set_avg,
            p2.aces_per_match_avg,
            p2.first_serve_pct,
            p2.hold_pct,
            p2.return_game_win_pct,
            p2.tiebreak_tendency,
            1.0 if p2.is_server_dominant else 0.0
        ], dtype=float)
        diff_vec = p1_vec - p2_vec
        return np.concatenate([p1_vec, p2_vec, diff_vec, np.array([float(match.best_of_sets)])], axis=0)

    @staticmethod
    def aces_features(match: TennisMatchStats, player_index: int) -> np.ndarray:
        # player_index: 1 ou 2
        p = match.player1 if player_index == 1 else match.player2
        opponent = match.player2 if player_index == 1 else match.player1
        return np.array([
            p.aces_per_match_avg,
            p.first_serve_pct,
            p.hold_pct,
            opponent.return_game_win_pct,
            float(match.best_of_sets)
        ], dtype=float)


# ============================================================
# 4. MODELES PAR SPORT (KNN & SIMILAIRES)
# ============================================================

WinnerLabel = Literal["HOME", "DRAW", "AWAY"]


class FootballModels:
    """
    Regroupe plusieurs modèles pour le foot :
    - résultat du match (HOME/DRAW/AWAY)
    - total de buts attendu (régression)
    - probabilité qu'un joueur marque
    """

    def __init__(self, n_neighbors_class: int = 21, n_neighbors_reg: int = 25):
        self.fb = FootballFeatureBuilder()
        self.result_model: Optional[Pipeline] = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors_class, weights="distance"))
        ])
        self.goals_model: Optional[Pipeline] = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=n_neighbors_reg, weights="distance"))
        ])
        self.player_model: Optional[Pipeline] = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors_class, weights="distance"))
        ])

    # ---------- Entraînement ----------

    def fit_result(self, X: np.ndarray, y: List[WinnerLabel]):
        self.result_model.fit(X, y)

    def fit_goals(self, X: np.ndarray, y_total_goals: List[float]):
        self.goals_model.fit(X, y_total_goals)

    def fit_player_scoring(self, X: np.ndarray, y_scored: List[int]):
        self.player_model.fit(X, y_scored)

    # ---------- Features helpers ----------

    def build_match_row(self, match: FootballMatchStats) -> np.ndarray:
        return self.fb.match_features(match).reshape(1, -1)

    def build_player_row(self, player: FootballPlayerStats, team: FootballTeamStats) -> np.ndarray:
        return self.fb.player_features(player, team).reshape(1, -1)

    # ---------- Prédictions ----------

    def predict_result_proba(self, match: FootballMatchStats) -> Dict[str, float]:
        if self.result_model is None:
            raise RuntimeError("result_model non entraîné.")
        X = self.build_match_row(match)
        proba = self.result_model.predict_proba(X)[0]
        classes = self.result_model.named_steps["knn"].classes_.tolist()
        out = {}
        for label in ["HOME", "DRAW", "AWAY"]:
            out[label] = float(proba[classes.index(label)]) if label in classes else 0.0
        return out

    def predict_expected_goals(self, match: FootballMatchStats) -> float:
        if self.goals_model is None:
            raise RuntimeError("goals_model non entraîné.")
        X = self.build_match_row(match)
        y = self.goals_model.predict(X)[0]
        return float(max(0.1, y))

    def predict_player_scoring_proba(self, player: FootballPlayerStats, team: FootballTeamStats) -> float:
        if self.player_model is None:
            raise RuntimeError("player_model non entraîné.")
        X = self.build_player_row(player, team)
        proba = self.player_model.predict_proba(X)[0]
        classes = self.player_model.named_steps["knn"].classes_.tolist()
        # proba qu'il marque (classe 1)
        if 1 in classes:
            return float(proba[classes.index(1)])
        return 0.0

    # ---------- Logique combinée pour l'analyse ----------

    def suggest_scorelines(self, match: FootballMatchStats, expected_goals: float, result_probs: Dict[str, float]) -> List[Dict[str, Any]]:
        # Répartition des buts selon les forces offensives
        home_ratio = match.home.goals_for_avg / max(match.home.goals_for_avg + match.away.goals_for_avg, 0.1)
        lambda_home = expected_goals * home_ratio
        lambda_away = expected_goals * (1 - home_ratio)
        scores = []
        for h in range(0, 7):
            for a in range(0, 7):
                p_h = poisson_prob(lambda_home, h)
                p_a = poisson_prob(lambda_away, a)
                base_p = p_h * p_a
                if h > a:
                    mult = 0.7 + 0.3 * result_probs["HOME"]
                elif h < a:
                    mult = 0.7 + 0.3 * result_probs["AWAY"]
                else:
                    mult = 0.7 + 0.3 * result_probs["DRAW"]
                p = base_p * mult
                scores.append({"score": f"{h}-{a}", "probability": p})
        scores.sort(key=lambda x: x["probability"], reverse=True)
        top2 = scores[:2]
        total = sum(x["probability"] for x in top2) or 1.0
        for x in top2:
            x["probability"] = float(x["probability"] / total)
        return top2

    def over_under_markets(self, expected_goals: float,
                           match: FootballMatchStats) -> Dict[str, Dict[str, str]]:
        # expected_goals pour les buts, moyennes pour le reste
        exp_goals = expected_goals
        exp_corners = match.home.corners_for_avg + match.away.corners_for_avg
        exp_cards = match.home.cards_for_avg + match.away.cards_for_avg
        exp_shots = match.home.shots_for_avg + match.away.shots_for_avg
        exp_shots_on = match.home.shots_on_target_for_avg + match.away.shots_on_target_for_avg
        exp_touches = None
        if match.home.touches_for_avg is not None and match.away.touches_for_avg is not None:
            exp_touches = match.home.touches_for_avg + match.away.touches_for_avg

        lines_goals = [0.5, 1.5, 2.5, 3.5]
        lines_corners = [4.5, 6.5, 8.5, 10.5]
        lines_cards = [2.5, 3.5, 4.5]
        lines_shots = [15.5, 20.5, 25.5]
        lines_shots_on = [8.5, 10.5, 12.5]
        lines_touches = [400.5, 600.5, 800.5] if exp_touches is not None else []

        def build(lines, expectation):
            return {str(line).replace(".", "_"): choose_over_under_from_expectation(expectation, line) for line in lines}

        markets = {
            "goals": build(lines_goals, exp_goals),
            "corners": build(lines_corners, exp_corners),
            "cards": build(lines_cards, exp_cards),
            "shots": build(lines_shots, exp_shots),
            "shots_on_target": build(lines_shots_on, exp_shots_on),
        }
        if exp_touches is not None:
            markets["touches"] = build(lines_touches, exp_touches)
        return markets

    def analyze(self, match: FootballMatchStats,
                player: Optional[FootballPlayerStats] = None) -> Dict[str, Any]:
        result_probs = self.predict_result_proba(match)
        expected_goals = self.predict_expected_goals(match)
        scorelines = self.suggest_scorelines(match, expected_goals, result_probs)
        markets = self.over_under_markets(expected_goals, match)

        player_block = None
        if player is not None:
            team = match.home if player.team_name == match.home.name else match.away
            p_score = self.predict_player_scoring_proba(player, team)
            player_block = {
                "player": player.name,
                "team": player.team_name,
                "probability_to_score": p_score
            }

        # Choix explicite d'un vainqueur unique
        winner_label = max(result_probs, key=result_probs.get)
        if winner_label == "HOME":
            winner_name = match.home.name
        elif winner_label == "AWAY":
            winner_name = match.away.name
        else:
            winner_name = "DRAW"

        return {
            "sport": "football",
            "winner": {
                "label": winner_label,
                "team_name": winner_name,
                "probabilities": result_probs
            },
            "expected_total_goals": expected_goals,
            "scoreline_candidates": scorelines,
            "markets": markets,
            "player_scoring": player_block
        }


# ---------- HOCKEY MODELS ----------

class HockeyModels:
    """
    Pour le hockey :
    - vainqueur (OT inclus)
    - total de buts attendu
    - probabilité qu'un joueur ait au moins 1 point
    """

    def __init__(self, n_neighbors_class: int = 21, n_neighbors_reg: int = 25):
        self.fb = HockeyFeatureBuilder()
        self.result_model: Optional[Pipeline] = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors_class, weights="distance"))
        ])
        self.goals_model: Optional[Pipeline] = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=n_neighbors_reg, weights="distance"))
        ])
        self.player_model: Optional[Pipeline] = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors_class, weights="distance"))
        ])

    def fit_result(self, X: np.ndarray, y: List[Literal["HOME", "AWAY"]]):
        self.result_model.fit(X, y)

    def fit_goals(self, X: np.ndarray, y_total_goals: List[float]):
        self.goals_model.fit(X, y_total_goals)

    def fit_player_points(self, X: np.ndarray, y_has_point: List[int]):
        self.player_model.fit(X, y_has_point)

    def build_match_row(self, match: HockeyMatchStats) -> np.ndarray:
        return self.fb.match_features(match).reshape(1, -1)

    def build_player_row(self, player: HockeyPlayerStats, team: HockeyTeamStats) -> np.ndarray:
        return self.fb.player_features(player, team).reshape(1, -1)

    def predict_result_proba(self, match: HockeyMatchStats) -> Dict[str, float]:
        if self.result_model is None:
            raise RuntimeError("result_model non entraîné.")
        X = self.build_match_row(match)
        proba = self.result_model.predict_proba(X)[0]
        classes = self.result_model.named_steps["knn"].classes_.tolist()
        out = {}
        for label in ["HOME", "AWAY"]:
            out[label] = float(proba[classes.index(label)]) if label in classes else 0.0
        return out

    def predict_expected_goals(self, match: HockeyMatchStats) -> float:
        if self.goals_model is None:
            raise RuntimeError("goals_model non entraîné.")
        X = self.build_match_row(match)
        y = self.goals_model.predict(X)[0]
        return float(max(0.1, y))

    def predict_player_points_proba(self, player: HockeyPlayerStats, team: HockeyTeamStats) -> float:
        if self.player_model is None:
            raise RuntimeError("player_model non entraîné.")
        X = self.build_player_row(player, team)
        proba = self.player_model.predict_proba(X)[0]
        classes = self.player_model.named_steps["knn"].classes_.tolist()
        if 1 in classes:
            return float(proba[classes.index(1)])
        return 0.0

    def goals_markets(self, expected_goals: float) -> Dict[str, str]:
        lines = [3.5, 4.5, 5.5, 6.5]
        return {str(l).replace(".", "_"): choose_over_under_from_expectation(expected_goals, l) for l in lines}

    def analyze(self, match: HockeyMatchStats,
                player: Optional[HockeyPlayerStats] = None) -> Dict[str, Any]:
        result_probs = self.predict_result_proba(match)
        expected_goals = self.predict_expected_goals(match)
        winner_label = "HOME" if result_probs["HOME"] >= result_probs["AWAY"] else "AWAY"
        winner_name = match.home.name if winner_label == "HOME" else match.away.name

        player_block = None
        if player is not None:
            team = match.home if player.team_name == match.home.name else match.away
            p_points = self.predict_player_points_proba(player, team)
            player_block = {
                "player": player.name,
                "team": player.team_name,
                "probability_1plus_point": p_points
            }

        return {
            "sport": "hockey",
            "winner_ot_included": {
                "label": winner_label,
                "team_name": winner_name,
                "probabilities": result_probs
            },
            "expected_total_goals": expected_goals,
            "goals_markets": self.goals_markets(expected_goals),
            "player_points": player_block
        }


# ---------- TENNIS MODELS ----------

class TennisModels:
    """
    Pour le tennis :
    - vainqueur du match
    - vainqueur du 1er set
    - nb de jeux (1er set + match)
    - 1 score exact
    - aces par joueur et total
    """

    def __init__(self, n_neighbors_class: int = 21, n_neighbors_reg: int = 25):
        self.fb = TennisFeatureBuilder()
        self.match_model: Optional[Pipeline] = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors_class, weights="distance"))
        ])
        self.first_set_model: Optional[Pipeline] = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors_class, weights="distance"))
        ])
        self.games_model: Optional[Pipeline] = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=n_neighbors_reg, weights="distance"))
        ])
        self.aces_model: Optional[Pipeline] = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=n_neighbors_reg, weights="distance"))
        ])

    def fit_match_winner(self, X: np.ndarray, y: List[Literal["PLAYER1", "PLAYER2"]]):
        self.match_model.fit(X, y)

    def fit_first_set_winner(self, X: np.ndarray, y: List[Literal["PLAYER1", "PLAYER2"]]):
        self.first_set_model.fit(X, y)

    def fit_games_total(self, X: np.ndarray, y_total_games: List[float]):
        self.games_model.fit(X, y_total_games)

    def fit_aces(self, X: np.ndarray, y_aces: List[float]):
        self.aces_model.fit(X, y_aces)

    def build_match_row(self, match: TennisMatchStats) -> np.ndarray:
        return self.fb.match_features(match).reshape(1, -1)

    def build_aces_row(self, match: TennisMatchStats, player_index: int) -> np.ndarray:
        return self.fb.aces_features(match, player_index).reshape(1, -1)

    def predict_match_winner_proba(self, match: TennisMatchStats) -> Dict[str, float]:
        if self.match_model is None:
            raise RuntimeError("match_model non entraîné.")
        X = self.build_match_row(match)
        proba = self.match_model.predict_proba(X)[0]
        classes = self.match_model.named_steps["knn"].classes_.tolist()
        out = {}
        for label in ["PLAYER1", "PLAYER2"]:
            out[label] = float(proba[classes.index(label)]) if label in classes else 0.0
        return out

    def predict_first_set_winner_proba(self, match: TennisMatchStats) -> Dict[str, float]:
        if self.first_set_model is None:
            raise RuntimeError("first_set_model non entraîné.")
        X = self.build_match_row(match)
        proba = self.first_set_model.predict_proba(X)[0]
        classes = self.first_set_model.named_steps["knn"].classes_.tolist()
        out = {}
        for label in ["PLAYER1", "PLAYER2"]:
            out[label] = float(proba[classes.index(label)]) if label in classes else 0.0
        return out

    def predict_expected_games_match(self, match: TennisMatchStats) -> float:
        if self.games_model is None:
            raise RuntimeError("games_model non entraîné.")
        X = self.build_match_row(match)
        g = self.games_model.predict(X)[0]
        return float(max(10.0, g))

    def predict_expected_aces(self, match: TennisMatchStats) -> Dict[str, float]:
        if self.aces_model is None:
            raise RuntimeError("aces_model non entraîné.")
        X1 = self.build_aces_row(match, 1)
        X2 = self.build_aces_row(match, 2)
        a1 = float(max(0.0, self.aces_model.predict(X1)[0]))
        a2 = float(max(0.0, self.aces_model.predict(X2)[0]))
        return {
            "player1_aces": a1,
            "player2_aces": a2,
            "total_aces": a1 + a2
        }

    def games_markets(self, expected_first_set_games: float,
                      expected_match_games: float) -> Dict[str, Dict[str, str]]:
        lines_first = [8.5, 9.5, 10.5, 11.5]
        lines_match = [20.5, 22.5, 24.5, 30.5]
        over_under_first = {str(l).replace(".", "_"): choose_over_under_from_expectation(expected_first_set_games, l)
                            for l in lines_first}
        over_under_match = {str(l).replace(".", "_"): choose_over_under_from_expectation(expected_match_games, l)
                            for l in lines_match}
        return {
            "first_set": over_under_first,
            "match": over_under_match
        }

    def choose_exact_score(self, match: TennisMatchStats,
                           win_label: str) -> str:
        # Ici : 1 seul score exact (logique simple)
        best_of = match.best_of_sets
        if best_of == 3:
            return "2-0" if win_label == "PLAYER1" else "0-2"
        else:
            return "3-0" if win_label == "PLAYER1" else "0-3"

    def analyze(self, match: TennisMatchStats) -> Dict[str, Any]:
        match_probs = self.predict_match_winner_proba(match)
        first_set_probs = self.predict_first_set_winner_proba(match)

        match_winner_label = "PLAYER1" if match_probs["PLAYER1"] >= match_probs["PLAYER2"] else "PLAYER2"
        match_winner_name = match.player1.name if match_winner_label == "PLAYER1" else match.player2.name

        first_set_label = "PLAYER1" if first_set_probs["PLAYER1"] >= first_set_probs["PLAYER2"] else "PLAYER2"
        first_set_winner_name = match.player1.name if first_set_label == "PLAYER1" else match.player2.name

        expected_match_games = self.predict_expected_games_match(match)
        # approx pour le 1er set : fraction du match
        expected_first_set_games = max(6.0, min(13.0, expected_match_games / (match.best_of_sets - 0.5)))
        games_markets = self.games_markets(expected_first_set_games, expected_match_games)

        aces_info = self.predict_expected_aces(match)
        total_aces_lines = [10.5, 15.5, 20.5]
        aces_over_under_total = {
            str(l).replace(".", "_"): choose_over_under_from_expectation(aces_info["total_aces"], l)
            for l in total_aces_lines
        }

        exact_score = self.choose_exact_score(match, match_winner_label)

        return {
            "sport": "tennis",
            "match_winner": {
                "label": match_winner_label,
                "player_name": match_winner_name,
                "probabilities": match_probs
            },
            "first_set_winner": {
                "label": first_set_label,
                "player_name": first_set_winner_name,
                "probabilities": first_set_probs
            },
            "games": {
                "expected_first_set_games": expected_first_set_games,
                "expected_match_games": expected_match_games,
                "markets": games_markets
            },
            "exact_match_score": exact_score,
            "aces": {
                "expected_player1_aces": aces_info["player1_aces"],
                "expected_player2_aces": aces_info["player2_aces"],
                "expected_total_aces": aces_info["total_aces"],
                "over_under_total": aces_over_under_total
            }
        }


# ============================================================
# 5. ORCHESTRATEUR MULTI-SPORT
# ============================================================

class MultiSportModels:
    """
    Orchestrateur : tu instancies un seul objet,
    tu entraînes chaque bloc séparément avec tes historiques réels,
    puis tu appelles analyze_football / analyze_hockey / analyze_tennis
    à partir des stats agrégées d'un match.
    """

    def __init__(self):
        self.football = FootballModels()
        self.hockey = HockeyModels()
        self.tennis = TennisModels()

    # Foot
    def analyze_football(self, match: FootballMatchStats,
                         player: Optional[FootballPlayerStats] = None) -> Dict[str, Any]:
        return self.football.analyze(match, player)

    # Hockey
    def analyze_hockey(self, match: HockeyMatchStats,
                       player: Optional[HockeyPlayerStats] = None) -> Dict[str, Any]:
        return self.hockey.analyze(match, player)

    # Tennis
    def analyze_tennis(self, match: TennisMatchStats) -> Dict[str, Any]:
        return self.tennis.analyze(match)


if __name__ == "__main__":
    # Demo ultra simplifiée avec des stats fictives juste pour vérifier que ça tourne.
    home = FootballTeamStats(
        name="Team A",
        goals_for_avg=2.1,
        goals_against_avg=1.0,
        shots_for_avg=13,
        shots_on_target_for_avg=6,
        corners_for_avg=6,
        cards_for_avg=2,
        touches_for_avg=500,
        win_rate=0.6,
        draw_rate=0.2,
        loss_rate=0.2
    )
    away = FootballTeamStats(
        name="Team B",
        goals_for_avg=1.2,
        goals_against_avg=1.5,
        shots_for_avg=10,
        shots_on_target_for_avg=4,
        corners_for_avg=4,
        cards_for_avg=3,
        touches_for_avg=480,
        win_rate=0.3,
        draw_rate=0.3,
        loss_rate=0.4
    )
    match = FootballMatchStats(home=home, away=away)
    player = FootballPlayerStats(
        name="Striker",
        team_name="Team A",
        goals_per90=0.6,
        assists_per90=0.2,
        shots_per90=3.5,
        xg_per90=0.7,
        xa_per90=0.3,
        minutes_last5=420,
        is_injured=False
    )

    # Création du moteur multi-sport
    engine = MultiSportModels()

    # Pour la démo, on entraîne les modèles avec des exemples fictifs
    fb = FootballFeatureBuilder()
    X_train_res = np.vstack([fb.match_features(match) for _ in range(30)])
    y_train_res = ["HOME"] * 30
    engine.football.fit_result(X_train_res, y_train_res)

    X_train_goals = X_train_res
    y_train_goals = [3.0] * 30
    engine.football.fit_goals(X_train_goals, y_train_goals)

    X_train_player = np.vstack([fb.player_features(player, home) for _ in range(50)])
    y_train_player = [1] * 50
    engine.football.fit_player_scoring(X_train_player, y_train_player)

    result = engine.analyze_football(match, player)
    print(result)
