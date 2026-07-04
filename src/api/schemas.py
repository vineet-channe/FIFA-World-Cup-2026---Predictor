"""Pydantic response models for the WC 2026 Predictor API."""

from __future__ import annotations

from pydantic import BaseModel


class TeamSummary(BaseModel):
    name: str
    iso_code: str | None
    kit_color: str
    elo: float
    fifa_points: float
    squad_value_eur: float
    wc_appearances: int


class TeamProbabilities(BaseModel):
    p_champion: float
    p_final: float
    p_semi: float
    p_quarter: float
    p_r16: float
    p_r32: float
    p_advance_groups: float
    avg_goals_for_pg: float
    avg_goals_against_pg: float


class RadarStats(BaseModel):
    elo_strength: float
    recent_form: float
    h2h_dominance: float
    squad_value: float
    tournament_experience: float
    advance_probability: float


class FormMatch(BaseModel):
    opponent: str
    result: str        # "W" | "D" | "L"
    score: str         # e.g. "2-1"
    date: str
    competition: str


class TournamentPath(BaseModel):
    advance_groups: float
    r16: float
    quarter: float
    semi: float
    final: float
    champion: float


class TeamDetail(BaseModel):
    summary: TeamSummary
    probabilities: TeamProbabilities
    radar_stats: RadarStats
    form_timeline: list[FormMatch]
    tournament_path: TournamentPath


class MatchPrediction(BaseModel):
    match_id: str
    team_a: str
    team_b: str
    round: str
    group: str | None
    match_date: str
    p_team_a_win: float
    p_draw: float
    p_team_b_win: float
    expected_score_a: float
    expected_score_b: float


class GroupData(BaseModel):
    letter: str
    teams: list[str]
    advance_probs: dict[str, float]
    matches: list[MatchPrediction]


class SimulationData(BaseModel):
    team_probabilities: dict[str, TeamProbabilities]
    match_predictions: dict[str, MatchPrediction]
    group_predictions: dict[str, dict]


class MetaInfo(BaseModel):
    model_version: str
    last_simulation: str
    n_simulations: int
    ensemble_brier_wc2022: float | None
    current_round: str | None = None
    matches_played: int | None = None
    matches_total: int | None = None
    last_updated: str | None = None


class MatchResultCard(BaseModel):
    match_id: str
    team_a: str
    team_b: str
    round: str
    group: str | None
    match_date: str | None
    status: str                   # "completed" | "predicted" | "scheduled"
    actual_score_a: int | None
    actual_score_b: int | None
    actual_winner: str | None
    p_team_a_win: float | None = None
    p_draw: float | None = None
    p_team_b_win: float | None = None
    expected_score_a: float | None = None
    expected_score_b: float | None = None
    model_correct: bool | None
    has_pre_match_prediction: bool = False


class RoundData(BaseModel):
    round_name: str
    round_key: str
    matches: list[MatchResultCard]
    completed: int
    total: int


class TournamentBracket(BaseModel):
    current_round: str
    last_updated: str
    rounds: list[RoundData]


class MatchPredictionAtStage(BaseModel):
    stage: str
    saved_at: str
    p_team_a_win: float
    p_draw: float
    p_team_b_win: float
    expected_score_a: float
    expected_score_b: float


class MatchPredictionHistory(BaseModel):
    team_a: str
    team_b: str
    actual_score_a: int | None
    actual_score_b: int | None
    actual_winner: str | None
    played: bool
    predictions: list[MatchPredictionAtStage]


class AccuracyStats(BaseModel):
    total_played: int
    correct_outcome: int
    correct_pct: float
    current_brier: float | None
    brier_by_round: dict[str, float]
    correct_by_round: dict[str, float]


class PredictRequest(BaseModel):
    team_a: str
    team_b: str


class TopFeature(BaseModel):
    name: str
    value: float
    favors: str    # "a" | "b" | "neutral"


class PredictResponse(BaseModel):
    p_team_a_win: float
    p_draw: float
    p_team_b_win: float
    expected_score_a: float
    expected_score_b: float
    top_features: list[TopFeature]


class H2HMatch(BaseModel):
    date: str
    competition: str
    score_a: int
    score_b: int
