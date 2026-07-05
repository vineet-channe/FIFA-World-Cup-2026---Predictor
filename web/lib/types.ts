// Mirror of Python Pydantic schemas

export interface TeamSummary {
  name: string;
  iso_code: string | null;
  kit_color: string;
  elo: number;
  fifa_points: number;
  squad_value_eur: number;
  wc_appearances: number;
}

export interface TeamProbabilities {
  p_champion: number;
  p_final: number;
  p_semi: number;
  p_quarter: number;
  p_r16: number;
  p_r32: number;
  p_advance_groups: number;
  avg_goals_for_pg: number;
  avg_goals_against_pg: number;
}

export interface RadarStats {
  elo_strength: number;
  recent_form: number;
  h2h_dominance: number;
  squad_value: number;
  tournament_experience: number;
  advance_probability: number;
}

export interface FormMatch {
  opponent: string;
  result: "W" | "D" | "L";
  score: string;
  date: string;
  competition: string;
}

export interface TournamentPath {
  advance_groups: number;
  r16: number;
  quarter: number;
  semi: number;
  final: number;
  champion: number;
}

export interface TeamDetail {
  summary: TeamSummary;
  probabilities: TeamProbabilities;
  radar_stats: RadarStats;
  form_timeline: FormMatch[];
  tournament_path: TournamentPath;
}

export interface MatchPrediction {
  match_id: string;
  team_a: string;
  team_b: string;
  round: string;
  group: string | null;
  match_date: string;
  p_team_a_win: number;
  p_draw: number;
  p_team_b_win: number;
  expected_score_a: number;
  expected_score_b: number;
}

export interface GroupData {
  letter: string;
  teams: string[];
  advance_probs: Record<string, number>;
  matches: MatchPrediction[];
}

export interface MetaInfo {
  model_version: string;
  last_simulation: string;
  n_simulations: number;
  ensemble_brier_wc2022: number | null;
  current_round?: string | null;
  matches_played?: number | null;
  matches_total?: number | null;
  last_updated?: string | null;
  is_stale: boolean;
  run_status: "ok" | "failed" | "running" | "unknown";
}

export interface TopFeature {
  name: string;
  value: number;
  favors: "a" | "b" | "neutral";
}

export interface PredictResponse {
  p_team_a_win: number;
  p_draw: number;
  p_team_b_win: number;
  expected_score_a: number;
  expected_score_b: number;
  top_features: TopFeature[];
}

export interface H2HMatch {
  date: string;
  competition: string;
  score_a: number;
  score_b: number;
}

export interface ModelComparison {
  naive_baseline_brier?: number;
  logistic_wc2018_brier?: number;
  random_forest_wc2018_brier?: number;
  xgboost_wc2018_brier?: number;
  lightgbm_wc2018_brier?: number;
  dixon_coles_wc2018_brier?: number;
  neural_net_wc2018_brier?: number;
  ensemble_wc2018_brier?: number;
  ensemble_wc2022_brier?: number;
  trained_at?: string;
  [key: string]: number | string | boolean | undefined;
}

export interface MatchResultCard {
  match_id: string;
  team_a: string;
  team_b: string;
  round: string;
  group: string | null;
  match_date: string | null;
  status: string;
  actual_score_a: number | null;
  actual_score_b: number | null;
  actual_winner: string | null;
  p_team_a_win: number | null;
  p_draw: number | null;
  p_team_b_win: number | null;
  expected_score_a: number | null;
  expected_score_b: number | null;
  model_correct: boolean | null;
  has_pre_match_prediction?: boolean;
}

export interface RoundData {
  round_name: string;
  round_key: string;
  matches: MatchResultCard[];
  completed: number;
  total: number;
}

export interface TournamentBracket {
  current_round: string;
  last_updated: string;
  rounds: RoundData[];
}

export interface MatchPredictionAtStage {
  stage: string;
  saved_at: string;
  p_team_a_win: number;
  p_draw: number;
  p_team_b_win: number;
  expected_score_a: number;
  expected_score_b: number;
}

export interface MatchPredictionHistory {
  team_a: string;
  team_b: string;
  actual_score_a: number | null;
  actual_score_b: number | null;
  actual_winner: string | null;
  played: boolean;
  predictions: MatchPredictionAtStage[];
}

export interface AccuracyStats {
  total_played: number;
  correct_outcome: number;
  correct_pct: number;
  current_brier: number | null;
  brier_by_round: Record<string, number>;
  correct_by_round: Record<string, number>;
}

// ── Story page types ──────────────────────────────────────────────────────────

export interface StageMeta {
  stage: string;
  display: string;
  display_short: string;
  display_long: string;
  saved_at: string;
  teams_alive: number;
  matches_completed: number;
  reconstructed: boolean;
  model_version: string;
}

export interface TeamStagePoint {
  stage: string;
  p_champion: number;
  p_final: number;
  p_semi: number;
  eliminated: boolean;
  eliminated_in?: string | null;
}

export interface StoryEvolution {
  stages: StageMeta[];
  teams: Record<string, TeamStagePoint[]>;
}

export interface MoverRecord {
  wins: number;
  draws: number;
  losses: number;
  gf: number;
  ga: number;
  matches: {
    opponent: string;
    score: string;
    result: "W" | "D" | "L";
    round: string;
  }[];
}

export interface MoverCard {
  team: string;
  before: number;
  after: number;
  delta: number;
  eliminated_in?: string | null;
  record: MoverRecord;
  elo_change: number | null;
}

export interface StoryMovers {
  from_stage: string;
  to_stage: string;
  from_display: string;
  from_display_short?: string;
  from_display_long?: string;
  to_display: string;
  to_display_short?: string;
  to_display_long?: string;
  new_results_count: number;
  risers: MoverCard[];
  fallers: MoverCard[];
}

export interface Upset {
  team_a: string;
  team_b: string;
  score: string;
  round: string;
  model_favoured: string;
  confidence: number;
  actual: string;
}

export interface StageAccuracy {
  stage: string;
  display: string;
  display_short?: string;
  display_long?: string;
  predicts_round: string;
  reconstructed?: boolean;
  n_matches: number;
  correct?: number;
  correct_pct?: number | null;
  brier?: number | null;
  upsets?: Upset[];
  status?: string;
}

export interface CalibrationBucket {
  bucket: string;
  midpoint: number;
  n: number;
  predicted_rate: number;
  actual_rate: number;
}

export interface StoryAccuracy {
  per_stage: StageAccuracy[];
  calibration: CalibrationBucket[];
  overall: {
    n_matches: number;
    correct_pct: number | null;
    brier: number | null;
    wc2022_benchmark_brier: number;
  };
}

export interface FeatureAttribution {
  feature: string;
  display: string;
  description: string;
  value: number;
  favours: "a" | "b" | "neutral";
  normalised_magnitude?: number;
}

export interface MatchExplanation {
  team_a: string;
  team_b: string;
  stage: string;
  stage_display: string;
  display?: string;
  display_short?: string;
  display_long?: string;
  cut_date: string;
  context: {
    team_a_elo: number | null;
    team_b_elo: number | null;
    h2h_meetings: number | null;
    round_importance: number | null;
  };
  summary: string;
  features: FeatureAttribution[];
  recorded_prediction: {
    p_team_a_win: number;
    p_draw: number;
    p_team_b_win: number;
    expected_score_a: number;
    expected_score_b: number;
  } | null;
  actual_result: {
    score_a: number;
    score_b: number;
    winner: string | null;
    round: string;
  } | null;
  reconstructed: boolean;
  notes: string;
}
