import type {
  AccuracyStats,
  GroupData,
  H2HMatch,
  MatchExplanation,
  MatchPredictionHistory,
  MetaInfo,
  ModelComparison,
  PredictResponse,
  StoryAccuracy,
  StoryEvolution,
  StoryMovers,
  TeamDetail,
  TeamSummary,
  TournamentBracket,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    cache: "no-store",
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    throw new Error(`API ${path} → ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export async function getTeams(): Promise<TeamSummary[]> {
  return apiFetch<TeamSummary[]>("/api/teams");
}

export async function getTeam(name: string): Promise<TeamDetail> {
  return apiFetch<TeamDetail>(`/api/team/${encodeURIComponent(name)}`);
}

export async function getGroups(): Promise<GroupData[]> {
  return apiFetch<GroupData[]>("/api/groups");
}

export async function getMeta(): Promise<MetaInfo> {
  return apiFetch<MetaInfo>("/api/meta");
}

export async function predictMatchup(
  teamA: string,
  teamB: string
): Promise<PredictResponse> {
  return apiFetch<PredictResponse>("/api/predict", {
    method: "POST",
    body: JSON.stringify({ team_a: teamA, team_b: teamB }),
  });
}

export async function getH2H(teamA: string, teamB: string): Promise<H2HMatch[]> {
  return apiFetch<H2HMatch[]>(
    `/api/h2h?team_a=${encodeURIComponent(teamA)}&team_b=${encodeURIComponent(teamB)}`
  );
}

export async function getModelComparison(): Promise<ModelComparison> {
  return apiFetch<ModelComparison>("/api/model-comparison");
}

export interface ProbabilityHistoryEntry {
  stage: string;
  saved_at?: string;
  p_champion: number;
  p_final: number;
  p_semi: number;
  p_quarter: number;
  eliminated: boolean;
  eliminated_in?: string;
}

export async function getProbabilityHistory(team: string) {
  return apiFetch<{ team: string; history: ProbabilityHistoryEntry[] }>(
    `/api/probability-history/${encodeURIComponent(team)}`
  );
}

export async function getTournamentBracket(): Promise<TournamentBracket> {
  return apiFetch<TournamentBracket>("/api/tournament-bracket");
}

export async function getMatchPredictionHistory(
  teamA: string,
  teamB: string
): Promise<MatchPredictionHistory> {
  const url = `${BASE}/api/match-prediction-history?team_a=${encodeURIComponent(teamA)}&team_b=${encodeURIComponent(teamB)}`;
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`No prediction history for ${teamA} vs ${teamB}`);
  return r.json() as Promise<MatchPredictionHistory>;
}

export async function getAccuracy(): Promise<AccuracyStats> {
  return apiFetch<AccuracyStats>("/api/accuracy");
}

export async function getStoryEvolution(): Promise<StoryEvolution> {
  return apiFetch<StoryEvolution>("/api/story/evolution");
}

export async function getStoryMovers(
  fromStage: string,
  toStage: string
): Promise<StoryMovers> {
  return apiFetch<StoryMovers>(
    `/api/story/movers?from_stage=${encodeURIComponent(fromStage)}&to_stage=${encodeURIComponent(toStage)}`
  );
}

export async function getStoryAccuracy(): Promise<StoryAccuracy> {
  return apiFetch<StoryAccuracy>("/api/story/accuracy");
}

export async function getMatchExplanation(
  teamA: string,
  teamB: string,
  stage: string
): Promise<MatchExplanation> {
  return apiFetch<MatchExplanation>(
    `/api/story/match-explanation?team_a=${encodeURIComponent(teamA)}&team_b=${encodeURIComponent(teamB)}&stage=${encodeURIComponent(stage)}`
  );
}
