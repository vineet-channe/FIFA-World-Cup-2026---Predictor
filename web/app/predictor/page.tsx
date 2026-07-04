import { getTeams } from "@/lib/api";
import { Predictor } from "@/components/match-predictor/predictor";

export const dynamic = "force-dynamic";

export default async function PredictorPage() {
  const teams = await getTeams();
  return <Predictor teams={teams} />;
}
