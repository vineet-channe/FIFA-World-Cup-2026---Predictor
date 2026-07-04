import { getTeams } from "@/lib/api";
import { TeamProfile } from "@/components/team-profile/profile";

export const dynamic = "force-dynamic";

export default async function TeamsPage() {
  const teams = await getTeams();
  return <TeamProfile teams={teams} />;
}
