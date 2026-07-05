import { getMeta } from "@/lib/api";

function formatUpdated(iso: string | null | undefined): string {
  if (!iso || iso === "unknown") return "—";
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso.slice(0, 16);
    return d.toLocaleString("en-GB", {
      month: "long", day: "numeric", year: "numeric",
      hour: "2-digit", minute: "2-digit",
      timeZone: "UTC", timeZoneName: "short",
    });
  } catch {
    return iso.slice(0, 16);
  }
}

export async function LiveStatusBar() {
  let meta;
  try {
    meta = await getMeta();
  } catch {
    return (
      <div
        className="flex items-center justify-center px-6 py-1.5 text-xs font-mono border-b"
        style={{ borderColor: "var(--line)", color: "var(--red)" }}
      >
        ⚠ Could not reach the prediction API
      </div>
    );
  }

  const played  = meta.matches_played ?? 0;
  const total   = meta.matches_total ?? 104;
  const round   = meta.current_round ?? "Pre-tournament";
  const updated = formatUpdated(meta.last_updated ?? meta.last_simulation);

  const showWarning = meta.is_stale || meta.run_status === "failed";
  const warningText =
    meta.run_status === "failed"
      ? "⚠ Last update attempt failed — showing most recent successful data"
      : meta.is_stale
      ? "⚠ Data may be stale — no update in over 18 hours"
      : null;

  return (
    <div
      className="hidden md:flex items-center justify-between px-6 py-1.5 text-xs font-mono border-b"
      style={{ borderColor: "var(--line)", color: "rgba(245,243,236,0.4)" }}
    >
      <span>⚡ Last updated: {updated}</span>
      <span>{round}</span>
      <span>{played} / {total} matches played</span>
      {showWarning && (
        <span style={{ color: "var(--amber)" }}>{warningText}</span>
      )}
    </div>
  );
}
