interface RoundPendingProps {
  roundName: string;
  reason?: string;
}

export function RoundPending({ roundName, reason }: RoundPendingProps) {
  return (
    <div
      style={{
        display: "flex", flexDirection: "column", alignItems: "center",
        justifyContent: "center", padding: "48px 24px", textAlign: "center",
        background: "var(--ink-raised)", border: "0.5px solid var(--line)",
        borderRadius: 12,
      }}
    >
      <div style={{ fontSize: 28, marginBottom: 8, opacity: 0.3 }}>⚽</div>
      <div style={{ fontSize: 14, fontWeight: 500, color: "var(--chalk)",
                    marginBottom: 4, fontFamily: "var(--font-body)" }}>
        {roundName} hasn&apos;t finished yet
      </div>
      <div style={{ fontSize: 12, color: "rgba(245,243,236,0.4)",
                    fontFamily: "var(--font-body)", maxWidth: 320 }}>
        {reason ?? "Check back after the final whistle — predictions and results for this round will appear here automatically once the pipeline picks up the results."}
      </div>
    </div>
  );
}
