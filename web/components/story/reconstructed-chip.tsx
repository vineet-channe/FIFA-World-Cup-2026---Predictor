export function ReconstructedChip({ className = "" }: { className?: string }) {
  return (
    <span
      className={`inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono uppercase tracking-wide bg-[var(--amber)]/20 text-[var(--amber)] border border-[var(--amber)]/40 ${className}`}
      title="Rebuilt from real group-stage results after a data-source fix. Simulation state reflects the end-of-groups boundary."
    >
      reconstructed
    </span>
  );
}
