"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { clsx } from "clsx";

const NAV_ITEMS = [
  { href: "/",          label: "Favourites", icon: "🏆" },
  { href: "/tournament", label: "Tournament", icon: "⚽" },
  { href: "/predictor", label: "Predictor",  icon: "🔮" },
  { href: "/teams",     label: "Teams",      icon: "📊" },
  { href: "/lab",       label: "Lab",        icon: "🧪" },
];

export function Nav() {
  const pathname = usePathname();

  return (
    <>
      {/* ── Desktop sticky header ── */}
      <header className="hidden md:flex sticky top-0 z-50 items-center justify-between px-6 h-14 border-b bg-[var(--ink)]"
        style={{ borderColor: "var(--line)" }}>
        <span className="font-display text-xl font-bold tracking-tight text-[var(--chalk)]">
          ⚽ WC26
        </span>
        <nav className="flex gap-1">
          {NAV_ITEMS.map(({ href, label }) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={clsx(
                  "px-4 py-1.5 rounded text-sm font-medium transition-colors duration-150",
                  active
                    ? "bg-[var(--turf)] text-[var(--chalk)]"
                    : "text-[var(--chalk)] opacity-60 hover:opacity-100 hover:bg-[var(--ink-raised)]"
                )}
              >
                {label}
              </Link>
            );
          })}
        </nav>
      </header>

      {/* ── Mobile bottom tab bar ── */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 z-50 flex border-t bg-[var(--ink-raised)]"
        style={{ borderColor: "var(--line)" }}>
        {NAV_ITEMS.map(({ href, label, icon }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                "flex-1 flex flex-col items-center justify-center py-2 gap-0.5 text-xs transition-colors duration-150",
                active
                  ? "text-[var(--turf-bright)]"
                  : "text-[var(--chalk)] opacity-50"
              )}
            >
              <span className="text-lg leading-none">{icon}</span>
              <span className="font-medium">{label}</span>
            </Link>
          );
        })}
      </nav>
    </>
  );
}
