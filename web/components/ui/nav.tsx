"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { clsx } from "clsx";

const NAV_ITEMS = [
  { href: "/",           label: "Favourites" },
  { href: "/tournament", label: "Tournament" },
  { href: "/predictor",  label: "Predictor" },
  { href: "/teams",      label: "Teams" },
  { href: "/story",      label: "Story" },
  { href: "/lab",        label: "Lab" },
];

export function Nav() {
  const pathname = usePathname();

  return (
    <>
      {/* ── Mobile top brand bar ── */}
      <div
        className="md:hidden flex items-center gap-2.5 px-4 h-12 border-b bg-[var(--ink)]"
        style={{ borderColor: "var(--line)" }}
      >
        <Link href="/" className="flex items-center gap-2.5 min-w-0">
          <Image
            src="/fifa-wc2026-logo-white.png"
            alt="FIFA World Cup 2026"
            width={24}
            height={40}
            className="h-8 w-auto shrink-0"
            priority
          />
          <span className="font-display text-[11px] font-bold tracking-wide text-[var(--chalk)] truncate">
            FIFA WORLD CUP 2026
          </span>
        </Link>
      </div>

      {/* ── Desktop sticky header ── */}
      <header className="hidden md:flex sticky top-0 z-50 items-center justify-between px-6 h-14 border-b bg-[var(--ink)]"
        style={{ borderColor: "var(--line)" }}>
        <Link href="/" className="flex items-center gap-3 shrink-0">
          <Image
            src="/fifa-wc2026-logo-white.png"
            alt="FIFA World Cup 2026"
            width={32}
            height={52}
            className="h-10 w-auto"
            priority
          />
          <span
            className="font-display text-sm font-bold tracking-wide text-[var(--chalk)] leading-tight"
            style={{ letterSpacing: "0.06em" }}
          >
            FIFA WORLD CUP 2026
          </span>
        </Link>
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
        {NAV_ITEMS.map(({ href, label }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                "flex-1 flex items-center justify-center px-0.5 py-2.5 text-[10px] leading-tight text-center font-medium transition-colors duration-150",
                active
                  ? "text-[var(--turf-bright)]"
                  : "text-[var(--chalk)] opacity-50"
              )}
            >
              {label}
            </Link>
          );
        })}
      </nav>
    </>
  );
}
