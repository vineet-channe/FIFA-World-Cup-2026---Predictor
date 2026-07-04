"use client";

import { useEffect, useRef, useState } from "react";
import { useReducedMotion } from "framer-motion";

interface FlipNumberProps {
  value: number;
  format?: "percent" | "decimal" | "integer";
  decimals?: number;
  className?: string;
}

/**
 * Odometer-style count-up from 0 to target value.
 * Checks prefers-reduced-motion and skips animation if set.
 */
export function FlipNumber({
  value,
  format = "percent",
  decimals = 1,
  className = "",
}: FlipNumberProps) {
  const shouldReduce = useReducedMotion();
  const [displayed, setDisplayed] = useState(shouldReduce ? value : 0);
  const rafRef = useRef<number | null>(null);
  const startTimeRef = useRef<number | null>(null);
  const DURATION = 350; // ms

  useEffect(() => {
    if (shouldReduce) {
      setDisplayed(value);
      return;
    }

    const startValue = 0;
    const endValue = value;

    function easeOut(t: number): number {
      return 1 - Math.pow(1 - t, 3);
    }

    function tick(timestamp: number) {
      if (!startTimeRef.current) startTimeRef.current = timestamp;
      const elapsed = timestamp - startTimeRef.current;
      const progress = Math.min(elapsed / DURATION, 1);
      const current = startValue + (endValue - startValue) * easeOut(progress);
      setDisplayed(current);
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(tick);
      }
    }

    startTimeRef.current = null;
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [value, shouldReduce]);

  const formatted = formatValue(displayed, format, decimals);
  return <span className={className}>{formatted}</span>;
}

function formatValue(
  value: number,
  format: FlipNumberProps["format"],
  decimals: number
): string {
  switch (format) {
    case "percent":
      return `${(value * 100).toFixed(decimals)}%`;
    case "integer":
      return Math.round(value).toString();
    case "decimal":
    default:
      return value.toFixed(decimals);
  }
}
