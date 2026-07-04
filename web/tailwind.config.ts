// Tailwind v4 uses CSS-first configuration (@theme in globals.css).
// This file is kept as a stub for tooling compatibility.
import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
};

export default config;
