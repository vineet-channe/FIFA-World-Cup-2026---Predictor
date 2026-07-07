import type { Metadata } from "next";
import "./globals.css";
import { Nav } from "@/components/ui/nav";
import { Footer } from "@/components/ui/footer";
import { LiveStatusBar } from "@/components/ui/live-status-bar";

export const metadata: Metadata = {
  title: "WC 2026 Predictor",
  description: "FIFA World Cup 2026 ML predictions — 10,000 Monte Carlo simulations",
};

export const dynamic = "force-dynamic";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        {/* Three-font stack loaded from Google Fonts CDN */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Big+Shoulders+Display:wght@700;800;900&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body
        className="flex flex-col"
        style={{
          minHeight: "100dvh",
          backgroundColor: "var(--ink)",
          color: "var(--chalk)",
          fontFamily: "var(--font-body)",
        }}
      >
        <Nav />
        <LiveStatusBar />
        {/* pb-20 reserves space for mobile bottom nav */}
        <main className="flex-1 pb-20 md:pb-0">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
