// ISO 3166-1 alpha-2 codes (lowercase for flag-icons CSS classes)
// and primary/away kit colours for all 48 WC 2026 teams.
// Kit colour is the away/alternate when the primary is too dark against --ink.

export const TEAM_ISO_CODES: Record<string, string> = {
  // Group A
  Mexico:               "mx",
  "South Africa":       "za",
  "South Korea":        "kr",
  Czechia:              "cz",
  // Group B
  USA:                  "us",
  Paraguay:             "py",
  Australia:            "au",
  Turkey:               "tr",
  // Group C
  Canada:               "ca",
  "Bosnia & Herzegovina": "ba",
  Qatar:                "qa",
  Switzerland:          "ch",
  // Group D
  Germany:              "de",
  "Curaçao":            "cw",
  "Ivory Coast":        "ci",
  Ecuador:              "ec",
  // Group E
  Netherlands:          "nl",
  Japan:                "jp",
  Sweden:               "se",
  Tunisia:              "tn",
  // Group F
  Brazil:               "br",
  Morocco:              "ma",
  Scotland:             "gb-sct",
  Haiti:                "ht",
  // Group G
  France:               "fr",
  Senegal:              "sn",
  Iraq:                 "iq",
  Norway:               "no",
  // Group H
  Spain:                "es",
  "Cape Verde":         "cv",
  "Saudi Arabia":       "sa",
  Uruguay:              "uy",
  // Group I
  Belgium:              "be",
  Egypt:                "eg",
  Iran:                 "ir",
  "New Zealand":        "nz",
  // Group J
  England:              "gb-eng",
  Croatia:              "hr",
  Ghana:                "gh",
  Panama:               "pa",
  // Group K
  Portugal:             "pt",
  "DR Congo":           "cd",
  Uzbekistan:           "uz",
  Colombia:             "co",
  // Group L
  Argentina:            "ar",
  Algeria:              "dz",
  Austria:              "at",
  Jordan:               "jo",
};

// Kit colours — away/alternate used when primary is invisible against dark background
export const TEAM_KIT_COLORS: Record<string, string> = {
  Mexico:               "#006847",
  "South Africa":       "#007A4D",
  "South Korea":        "#C60C30",
  Czechia:              "#D7141A",
  USA:                  "#B22234",
  Paraguay:             "#D52B1E",
  Australia:            "#FFD700",
  Turkey:               "#E30A17",
  Canada:               "#FF0000",
  "Bosnia & Herzegovina": "#003DA5",
  Qatar:                "#8D1B3D",
  Switzerland:          "#FF0000",
  Germany:              "#A8A9AD",  // away silver — white invisible on chalk
  "Curaçao":            "#003DA5",
  "Ivory Coast":        "#F77F00",
  Ecuador:              "#FFD100",
  Netherlands:          "#FF6600",
  Japan:                "#BC002D",
  Sweden:               "#006AA7",
  Tunisia:              "#E70013",
  Brazil:               "#FFD700",
  Morocco:              "#C1272D",
  Scotland:             "#003F87",
  Haiti:                "#00209F",
  France:               "#003189",
  Senegal:              "#00853F",
  Iraq:                 "#007A3D",
  Norway:               "#EF2B2D",
  Spain:                "#AA151B",
  "Cape Verde":         "#003893",
  "Saudi Arabia":       "#006C35",
  Uruguay:              "#5EB6E4",
  Belgium:              "#EF3340",
  Egypt:                "#CE1126",
  Iran:                 "#239F40",
  "New Zealand":        "#A8A9AD",  // away silver
  England:              "#CF142B",  // away red — white invisible
  Croatia:              "#FF0000",
  Ghana:                "#FCD116",
  Panama:               "#DA121A",
  Portugal:             "#006600",
  "DR Congo":           "#007FFF",
  Uzbekistan:           "#1EB53A",
  Colombia:             "#FCD116",
  Argentina:            "#74ACDF",
  Algeria:              "#006233",
  Austria:              "#ED2939",
  Jordan:               "#007A3D",
};

export function getFlagClass(team: string): string {
  const code = TEAM_ISO_CODES[team];
  if (!code) return "";
  // flag-icons uses fi fi-xx for standard codes, fi fi-xx-yy for subdivisions
  return `fi fi-${code}`;
}

export function getKitColor(team: string): string {
  return TEAM_KIT_COLORS[team] ?? "var(--turf)";
}

export function getFlagEmoji(team: string): string {
  const code = TEAM_ISO_CODES[team];
  if (!code || code.includes("-")) return "⚽";
  return code
    .toUpperCase()
    .split("")
    .map((c) => String.fromCodePoint(0x1f1e6 + c.charCodeAt(0) - 65))
    .join("");
}
