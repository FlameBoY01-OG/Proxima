// One hue per genre. The map colours nodes by genre so clusters read at a glance.
export const GENRE_COLORS: Record<string, string> = {
  action: "#f87171",
  romance: "#f472b6",
  scifi: "#38bdf8",
  fantasy: "#a78bfa",
  slice_of_life: "#34d399",
  mystery: "#fbbf24",
  sports: "#fb923c",
  horror: "#c084fc",
};

export const FALLBACK_COLOR = "#67e8f9";

export function genreColor(genre: unknown): string {
  return (typeof genre === "string" && GENRE_COLORS[genre]) || FALLBACK_COLOR;
}

/** "slice_of_life" -> "Slice Of Life" for display. */
export function prettyLabel(value: string): string {
  return value
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}
