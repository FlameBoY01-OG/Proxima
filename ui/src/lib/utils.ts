/** Tiny classnames helper (shadcn uses clsx + tailwind-merge; we keep it dep-free). */
export function cn(...parts: Array<string | false | null | undefined>): string {
  return parts.filter(Boolean).join(" ");
}

/** Mirror of the backend's metadata predicate so the map can fade non-matches
 *  client-side without a round-trip. Value may be a scalar or a list (membership). */
export function matchesFilter(
  metadata: Record<string, unknown>,
  filter: Record<string, unknown> | null,
): boolean {
  if (!filter) return true;
  for (const [key, want] of Object.entries(filter)) {
    const have = metadata[key];
    if (Array.isArray(want)) {
      if (!want.includes(have)) return false;
    } else if (have !== want) {
      return false;
    }
  }
  return true;
}
