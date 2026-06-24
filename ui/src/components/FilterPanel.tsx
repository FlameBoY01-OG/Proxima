import { genreColor, prettyLabel } from "../lib/genres";
import { Badge, Card, CardContent, CardHeader, CardTitle } from "./ui";

export interface Facets {
  genre: string[];
  studio: string[];
}

interface Props {
  facets: Facets;
  selected: { genre: Set<string>; studio: Set<string> };
  onToggle: (facet: "genre" | "studio", value: string) => void;
  onClear: () => void;
}

export function FilterPanel({ facets, selected, onToggle, onClear }: Props) {
  const anyActive = selected.genre.size > 0 || selected.studio.size > 0;

  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <CardTitle>Filters</CardTitle>
        {anyActive && (
          <button onClick={onClear} className="text-[11px] text-muted hover:text-fg">
            clear
          </button>
        )}
      </CardHeader>
      <CardContent className="space-y-3">
        <div>
          <p className="mb-1.5 text-[11px] uppercase tracking-wide text-muted">Genre</p>
          <div className="flex flex-wrap gap-1.5">
            {facets.genre.map((g) => (
              <Badge
                key={g}
                color={genreColor(g)}
                active={selected.genre.has(g)}
                onClick={() => onToggle("genre", g)}
              >
                {prettyLabel(g)}
              </Badge>
            ))}
          </div>
        </div>
        {facets.studio.length > 0 && (
          <div>
            <p className="mb-1.5 text-[11px] uppercase tracking-wide text-muted">Studio</p>
            <div className="flex flex-wrap gap-1.5">
              {facets.studio.map((s) => (
                <Badge
                  key={s}
                  active={selected.studio.has(s)}
                  onClick={() => onToggle("studio", s)}
                >
                  {s}
                </Badge>
              ))}
            </div>
          </div>
        )}
        <p className="text-[11px] text-muted">
          Filters post-filter the search and dim non-matching nodes on the map.
        </p>
      </CardContent>
    </Card>
  );
}
