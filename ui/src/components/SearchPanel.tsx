import type { ProjectedPoint, SearchHit } from "../lib/api";
import { genreColor, prettyLabel } from "../lib/genres";
import { Card, CardContent, CardHeader, CardTitle } from "./ui";

interface Props {
  points: ProjectedPoint[];
  selectedId: number | null;
  results: SearchHit[];
  tookMs: number | null;
  k: number;
  onSelect: (id: number) => void;
  onChangeK: (k: number) => void;
}

export function SearchPanel({
  points,
  selectedId,
  results,
  tookMs,
  k,
  onSelect,
  onChangeK,
}: Props) {
  const sorted = [...points].sort((a, b) =>
    (a.metadata.title ?? "").localeCompare(b.metadata.title ?? ""),
  );

  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <CardTitle>Search playground</CardTitle>
        {tookMs != null && (
          <span className="font-mono text-[11px] text-accent">{tookMs.toFixed(2)} ms</span>
        )}
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Pick a title -> its vector becomes the query ("find titles like this"). */}
        <div className="flex items-center gap-2">
          <select
            value={selectedId ?? ""}
            onChange={(e) => e.target.value && onSelect(Number(e.target.value))}
            className="min-w-0 flex-1 rounded-lg border border-line bg-ink-800 px-2 py-1.5 text-sm text-fg outline-none focus:border-accent/50"
          >
            <option value="" disabled>
              Find titles like…
            </option>
            {sorted.map((p) => (
              <option key={p.id} value={p.id}>
                {p.metadata.title ?? `#${p.id}`}
              </option>
            ))}
          </select>
          <label className="flex items-center gap-1 text-xs text-muted">
            k
            <input
              type="number"
              min={1}
              max={20}
              value={k}
              onChange={(e) => onChangeK(Math.max(1, Math.min(20, Number(e.target.value))))}
              className="w-12 rounded-md border border-line bg-ink-800 px-1.5 py-1 text-center text-sm text-fg outline-none focus:border-accent/50"
            />
          </label>
        </div>

        {/* Ranked results with scores. */}
        <ol className="space-y-1">
          {results.map((hit, i) => {
            const isQuery = hit.id === selectedId;
            return (
              <li
                key={hit.id}
                onClick={() => onSelect(hit.id)}
                className={`flex cursor-pointer items-center gap-2 rounded-lg px-2 py-1.5 text-sm transition-colors hover:bg-ink-700 ${
                  isQuery ? "bg-ink-700" : ""
                }`}
              >
                <span className="w-4 text-right font-mono text-[11px] text-muted">{i + 1}</span>
                <span
                  className="h-2.5 w-2.5 shrink-0 rounded-full"
                  style={{ backgroundColor: genreColor(hit.metadata.genre) }}
                />
                <span className="min-w-0 flex-1 truncate">
                  {hit.metadata.title ?? `#${hit.id}`}
                  {isQuery && <span className="ml-1 text-[10px] text-accent">query</span>}
                </span>
                <span className="font-mono text-[11px] text-muted">{hit.distance.toFixed(3)}</span>
              </li>
            );
          })}
          {results.length === 0 && (
            <li className="px-2 py-3 text-sm text-muted">
              Pick a title or click a node to search.
            </li>
          )}
        </ol>
        {results.length > 0 && (
          <p className="px-2 text-[11px] text-muted">
            Lower distance = nearer. Coloured by {prettyLabel("genre")}.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
