import type { CollectionInfo } from "../lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "./ui";

interface Props {
  collections: CollectionInfo[];
  selected: string | null;
  onSelect: (name: string) => void;
}

export function CollectionBrowser({ collections, selected, onSelect }: Props) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Collections</CardTitle>
      </CardHeader>
      <CardContent className="space-y-1">
        {collections.length === 0 && (
          <p className="text-sm text-muted">None yet.</p>
        )}
        {collections.map((c) => (
          <button
            key={c.name}
            onClick={() => onSelect(c.name)}
            className={`flex w-full items-center justify-between rounded-lg px-2.5 py-1.5 text-sm transition-colors ${
              c.name === selected ? "bg-ink-700 text-fg" : "text-muted hover:bg-ink-800 hover:text-fg"
            }`}
          >
            <span className="truncate">{c.name}</span>
            <span className="font-mono text-[11px] text-muted">
              {c.count} · {c.metric} · {c.dim}d
            </span>
          </button>
        ))}
      </CardContent>
    </Card>
  );
}
