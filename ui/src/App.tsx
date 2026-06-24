import { useCallback, useEffect, useMemo, useState } from "react";
import { Boxes } from "lucide-react";
import {
  api,
  type CollectionInfo,
  type Metrics,
  type ProjectedPoint,
  type SearchHit,
} from "./lib/api";
import { genreColor, prettyLabel } from "./lib/genres";
import { CollectionBrowser } from "./components/CollectionBrowser";
import { DataControls } from "./components/DataControls";
import { FilterPanel, type Facets } from "./components/FilterPanel";
import { MetricsPanel } from "./components/MetricsPanel";
import { SearchPanel } from "./components/SearchPanel";
import { VectorMap } from "./components/VectorMap";

export default function App() {
  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [collection, setCollection] = useState<string | null>(null);
  const [points, setPoints] = useState<ProjectedPoint[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);

  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [results, setResults] = useState<SearchHit[]>([]);
  const [tookMs, setTookMs] = useState<number | null>(null);
  const [k, setK] = useState(10);

  const [genreSel, setGenreSel] = useState<Set<string>>(new Set());
  const [studioSel, setStudioSel] = useState<Set<string>>(new Set());

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backendUp, setBackendUp] = useState<boolean | null>(null);

  // Facets and the active filter dict are derived from the loaded points.
  const facets: Facets = useMemo(() => {
    const genres = new Set<string>();
    const studios = new Set<string>();
    for (const p of points) {
      if (p.metadata.genre) genres.add(p.metadata.genre);
      if (p.metadata.studio) studios.add(p.metadata.studio);
    }
    return { genre: [...genres].sort(), studio: [...studios].sort() };
  }, [points]);

  const filter = useMemo(() => {
    const f: Record<string, string[]> = {};
    if (genreSel.size) f.genre = [...genreSel];
    if (studioSel.size) f.studio = [...studioSel];
    return Object.keys(f).length ? f : null;
  }, [genreSel, studioSel]);

  // ---- data loading -----------------------------------------------------

  const refreshCollections = useCallback(async () => {
    const cols = await api.listCollections();
    setCollections(cols);
    return cols;
  }, []);

  const loadCollectionData = useCallback(async (name: string) => {
    const [pts, m] = await Promise.all([api.projection(name), api.metrics(name)]);
    setPoints(pts);
    setMetrics(m);
  }, []);

  const guard = useCallback(async (fn: () => Promise<void>) => {
    try {
      setError(null);
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  // Initial boot: confirm backend, load collections, pick one.
  useEffect(() => {
    void guard(async () => {
      await api.health();
      setBackendUp(true);
      const cols = await refreshCollections();
      const pick = cols.find((c) => c.name === "anime") ?? cols[0];
      if (pick) setCollection(pick.name);
    }).catch(() => setBackendUp(false));
  }, [guard, refreshCollections]);

  // When the active collection changes, (re)load its map + metrics.
  useEffect(() => {
    if (!collection) return;
    setSelectedId(null);
    setResults([]);
    setTookMs(null);
    void guard(() => loadCollectionData(collection));
  }, [collection, guard, loadCollectionData]);

  // Re-run the search whenever the query, k, or filter changes.
  useEffect(() => {
    if (!collection || selectedId == null) return;
    void guard(async () => {
      const r = await api.searchById(collection, selectedId, k, filter);
      setResults(r.results);
      setTookMs(r.took_ms);
    });
  }, [collection, selectedId, k, filter, guard]);

  // ---- actions ----------------------------------------------------------

  const onSeed = () =>
    void guard(async () => {
      setBusy(true);
      try {
        const { collection: name } = await api.seedDemo();
        await refreshCollections();
        if (collection === name) await loadCollectionData(name);
        else setCollection(name);
      } finally {
        setBusy(false);
      }
    });

  const onReset = () =>
    void guard(async () => {
      setBusy(true);
      try {
        await api.resetDemo();
        await refreshCollections();
        if (collection) await loadCollectionData(collection);
        setSelectedId(null);
        setResults([]);
      } finally {
        setBusy(false);
      }
    });

  const toggleFacet = (facet: "genre" | "studio", value: string) => {
    const setter = facet === "genre" ? setGenreSel : setStudioSel;
    setter((prev) => {
      const next = new Set(prev);
      next.has(value) ? next.delete(value) : next.add(value);
      return next;
    });
  };

  const clearFilters = () => {
    setGenreSel(new Set());
    setStudioSel(new Set());
  };

  // ---- render -----------------------------------------------------------

  return (
    <div className="flex h-screen flex-col">
      <header className="flex items-center justify-between border-b border-line px-5 py-3">
        <div className="flex items-center gap-2.5">
          <Boxes className="text-accent" size={20} />
          <div>
            <h1 className="text-sm font-semibold tracking-wide">PROXIMA</h1>
            <p className="text-[11px] text-muted">hand-rolled vector database</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {backendUp === false && (
            <span className="text-xs text-red-400">backend offline — start uvicorn on :8000</span>
          )}
          <DataControls onSeed={onSeed} onReset={onReset} busy={busy} />
        </div>
      </header>

      {error && (
        <div className="border-b border-red-500/30 bg-red-500/10 px-5 py-1.5 text-xs text-red-300">
          {error}
        </div>
      )}

      <main className="flex min-h-0 flex-1 gap-4 p-4">
        {/* Left: collections + metrics */}
        <aside className="flex w-72 shrink-0 flex-col gap-4 overflow-y-auto">
          <CollectionBrowser
            collections={collections}
            selected={collection}
            onSelect={setCollection}
          />
          <MetricsPanel metrics={metrics} />
        </aside>

        {/* Center: the vector-space map */}
        <section className="flex min-w-0 flex-1 flex-col rounded-xl border border-line bg-ink-900/40">
          <div className="flex items-center justify-between border-b border-line px-4 py-2.5">
            <div>
              <h2 className="text-xs font-semibold uppercase tracking-wider text-muted">
                Vector space
              </h2>
              <p className="text-[11px] text-muted">PCA projection · 16D → 2D</p>
            </div>
            <div className="flex flex-wrap justify-end gap-1.5">
              {facets.genre.map((g) => (
                <span key={g} className="flex items-center gap-1 text-[10px] text-muted">
                  <span
                    className="h-2 w-2 rounded-full"
                    style={{ backgroundColor: genreColor(g) }}
                  />
                  {prettyLabel(g)}
                </span>
              ))}
            </div>
          </div>
          <div className="min-h-0 flex-1 p-2">
            <VectorMap
              points={points}
              selectedId={selectedId}
              neighbors={results}
              filter={filter}
              onSelect={setSelectedId}
            />
          </div>
        </section>

        {/* Right: search + filters */}
        <aside className="flex w-80 shrink-0 flex-col gap-4 overflow-y-auto">
          <SearchPanel
            points={points}
            selectedId={selectedId}
            results={results}
            tookMs={tookMs}
            k={k}
            onSelect={setSelectedId}
            onChangeK={setK}
          />
          <FilterPanel
            facets={facets}
            selected={{ genre: genreSel, studio: studioSel }}
            onToggle={toggleFacet}
            onClear={clearFilters}
          />
        </aside>
      </main>
    </div>
  );
}
