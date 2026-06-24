# ui/ — Proxima demo surface

React + Vite + TypeScript + Tailwind. Talks to the FastAPI backend on `:8000`.

## Run it

1. Start the backend (from the repo root):

   ```bash
   ./venv/Scripts/python.exe -m uvicorn proxima.api.server:app --port 8000
   ```

2. Start the UI (from `ui/`):

   ```bash
   npm install      # first time only
   npm run dev      # http://localhost:5173
   ```

3. Click **Seed demo data**, then click any node (or pick a title) to search.

Backend URL is configurable via `VITE_API_BASE` (defaults to `http://127.0.0.1:8000`).

## What's here

- **Vector-space map** (`components/VectorMap.tsx`) — hand-rolled SVG scatter of
  the PCA-projected vectors; glowing nodes coloured by genre. Selecting a node
  highlights it and draws links to its k nearest neighbours.
- **Search playground** (`SearchPanel`) — pick a title → ranked neighbours with
  distance scores; `k` is adjustable.
- **Filters** (`FilterPanel`) — genre/studio toggles that post-filter the search
  and dim non-matching nodes on the map.
- **Live metrics** (`MetricsPanel`) — vector count, recall@10 (vs brute force),
  avg latency, QPS.
- **Data controls** (`DataControls`) — Seed / Clear, wired to `/demo/seed` and
  `/demo/reset`.
- **Collection browser** (`CollectionBrowser`).

## Design note

We hand-rolled the scatter in SVG rather than pulling in d3/plotly — full control
over the glow + neighbour-line aesthetic, zero charting deps. UI primitives in
`components/ui.tsx` follow the shadcn/ui pattern (Tailwind + `cn`); run
`npx shadcn@latest add ...` later if you want the full component library.
