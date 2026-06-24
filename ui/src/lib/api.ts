// Typed client for the Proxima FastAPI backend.
// Base URL is configurable via VITE_API_BASE; defaults to the local server.

const BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export interface Metadata {
  title?: string;
  genre?: string;
  year?: number;
  studio?: string;
  [k: string]: unknown;
}

export interface CollectionInfo {
  name: string;
  dim: number;
  metric: string;
  count: number;
}

export interface ProjectedPoint {
  id: number;
  x: number;
  y: number;
  metadata: Metadata;
}

export interface SearchHit {
  id: number;
  distance: number;
  metadata: Metadata;
}

export interface SearchByIdResponse {
  query_id: number;
  results: SearchHit[];
  took_ms: number;
}

export interface Metrics {
  vector_count: number;
  recall_at_10: number | null;
  avg_latency_ms: number | null;
  qps: number | null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>("/health"),

  listCollections: () =>
    request<{ collections: CollectionInfo[] }>("/collections").then((r) => r.collections),

  projection: (collection: string) =>
    request<{ points: ProjectedPoint[] }>(
      `/collections/${collection}/projection`,
    ).then((r) => r.points),

  searchById: (
    collection: string,
    id: number,
    k: number,
    filter: Record<string, unknown> | null,
  ) =>
    request<SearchByIdResponse>(`/collections/${collection}/search_by_id`, {
      method: "POST",
      body: JSON.stringify({ id, k, filter }),
    }),

  metrics: (collection: string) =>
    request<Metrics>(`/collections/${collection}/metrics`),

  seedDemo: () => request<{ collection: string; count: number }>("/demo/seed", { method: "POST" }),

  resetDemo: () => request<{ collection: string; removed: number }>("/demo/reset", { method: "POST" }),
};
