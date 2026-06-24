import type { Metrics } from "../lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "./ui";

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-line bg-ink-800/60 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-muted">{label}</div>
      <div className="font-mono text-lg text-fg">{value}</div>
      {hint && <div className="text-[10px] text-muted">{hint}</div>}
    </div>
  );
}

export function MetricsPanel({ metrics }: { metrics: Metrics | null }) {
  const recall = metrics?.recall_at_10;
  const latency = metrics?.avg_latency_ms;
  const qps = metrics?.qps;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Live metrics</CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-2 gap-2">
        <Stat label="Vectors" value={metrics ? String(metrics.vector_count) : "—"} />
        <Stat
          label="Recall@10"
          value={recall != null ? `${(recall * 100).toFixed(1)}%` : "—"}
          hint="vs brute force"
        />
        <Stat
          label="Latency"
          value={latency != null ? `${latency.toFixed(2)} ms` : "—"}
          hint="avg / query"
        />
        <Stat label="QPS" value={qps != null ? Math.round(qps).toLocaleString() : "—"} />
      </CardContent>
    </Card>
  );
}
