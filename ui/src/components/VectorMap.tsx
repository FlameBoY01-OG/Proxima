import { useMemo, useState } from "react";
import type { ProjectedPoint, SearchHit } from "../lib/api";
import { genreColor } from "../lib/genres";
import { matchesFilter } from "../lib/utils";

interface Props {
  points: ProjectedPoint[];
  selectedId: number | null;
  neighbors: SearchHit[]; // search results (incl. the query point itself)
  filter: Record<string, unknown> | null;
  onSelect: (id: number) => void;
}

// The SVG coordinate space. The element scales to its container via viewBox.
const W = 1000;
const H = 680;
const PAD = 56;

export function VectorMap({ points, selectedId, neighbors, filter, onSelect }: Props) {
  const [hovered, setHovered] = useState<number | null>(null);

  // Build linear scales from data-space (PCA coords) to screen-space. Memoized
  // so we don't recompute the extent on every hover. Guards a zero-width range
  // (all points identical) by centering.
  const { sx, sy, byId } = useMemo(() => {
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    const sx = (x: number) => PAD + ((x - minX) / spanX) * (W - 2 * PAD);
    // Invert Y so positive points up, matching how we read a plot.
    const sy = (y: number) => H - PAD - ((y - minY) / spanY) * (H - 2 * PAD);
    const byId = new Map(points.map((p) => [p.id, p]));
    return { sx, sy, byId };
  }, [points]);

  if (points.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-muted">
        <p className="text-sm">No vectors yet — seed the demo data to populate the map.</p>
      </div>
    );
  }

  const neighborIds = new Set(neighbors.map((n) => n.id).filter((id) => id !== selectedId));
  const selected = selectedId != null ? byId.get(selectedId) : undefined;

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-full w-full"
      preserveAspectRatio="xMidYMid meet"
    >
      {/* Neighbour links: drawn first so nodes sit on top. */}
      {selected &&
        neighbors
          .filter((n) => n.id !== selectedId)
          .map((n) => {
            const p = byId.get(n.id);
            if (!p) return null;
            // Nearer neighbours get brighter, slightly thicker links.
            const strength = 1 - Math.min(neighbors.indexOf(n) / neighbors.length, 1);
            return (
              <line
                key={`l-${n.id}`}
                x1={sx(selected.x)}
                y1={sy(selected.y)}
                x2={sx(p.x)}
                y2={sy(p.y)}
                stroke="#67e8f9"
                strokeWidth={0.6 + strength * 1.4}
                strokeOpacity={0.18 + strength * 0.4}
              />
            );
          })}

      {/* Nodes */}
      {points.map((p) => {
        const color = genreColor(p.metadata.genre);
        const isSelected = p.id === selectedId;
        const isNeighbor = neighborIds.has(p.id);
        const dim = !matchesFilter(p.metadata, filter);
        const baseR = isSelected ? 9 : isNeighbor ? 6.5 : 5;
        const opacity = dim ? 0.12 : 1;

        return (
          <g
            key={p.id}
            transform={`translate(${sx(p.x)} ${sy(p.y)})`}
            onMouseEnter={() => setHovered(p.id)}
            onMouseLeave={() => setHovered(null)}
            onClick={() => onSelect(p.id)}
            style={{ cursor: "pointer" }}
            opacity={opacity}
          >
            {/* soft glow */}
            <circle r={baseR * 2.6} fill={color} opacity={isSelected ? 0.28 : 0.14} />
            {/* core */}
            <circle r={baseR} fill={color} />
            {/* selection / neighbour rings */}
            {isSelected && <circle r={baseR + 5} fill="none" stroke="#fff" strokeWidth={1.5} />}
            {isNeighbor && <circle r={baseR + 3} fill="none" stroke="#67e8f9" strokeWidth={1.2} />}
          </g>
        );
      })}

      {/* Hover / selection label, drawn last so it's never occluded. */}
      {[hovered, selectedId].map((id, i) => {
        if (id == null) return null;
        const p = byId.get(id);
        if (!p) return null;
        const label = p.metadata.title ?? `#${p.id}`;
        return (
          <text
            key={`t-${i}-${id}`}
            x={sx(p.x)}
            y={sy(p.y) - 16}
            textAnchor="middle"
            className="fill-fg font-sans"
            style={{ fontSize: 13, paintOrder: "stroke" }}
            stroke="#06070b"
            strokeWidth={3}
          >
            {label}
          </text>
        );
      })}
    </svg>
  );
}
