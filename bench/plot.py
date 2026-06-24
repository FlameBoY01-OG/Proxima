"""Render the recall-vs-QPS curve as a self-contained SVG.

Hand-rolled (no matplotlib) — same philosophy as the UI map: for a handful of
points an SVG line chart is trivial, dependency-free, and commits cleanly into
the README. The classic ANN plot puts recall on the x-axis and throughput (QPS)
on the y-axis; each point is one ef_search value. Up-and-to-the-right is better.
"""

from __future__ import annotations

W, H = 720, 460
PAD_L, PAD_R, PAD_T, PAD_B = 70, 30, 50, 60


def render_curve_svg(rows: list[dict], path: str, title: str = "Recall@10 vs QPS") -> None:
    if not rows:
        raise ValueError("no rows to plot")

    recalls = [r["recall"] for r in rows]
    qps = [r["qps"] for r in rows]

    # Axis ranges with a little headroom; recall capped at 1.0.
    x_min = min(min(recalls), 0.0)
    x_max = 1.0
    y_min = 0.0
    y_max = max(qps) * 1.1

    def sx(x: float) -> float:
        return PAD_L + (x - x_min) / (x_max - x_min or 1) * (W - PAD_L - PAD_R)

    def sy(y: float) -> float:
        return H - PAD_B - (y - y_min) / (y_max - y_min or 1) * (H - PAD_T - PAD_B)

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="Inter, system-ui, sans-serif">'
    )
    parts.append(f'<rect width="{W}" height="{H}" fill="#0a0c12" rx="12"/>')
    parts.append(
        f'<text x="{PAD_L}" y="28" fill="#e7eaf2" font-size="15" '
        f'font-weight="600">{title}</text>'
    )

    # Gridlines + y ticks (QPS).
    for i in range(5):
        gy = PAD_T + i * (H - PAD_T - PAD_B) / 4
        val = y_max - i * (y_max - y_min) / 4
        parts.append(
            f'<line x1="{PAD_L}" y1="{gy:.1f}" x2="{W - PAD_R}" y2="{gy:.1f}" '
            f'stroke="#1e2433" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{PAD_L - 8}" y="{gy + 4:.1f}" fill="#7c869b" font-size="11" '
            f'text-anchor="end">{val:,.0f}</text>'
        )

    # x ticks (recall).
    for i in range(6):
        xv = x_min + i * (x_max - x_min) / 5
        gx = sx(xv)
        parts.append(
            f'<text x="{gx:.1f}" y="{H - PAD_B + 18:.1f}" fill="#7c869b" font-size="11" '
            f'text-anchor="middle">{xv:.2f}</text>'
        )

    # Axis labels.
    parts.append(
        f'<text x="{(PAD_L + W - PAD_R) / 2:.0f}" y="{H - 16}" fill="#a8b0c0" '
        f'font-size="12" text-anchor="middle">recall@10 (vs brute force)</text>'
    )
    parts.append(
        f'<text x="20" y="{(PAD_T + H - PAD_B) / 2:.0f}" fill="#a8b0c0" font-size="12" '
        f'text-anchor="middle" transform="rotate(-90 20 {(PAD_T + H - PAD_B) / 2:.0f})">'
        f'queries / sec</text>'
    )

    # The curve (sorted by recall so the polyline reads left-to-right).
    pts = sorted(zip(recalls, qps), key=lambda t: t[0])
    poly = " ".join(f"{sx(r):.1f},{sy(q):.1f}" for r, q in pts)
    parts.append(f'<polyline points="{poly}" fill="none" stroke="#67e8f9" stroke-width="2"/>')

    # Points + ef_search labels.
    for r in rows:
        cx, cy = sx(r["recall"]), sy(r["qps"])
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="#67e8f9"/>')
        parts.append(
            f'<text x="{cx:.1f}" y="{cy - 10:.1f}" fill="#e7eaf2" font-size="10" '
            f'text-anchor="middle">ef={r["ef_search"]}</text>'
        )

    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
