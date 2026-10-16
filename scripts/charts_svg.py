"""Small dependency-free SVG charts for eval/EVAL.md and the offline HTML docs.

Mark specs: columns <= 24 px with a 4 px rounded data end and a 2 px surface gap; 2 px lines;
end markers r = 4 with a 2 px surface ring; hairline grid; text in ink tokens (never the series
colour); legend whenever there are two or more series, plus direct end labels on lines.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical order (slots 1-4)
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _t(x: float, y: float, s: str, size: int = 12, fill: str = INK2, anchor: str = "start", weight: str = "400") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" font-family=\'{FONT}\'>{html.escape(s)}</text>')


def _legend(items: Sequence[Tuple[str, str]], x: float, y: float) -> str:
    out = []
    for name, color in items:
        out.append(f'<rect x="{x:.1f}" y="{y - 9:.1f}" width="10" height="10" rx="2" fill="{color}"/>')
        out.append(_t(x + 15, y, name, 12, INK2))
        x += 26 + 7 * len(name)
    return "".join(out)


def _col(x: float, y0: float, y1: float, w: float, color: str, r: float = 4.0) -> str:
    h = y0 - y1
    if h <= 0.5:
        return ""
    r = min(r, h, w / 2)
    return (f'<path d="M{x:.1f},{y0:.1f} L{x:.1f},{y1 + r:.1f} Q{x:.1f},{y1:.1f} {x + r:.1f},{y1:.1f} '
            f'L{x + w - r:.1f},{y1:.1f} Q{x + w:.1f},{y1:.1f} {x + w:.1f},{y1 + r:.1f} L{x + w:.1f},{y0:.1f} Z" fill="{color}"/>')


def grouped_columns(title: str, cats: Sequence[str], series: Sequence[Tuple[str, Sequence[float]]], width: int = 760, height: int = 300) -> str:
    """Values are fractions 0..1, drawn as percent."""
    ml, mr, mt, mb = 44, 12, 52, 56
    pw, ph = width - ml - mr, height - mt - mb
    y = lambda v: mt + ph * (1 - v)  # noqa: E731
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{html.escape(title)}">',
           f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>', _t(ml, 20, title, 14, INK, weight="600")]
    out.append(_legend([(n, SERIES[i]) for i, (n, _) in enumerate(series)], ml, 40))
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID if v else AXIS}" stroke-width="1"/>')
        out.append(_t(ml - 6, y(v) + 4, f"{int(v * 100)}%", 11, MUTED, "end"))
    band = pw / max(1, len(cats))
    bw = min(24.0, (band - 16) / max(1, len(series)) - 2)
    for i, c in enumerate(cats):
        gx = ml + band * i + (band - (bw + 2) * len(series) + 2) / 2
        for k, (_, vals) in enumerate(series):
            v = max(0.0, min(1.0, float(vals[i])))
            out.append(_col(gx + k * (bw + 2), y(0), y(v), bw, SERIES[k]))
        out.append(_t(ml + band * i + band / 2, y(0) + 16, c, 11, INK2, "middle"))
    out.append("</svg>")
    return "".join(out)


def line_chart(title: str, xs: Sequence[float], series: Sequence[Tuple[str, Sequence[float]]], xlabel: str, width: int = 760, height: int = 300,
               ymin: float = 0.0, ymax: float = 1.0) -> str:
    ml, mr, mt, mb = 44, 130, 52, 44
    pw, ph = width - ml - mr, height - mt - mb
    x0, x1 = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1
    fx = lambda v: ml + pw * (v - x0) / (x1 - x0)  # noqa: E731
    fy = lambda v: mt + ph * (1 - (v - ymin) / (ymax - ymin))  # noqa: E731
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{html.escape(title)}">',
           f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>', _t(ml, 20, title, 14, INK, weight="600")]
    if len(series) > 1:
        out.append(_legend([(n, SERIES[i]) for i, (n, _) in enumerate(series)], ml, 40))
    steps = 4
    for k in range(steps + 1):
        v = ymin + (ymax - ymin) * k / steps
        out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{fy(v):.1f}" y2="{fy(v):.1f}" stroke="{GRID if k else AXIS}" stroke-width="1"/>')
        out.append(_t(ml - 6, fy(v) + 4, f"{v * 100:.0f}%", 11, MUTED, "end"))
    for v in _ticks(x0, x1):
        out.append(_t(fx(v), mt + ph + 16, f"{v:g}", 11, MUTED, "middle"))
    out.append(_t(ml + pw / 2, height - 8, xlabel, 11, INK2, "middle"))
    ends: List[Tuple[float, str, str]] = []
    for i, (name, ys) in enumerate(series):
        pts = " ".join(f"{fx(a):.1f},{fy(b):.1f}" for a, b in zip(xs, ys))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{SERIES[i]}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        ex, ey = fx(xs[-1]), fy(ys[-1])
        out.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4" fill="{SERIES[i]}" stroke="{SURFACE}" stroke-width="2"/>')
        ends.append((ey, f"{name} {ys[-1] * 100:.0f}%", SERIES[i]))
    ends.sort()
    last = -1e9
    for ey, label, _c in ends:  # leader-free end labels, nudged only if they would overlap
        yy = max(ey + 4, last + 14)
        out.append(_t(ml + pw + 10, yy, label, 12, INK2))
        last = yy
    out.append("</svg>")
    return "".join(out)


def histogram(title: str, a: Sequence[float], b: Sequence[float], thr: float, names: Tuple[str, str], width: int = 760, height: int = 260, bins: int = 30) -> str:
    ml, mr, mt, mb = 44, 12, 52, 44
    pw, ph = width - ml - mr, height - mt - mb
    mx = max(list(a) + list(b) + [thr]) * 1.05

    def counts(v: Sequence[float]) -> List[float]:
        c = [0] * bins
        for x in v:
            c[min(bins - 1, int(x / mx * bins))] += 1
        return [n / max(1, len(v)) for n in c]

    ca, cb = counts(a), counts(b)
    ymax = max(ca + cb) * 1.1 or 1
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{html.escape(title)}">',
           f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>', _t(ml, 20, title, 14, INK, weight="600")]
    out.append(_legend([(names[0], SERIES[0]), (names[1], SERIES[1])], ml, 40))
    out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}" stroke="{AXIS}" stroke-width="1"/>')
    bw = pw / bins
    for k, c in enumerate((ca, cb)):
        for i, v in enumerate(c):
            if v:
                hgt = ph * v / ymax
                x = ml + i * bw + 1 + k * (bw / 2 - 1)
                out.append(_col(x, mt + ph, mt + ph - hgt, bw / 2 - 2, SERIES[k], r=2))
    tx = ml + pw * thr / mx
    out.append(f'<line x1="{tx:.1f}" x2="{tx:.1f}" y1="{mt}" y2="{mt + ph}" stroke="{INK}" stroke-width="1.5"/>')
    out.append(_t(tx + 4, mt + 12, f"novelty threshold {thr}", 11, INK2))
    for i in range(6):
        v = mx * i / 5
        out.append(_t(ml + pw * i / 5, mt + ph + 16, f"{v:.2f}", 11, MUTED, "middle"))
    out.append(_t(ml + pw / 2, height - 6, "cosine distance to the nearest labeled case", 11, INK2, "middle"))
    out.append("</svg>")
    return "".join(out)


def _ticks(a: float, b: float) -> List[float]:
    span = b - a
    for step in (1, 2, 5, 10, 20, 25, 50, 100, 200):
        if span / step <= 8:
            break
    start = int(a // step) * step
    return [v for v in range(int(start), int(b) + 1, step) if v >= a]


def write_charts(res: Dict[str, Any], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    short = {"normal": "normal", "cold_spot_hardware": "cold spot", "tc_sensor_fault": "TC fault", "combustion_instability": "instability",
             "dynamics_probe_fault": "probe fault", "hets_trip_hardware": "HETS", "igv_lvdt_drift": "IGV drift", "fuel_control_oscillation": "fuel osc"}
    pc = res["test"]["per_class"]
    svg = grouped_columns("Action accuracy by signature — held-out test split", [short.get(p["class"], p["class"]) for p in pc],
                          [("Learned path", [p["learned_action_acc"] for p in pc]), ("Rules", [p["rules_action_acc"] for p in pc])])
    (out_dir / "per_class.svg").write_text(svg, encoding="utf-8")
    paths["per_class"] = "charts/per_class.svg"
    if res.get("sim_site"):
        c = res["sim_site"]["curve"]
        svg = line_chart("Labeling simulation — accuracy on held-out site cases", [p["labels"] for p in c],
                         [("Learned action", [p["learned"] for p in c]), ("Rules action", [p["rules"] for p in c]),
                          ("Learned signature", [p.get("learned_signature", p["learned"]) for p in c])], "engineer labels given (retrain every 5)")
        (out_dir / "sim_site.svg").write_text(svg, encoding="utf-8")
        paths["sim_site"] = "charts/sim_site.svg"
    u = res.get("unseen") or {}
    if u.get("nn_test_all"):
        svg = histogram("Nearest labeled case distance — known signatures vs unseen bearing fault", u["nn_test_all"], u["nn_unseen_all"],
                        u["nn_distance_median"]["threshold"], ("known signatures (test)", "unseen fault"))
        (out_dir / "novelty.svg").write_text(svg, encoding="utf-8")
        paths["novelty"] = "charts/novelty.svg"
    if res.get("sim_cold"):
        ft, zs = res["sim_cold"]["fine_tuned"], res["sim_cold"]["zero_shot"]
        svg = line_chart("Cold start — test signature accuracy vs labels", [p["labels"] for p in ft],
                         [("Fine-tuned TS Pulse", [p["learned_acc"] for p in ft]), ("Zero-shot TS Pulse", [p["learned_acc"] for p in zs])], "labels given")
        (out_dir / "sim_cold.svg").write_text(svg, encoding="utf-8")
        paths["sim_cold"] = "charts/sim_cold.svg"
    return paths
