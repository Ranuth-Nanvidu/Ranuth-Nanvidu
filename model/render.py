"""Draws the decision boundary as a small, self-contained SVG."""

from . import config
from .knn import confidence, predict

W, H = config.SVG_WIDTH, config.SVG_HEIGHT
PAD_LEFT, PAD_BOTTOM, PAD_TOP, PAD_RIGHT = 36, 36, 14, 14
PLOT_X, PLOT_Y = PAD_LEFT, PAD_TOP
PLOT_W, PLOT_H = W - PAD_LEFT - PAD_RIGHT, H - PAD_TOP - PAD_BOTTOM
SPAN = config.COORD_MAX - config.COORD_MIN
TICKS = (0, 2, 4, 6, 8, 10)
FONT = "ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,'Liberation Mono',monospace"

STYLE = (
    ".ring{transform-box:fill-box;transform-origin:center;"
    "animation:pulse 1.6s ease-out 3}"
    "@keyframes pulse{0%{opacity:0;transform:scale(.4)}"
    "45%{opacity:1}100%{opacity:1;transform:scale(1)}}"
    "@media (prefers-reduced-motion:reduce){.ring{animation:none}}"
)


def num(v):
    """Compact number for SVG attributes: 12.50 -> 12.5, 3.00 -> 3."""
    return f"{v:.2f}".rstrip("0").rstrip(".")


def to_px(x, y):
    return (
        PLOT_X + (x - config.COORD_MIN) / SPAN * PLOT_W,
        PLOT_Y + PLOT_H - (y - config.COORD_MIN) / SPAN * PLOT_H,
    )


def region_opacity(conf):
    """0.10 at 50% confidence up to 0.32 at 100%."""
    return round(0.10 + 0.22 * (conf - 0.5) * 2, 2)


def region_runs(points):
    """Classify every grid cell and merge horizontal runs of the same colour and opacity.

    Returns (row, first_col, length, label, opacity) tuples. Row 0 is the top of the plot.
    """
    cols, rows = config.GRID_COLS, config.GRID_ROWS
    runs = []
    for r in range(rows):
        y = config.COORD_MAX - (r + 0.5) * SPAN / rows
        for c in range(cols):
            x = config.COORD_MIN + (c + 0.5) * SPAN / cols
            label, p = predict(points, x, y)
            style = (label, region_opacity(confidence(label, p)))
            last = runs[-1] if runs else None
            if last and last[0] == r and (last[3], last[4]) == style:
                runs[-1] = (r, last[1], last[2] + 1, *style)
            else:
                runs.append((r, c, 1, *style))
    return runs


def render(points, theme, newest=None, visitors=0, runs=None):
    """Return SVG text. `newest` is the (x, y, label) point to highlight, if any."""
    colors = config.THEMES[theme]
    runs = region_runs(points) if runs is None else runs
    cell_w, cell_h = PLOT_W / config.GRID_COLS, PLOT_H / config.GRID_ROWS
    label = f"KNN decision boundary trained by {visitors} visitor{'' if visitors == 1 else 's'}"

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" aria-label="{label}">',
        f"<title>{label}</title>",
        f"<style>{STYLE}</style>",
        f'<rect x=".5" y=".5" width="{W - 1}" height="{H - 1}" rx="8" '
        f'fill="{colors["bg"]}" stroke="{colors["border"]}"/>',
    ]

    # Decision regions, one <rect> per merged run.
    out.append('<g shape-rendering="crispEdges">')
    for r, c, length, run_label, opacity in runs:
        out.append(
            f'<rect x="{num(PLOT_X + c * cell_w)}" y="{num(PLOT_Y + r * cell_h)}" '
            f'width="{num(length * cell_w)}" height="{num(cell_h)}" '
            f'fill="{config.POINT_COLORS[run_label]}" fill-opacity="{opacity}"/>'
        )
    out.append("</g>")

    # Axes, tick marks and labels.
    bottom, right = PLOT_Y + PLOT_H, PLOT_X + PLOT_W
    out.append(f'<g stroke="{colors["axis"]}" stroke-width="1" shape-rendering="crispEdges">')
    out.append(f'<line x1="{PLOT_X}" y1="{PLOT_Y}" x2="{PLOT_X}" y2="{bottom}"/>')
    out.append(f'<line x1="{PLOT_X}" y1="{bottom}" x2="{right}" y2="{bottom}"/>')
    for t in TICKS:
        tx, ty = to_px(t, t)
        out.append(f'<line x1="{num(tx)}" y1="{bottom}" x2="{num(tx)}" y2="{bottom + 4}"/>')
        out.append(f'<line x1="{PLOT_X - 4}" y1="{num(ty)}" x2="{PLOT_X}" y2="{num(ty)}"/>')
    out.append("</g>")
    out.append(f'<g fill="{colors["axis"]}" font-family="{FONT}" font-size="10">')
    for t in TICKS:
        tx, ty = to_px(t, t)
        out.append(f'<text x="{num(tx)}" y="{bottom + 16}" text-anchor="middle">{t}</text>')
        out.append(f'<text x="{PLOT_X - 8}" y="{num(ty)}" dy=".35em" text-anchor="end">{t}</text>')
    out.append("</g>")

    # Training points, outlined in the background colour so they stand out.
    out.append(f'<g stroke="{colors["bg"]}" stroke-width="1.5">')
    for x, y, point_label in points:
        px, py = to_px(x, y)
        out.append(f'<circle cx="{num(px)}" cy="{num(py)}" r="5" fill="{config.POINT_COLORS[point_label]}"/>')
    out.append("</g>")

    if newest:
        px, py = to_px(newest[0], newest[1])
        out.append(
            f'<circle class="ring" cx="{num(px)}" cy="{num(py)}" r="10" fill="none" '
            f'stroke="{config.POINT_COLORS[newest[2]]}" stroke-width="2"/>'
        )

    out.append("</svg>")
    return "\n".join(out) + "\n"
