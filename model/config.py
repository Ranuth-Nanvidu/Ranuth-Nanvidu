"""Every constant the model, renderer and README writer use, in one place."""

from pathlib import Path

OWNER = "Ranuth-Nanvidu"
REPO = f"{OWNER}/{OWNER}"
BRANCH = "main"

# Model
LABELS = ("blue", "orange")
K = 5
COORD_MIN, COORD_MAX = 0.0, 10.0

# Limits
MAX_VISITOR_POINTS = 300
RATE_LIMIT = 3
RATE_WINDOW_HOURS = 24
DUPLICATE_DISTANCE = 0.05

# Files
ROOT = Path(__file__).resolve().parent.parent
POINTS_CSV = ROOT / "data" / "points.csv"
README = ROOT / "README.md"
SVG_PATHS = {
    "dark": ROOT / "assets" / "model-dark.svg",
    "light": ROOT / "assets" / "model-light.svg",
}
OUT_DIR = ROOT / "out"

# Seed data
SEED_USER = "seed"
SEED_TIMESTAMP = "2026-11-18T00:00:00Z"
SEED_RANDOM = 7

# Rendering
SVG_WIDTH, SVG_HEIGHT = 560, 320
GRID_COLS, GRID_ROWS = 80, 46
POINT_COLORS = {"blue": "#378ADD", "orange": "#EF9F27"}
THEMES = {
    "dark": {"bg": "#0d1117", "axis": "#8b949e", "border": "#30363d"},
    "light": {"bg": "#ffffff", "axis": "#6e7781", "border": "#d0d7de"},
}
