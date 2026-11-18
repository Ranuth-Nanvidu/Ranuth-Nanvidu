"""Entry point: parse the issue event, validate it, update the data, re-render.

    python -m model.train                      # inside the Action ($GITHUB_EVENT_PATH)
    python -m model.train --event FILE --dry-run
    python -m model.train --render-only
    python -m model.train --seed-data          # one-off: create data/points.csv

This module never touches the network. Issue text is untrusted: only the strict
regexes below read it, and nothing from it reaches the outputs except validated
numbers, a validated login and an issue number.
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from datetime import datetime, timedelta, timezone

from . import config, knn, readme, render

FIELDS = ["x", "y", "label", "user", "issue", "timestamp"]

TRAIN_TITLE = re.compile(r"^train\s*:\s*(.*)$", re.IGNORECASE)
PREDICT_TITLE = re.compile(r"^predict$", re.IGNORECASE)
X_VALUE = re.compile(r"(?i)\bx\s*[=:]\s*(-?\d+(?:\.\d+)?)")
Y_VALUE = re.compile(r"(?i)\by\s*[=:]\s*(-?\d+(?:\.\d+)?)")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
LOGIN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})")

FORMAT_HELP = (
    "**Format**\n"
    "- Title: `train: blue`, `train: orange` or `predict`\n"
    "- Body: `x=3.2, y=1.7` (both between 0 and 10)"
)


# ---------- parsing ----------

def parse_title(title):
    """('train', 'blue'|'orange'|None), ('predict', None), or None for unrelated issues.

    A `train:` title with an unknown label returns ('train', None) so it can be rejected.
    """
    title = (title or "").strip()
    if PREDICT_TITLE.match(title):
        return "predict", None
    m = TRAIN_TITLE.match(title)
    if not m:
        return None
    label = m.group(1).lower()
    return "train", label if label in config.LABELS else None


def parse_body(body):
    """Return ((x, y), None) or (None, error). Errors: ('format',) or ('range', x, y)."""
    text = HTML_COMMENT.sub(" ", body or "")
    mx, my = X_VALUE.search(text), Y_VALUE.search(text)
    if not (mx and my):
        return None, ("format",)
    x, y = round(float(mx.group(1)), 2), round(float(my.group(1)), 2)
    if not all(config.COORD_MIN <= v <= config.COORD_MAX for v in (x, y)):
        return None, ("range", x, y)
    return (x, y), None


# ---------- data ----------

def parse_time(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def format_time(dt):
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_rows(path=config.POINTS_CSV):
    with open(path, newline="", encoding="utf-8") as f:
        return [
            {**r, "x": float(r["x"]), "y": float(r["y"]), "issue": int(r["issue"])}
            for r in csv.DictReader(f)
        ]


def save_rows(rows, path=config.POINTS_CSV):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, FIELDS, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "x": f"{r['x']:.2f}", "y": f"{r['y']:.2f}"})


def as_points(rows):
    return [(r["x"], r["y"], r["label"]) for r in rows]


def is_seed(row):
    return row["user"] == config.SEED_USER


def visitor_count(rows):
    return len({r["user"].lower() for r in rows if not is_seed(r)})


def last_visitor(rows):
    return next((r for r in reversed(rows) if not is_seed(r)), None)


def seed_rows():
    """About 24 reproducible starter points: two clusters plus a few on the 'wrong' side."""
    rng = random.Random(config.SEED_RANDOM)
    clamp = lambda v: round(min(9.6, max(0.4, v)), 2)
    rows = []
    for label, (cx, cy) in (("blue", (3.0, 3.2)), ("orange", (6.8, 6.5))):
        for _ in range(11):
            rows.append((clamp(rng.gauss(cx, 0.9)), clamp(rng.gauss(cy, 0.9)), label))
    rows += [(4.3, 6.4, "orange"), (5.9, 3.9, "blue"), (4.9, 3.0, "orange")]
    return [
        {"x": x, "y": y, "label": label, "user": config.SEED_USER, "issue": 0,
         "timestamp": config.SEED_TIMESTAMP}
        for x, y, label in rows
    ]


# ---------- limits ----------

def rate_limit_reset(rows, user, now):
    """When the user may add another point, or None if they're under the limit now."""
    if user.lower() == config.OWNER.lower():
        return None
    window_start = now - timedelta(hours=config.RATE_WINDOW_HOURS)
    recent = sorted(
        t for t in (parse_time(r["timestamp"]) for r in rows if r["user"].lower() == user.lower())
        if t > window_start
    )
    if len(recent) < config.RATE_LIMIT:
        return None
    return recent[-config.RATE_LIMIT] + timedelta(hours=config.RATE_WINDOW_HOURS)


def find_duplicate(rows, user, x, y):
    """The user's existing point within DUPLICATE_DISTANCE of (x, y), if any."""
    for r in rows:
        if r["user"].lower() == user.lower() and math.dist((r["x"], r["y"]), (x, y)) <= config.DUPLICATE_DISTANCE:
            return r
    return None


def apply_cap(rows):
    """Drop the oldest visitor points beyond MAX_VISITOR_POINTS. Seed points always stay."""
    excess = sum(not is_seed(r) for r in rows) - config.MAX_VISITOR_POINTS
    kept = []
    for r in rows:
        if excess > 0 and not is_seed(r):
            excess -= 1
            continue
        kept.append(r)
    return kept


# ---------- comments ----------

def fmt(v):
    return f"{v:g}"


def pct(v):
    return f"{round(v * 100)}%"


def article(label):
    return "an" if label[0] in "aeiou" else "a"


def plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def wait_text(delta):
    minutes = max(1, math.ceil(delta.total_seconds() / 60))
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if hours else f"{minutes}m"


def train_comment(user, label, x, y, before, after, visitors, accuracy, version):
    before_label, before_conf = before
    if before_label != label and after[0] == label:
        mind = (f"Before your point, the model thought this spot was **{before_label}** "
                f"({pct(before_conf)} sure), so you just changed its mind.")
    elif before_label != label:
        mind = (f"Before your point, the model thought this spot was **{before_label}** "
                f"({pct(before_conf)} sure). It isn't convinced yet, but a point or two more "
                "nearby could change its mind.")
    elif before_conf < 1:
        mind = (f"The model already thought this was **{label}** ({pct(before_conf)} sure). "
                "You've made it more confident.")
    else:
        mind = (f"The model already thought this was **{label}** ({pct(before_conf)} sure), "
                "and your point backs that up.")
    pic = readme.picture(readme.image_url("dark", version), readme.image_url("light", version))
    return (
        f"Thanks @{user}! Added {article(label)} **{label}** point at ({fmt(x)}, {fmt(y)}).\n\n"
        f"{mind}\n\n"
        f"It's now trained by **{plural(visitors, 'visitor')}** · accuracy **{pct(accuracy)}**. "
        "The README updates in about a minute.\n\n"
        f"{pic}\n"
    )


def predict_comment(x, y, label, conf, k):
    return (
        f"At ({fmt(x)}, {fmt(y)}) the model says **{label}** ({pct(conf)} sure), "
        f"based on its {plural(k, 'nearest neighbour')}.\n\n"
        f"Think it's wrong? [Add a point]({readme.issue_link('train: ' + ('orange' if label == 'blue' else 'blue'))}) "
        "and change its mind.\n"
    )


def reject_comment(user, reason, retry_title):
    return (
        f"Sorry @{user}, I couldn't use this one. {reason}\n\n"
        f"{FORMAT_HELP}\n\n"
        f"[Try again with a pre-filled issue]({readme.issue_link(retry_title)})\n"
    )


# ---------- the pipeline ----------

def handle_event(event, rows, now):
    """Decide what to do with an issue event. Pure: returns a result dict and writes nothing.

    status is one of trained / predicted / rejected / ignored. For trained, `rows`
    holds the new data set.
    """
    ignored = {"status": "ignored", "label": "none", "issue": 0}
    issue = event.get("issue") if isinstance(event, dict) else None
    if not isinstance(issue, dict):
        return ignored
    author = issue.get("user") or {}
    login = author.get("login") or ""
    number = issue.get("number")
    if author.get("type") == "Bot" or not LOGIN.fullmatch(login) or type(number) is not int:
        return ignored
    kind = parse_title(issue.get("title"))
    if kind is None:
        return ignored

    action, label = kind
    result = {"label": label or "none", "issue": number, "user": login}
    retry_title = "predict" if action == "predict" else f"train: {label or 'blue'}"

    def reject(reason):
        return {**result, "status": "rejected", "comment": reject_comment(login, reason, retry_title)}

    if action == "train" and label is None:
        return reject("The label in the title must be `blue` or `orange`.")
    coords, error = parse_body(issue.get("body"))
    if error and error[0] == "format":
        return reject("I couldn't find both an `x` and a `y` value in the issue body.")
    if error:
        return reject(f"Both values need to be between 0 and 10 (you sent x={fmt(error[1])}, y={fmt(error[2])}).")

    x, y = coords
    points = as_points(rows)
    if action == "predict":
        pred, p = knn.predict(points, x, y)
        k = min(config.K, len(points))
        comment = predict_comment(x, y, pred, knn.confidence(pred, p), k)
        return {**result, "status": "predicted", "label": pred, "comment": comment}

    reset = rate_limit_reset(rows, login, now)
    if reset:
        return reject(
            f"You've already added {config.RATE_LIMIT} points in the last {config.RATE_WINDOW_HOURS} hours. "
            f"You can add another in {wait_text(reset - now)}."
        )
    dup = find_duplicate(rows, login, x, y)
    if dup:
        return reject(
            f"You already added a point at ({fmt(dup['x'])}, {fmt(dup['y'])}), which is almost the same spot. "
            "Try somewhere new!"
        )

    before_label, before_p = knn.predict(points, x, y)
    new_row = {"x": x, "y": y, "label": label, "user": login, "issue": number, "timestamp": format_time(now)}
    new_rows = apply_cap(rows + [new_row])
    new_points = as_points(new_rows)
    after = knn.predict(new_points, x, y)
    comment = train_comment(
        login, label, x, y,
        before=(before_label, knn.confidence(before_label, before_p)),
        after=after,
        visitors=visitor_count(new_rows),
        accuracy=knn.loo_accuracy(new_points),
        version=len(new_rows),
    )
    return {
        **result, "status": "trained", "comment": comment, "rows": new_rows,
        "commit": f"train: +1 {label} point by @{login} (#{number})",
    }


def render_all(rows, write=True):
    """Re-render both SVGs and the README block from `rows`. Returns {path: text}."""
    points = as_points(rows)
    newest = last_visitor(rows)
    visitors = visitor_count(rows)
    runs = render.region_runs(points)
    outputs = {
        path: render.render(points, theme, newest and (newest["x"], newest["y"], newest["label"]), visitors, runs)
        for theme, path in config.SVG_PATHS.items()
    }
    with open(config.README, newline="", encoding="utf-8") as f:
        text = f.read()
    block = readme.build_block(len(rows), visitors, newest and newest["user"], knn.loo_accuracy(points))
    outputs[config.README] = readme.replace_block(text, block)
    if write:
        for path, content in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", newline="", encoding="utf-8") as f:
                f.write(content)
    return outputs


def write_outputs(status, label="none", issue=0, comment=None, commit=None):
    config.OUT_DIR.mkdir(exist_ok=True)
    if comment is not None:
        (config.OUT_DIR / "comment.md").write_text(comment, encoding="utf-8")
    if commit is not None:
        (config.OUT_DIR / "commit.txt").write_text(commit + "\n", encoding="utf-8")
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"status={status}\nlabel={label}\nissue={issue}\n")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m model.train", description=__doc__.split("\n")[0])
    parser.add_argument("--event", help="event JSON file (default: $GITHUB_EVENT_PATH)")
    parser.add_argument("--dry-run", action="store_true", help="print the result without writing files")
    parser.add_argument("--render-only", action="store_true", help="re-render SVGs and README from the CSV")
    parser.add_argument("--seed-data", action="store_true", help="create data/points.csv with seed points")
    args = parser.parse_args(argv)

    if args.seed_data:
        if config.POINTS_CSV.exists():
            sys.exit(f"{config.POINTS_CSV} already exists; delete it first to regenerate seeds.")
        config.POINTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        save_rows(seed_rows())
        print(f"Wrote {len(seed_rows())} seed points to {config.POINTS_CSV}")
        return

    rows = load_rows()
    if args.render_only:
        render_all(rows, write=not args.dry_run)
        if not args.dry_run:
            write_outputs("rendered", commit="model: re-render plot")
        print(f"Rendered {len(rows)} points.")
        return

    event_path = args.event or os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        sys.exit("No event: pass --event FILE or set GITHUB_EVENT_PATH.")
    with open(event_path, encoding="utf-8") as f:
        event = json.load(f)

    result = handle_event(event, rows, datetime.now(timezone.utc))
    status = result["status"]

    if args.dry_run:
        print(f"status={status} label={result['label']} issue={result['issue']}")
        if "commit" in result:
            print(f"commit: {result['commit']}")
        if "comment" in result:
            print("\n--- comment ---\n" + result["comment"])
        if status == "trained":
            sizes = {p.name: len(t.encode()) for p, t in render_all(result["rows"], write=False).items()}
            print(f"would write: {sizes}")
        return

    if status == "trained":
        save_rows(result["rows"])
        render_all(result["rows"])
    write_outputs(status, result["label"], result["issue"], result.get("comment"), result.get("commit"))
    print(f"status={status}")


if __name__ == "__main__":
    main()
