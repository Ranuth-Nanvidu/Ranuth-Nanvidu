"""Rewrites the model section of README.md, between the markers only."""

from html import escape
from urllib.parse import quote

from . import config

START = "<!-- MODEL:START -->"
END = "<!-- MODEL:END -->"
DEFAULT_BODY = "x=5.0, y=5.0\n\n<!-- Change the numbers (0–10) and press Submit -->"


def issue_link(title, body=DEFAULT_BODY):
    """A pre-filled 'new issue' URL."""
    return (
        f"https://github.com/{config.REPO}/issues/new"
        f"?title={quote(title, safe='')}&body={quote(body, safe='')}"
    )


def image_url(theme, version):
    return f"https://raw.githubusercontent.com/{config.REPO}/{config.BRANCH}/assets/model-{theme}.svg?v={version}"


def picture(dark_src, light_src, alt="KNN decision boundary trained by visitors"):
    return (
        "<picture>\n"
        f'  <source media="(prefers-color-scheme: dark)" srcset="{dark_src}">\n'
        f'  <source media="(prefers-color-scheme: light)" srcset="{light_src}">\n'
        f'  <img alt="{alt}" src="{dark_src}" width="{config.SVG_WIDTH}">\n'
        "</picture>"
    )


def build_block(total, visitors, last_user, accuracy):
    """The generated HTML. Contains no blank lines, so GitHub treats it as one HTML block."""
    stats = [f"Trained by {visitors} visitor{'' if visitors == 1 else 's'}"]
    if last_user:
        stats.append(f"last update by @{last_user}")
    if accuracy is not None:
        stats.append(f"accuracy {round(accuracy * 100)}%")
    links = " · ".join(
        f'<a href="{escape(issue_link(title))}">{text}</a>'
        for title, text in (
            ("train: blue", "Add a blue point"),
            ("train: orange", "Add an orange point"),
            ("predict", "Ask for a prediction"),
        )
    )
    pic = picture(f"./assets/model-dark.svg?v={total}", f"./assets/model-light.svg?v={total}")
    pic = "\n".join("  " + line for line in pic.splitlines())
    stats_text = " · ".join(stats)
    return "\n".join([
        '<h3 align="center">Train my model</h3>',
        f'<p align="center">This KNN classifier (k = {config.K}) is trained entirely by visitors. '
        "Add a point and a GitHub Action retrains it within a minute.</p>",
        '<p align="center">',
        pic,
        "</p>",
        f'<p align="center"><code>{stats_text}</code></p>',
        f'<p align="center">{links}</p>',
    ])


def replace_block(text, block):
    """Swap whatever sits between the markers for `block`. Appends the section if the markers are missing."""
    newline = "\r\n" if "\r\n" in text else "\n"
    block = block.replace("\n", newline)
    start = text.find(START)
    end = text.find(END, start + len(START)) if start != -1 else -1
    if start == -1 or end == -1:
        return text.rstrip("\r\n") + newline * 2 + START + newline + block + newline + END + newline
    return text[: start + len(START)] + newline + block + newline + text[end:]
