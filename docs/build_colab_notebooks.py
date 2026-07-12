"""Generate Colab-ready notebooks from the tutorial pages.

Each tutorial under `docs/tutorials/` is a Quarto page executed as HTML for the
website. Google Colab opens notebooks from the GitHub repository rather than from
the published site, so this script writes a runnable notebook copy of each
tutorial to `docs/_notebooks/` and adds an "Open in Colab" badge to the top of the
page. Every notebook opens with a setup cell that installs `heormodel` from PyPI.
The discrete-event simulation replication also imports model code and reads a
mortality table from the `examples/` folder, so its setup cell clones the
repository for those files after installing the package.

Run it with `uv run python docs/build_colab_notebooks.py`. The command is
idempotent: it rewrites the badge region in each page and the notebooks in place,
so a second run over unchanged tutorials produces no diff.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat
import yaml

DOCS = Path(__file__).parent
TUTORIALS = DOCS / "tutorials"
NOTEBOOKS = DOCS / "_notebooks"

GH_USER = "pedroliman"
GH_REPO = "heormodel"
GH_REF = "main"  # branch the notebooks install from and Colab opens from
SITE_URL = "https://pedroliman.github.io/heormodel/"
BADGE_IMG = "https://colab.research.google.com/assets/colab-badge.svg"
REPO_GIT = f"https://github.com/{GH_USER}/{GH_REPO}.git"

BADGE_START = "<!-- colab-badge:start -->"
BADGE_END = "<!-- colab-badge:end -->"

# Optional install extras keyed by tutorial file stem. A tutorial not listed here
# installs the base package.
EXTRAS = {"des": "des", "calibration-workflow": "calibration"}

# Tutorials whose code loads files from the repository, so the notebook clones it.
CLONE = {"mdm-des"}


def install_spec(stem: str) -> str:
    """Return the pip requirement string that installs the tutorial's package."""
    extra = EXTRAS.get(stem)
    return f"heormodel[{extra}]" if extra else "heormodel"


def setup_code(stem: str) -> str:
    """Return the first code cell that makes the tutorial runnable in Colab."""
    if stem in CLONE:
        return (
            "# Install heormodel from PyPI, then fetch the model code and mortality\n"
            "# table this tutorial reads from the examples folder.\n"
            f"%pip install -q {install_spec(stem)}\n"
            f"!git clone -q --depth 1 {REPO_GIT}\n"
            f"%cd {GH_REPO}"
        )
    return (
        "# Install heormodel from PyPI.\n"
        f"%pip install -q {install_spec(stem)}"
    )


def split_front_matter(text: str) -> tuple[dict, str]:
    """Return the parsed YAML header and the body of a Quarto page."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.index("\n---", 4)
    header = yaml.safe_load(text[4:end]) or {}
    body = text[end + 4 :].lstrip("\n")
    return header, body


def strip_badge_region(body: str) -> str:
    """Remove a previously inserted badge region so re-runs stay idempotent."""
    pattern = re.compile(
        re.escape(BADGE_START) + r".*?" + re.escape(BADGE_END) + r"\n*", re.DOTALL
    )
    return pattern.sub("", body).lstrip("\n")


def rewrite_links(markdown: str) -> str:
    """Point cross-references to other pages at the published site.

    Links in the tutorials are written relative to `docs/tutorials/`. Colab has no
    site to resolve them against, so rewrite each `.qmd` target to its rendered
    page under the site URL.
    """

    def replace(match: re.Match[str]) -> str:
        target, fragment = match.group(1), match.group(2) or ""
        rel = (Path("tutorials") / target).as_posix()
        parts: list[str] = []
        for part in rel.split("/"):
            if part == "..":
                parts.pop()
            elif part not in ("", "."):
                parts.append(part)
        page = "/".join(parts)[: -len(".qmd")] + ".html"
        return f"]({SITE_URL}{page}{fragment})"

    return re.sub(r"\]\(([^)]+\.qmd)(#[^)]*)?\)", replace, markdown)


def parse_cells(body: str) -> list[tuple[str, str]]:
    """Split a tutorial body into ("markdown"|"code", text) cells.

    Executable `python` blocks become code cells; everything else, including plain
    fenced blocks and tables, stays in markdown.
    """
    cells: list[tuple[str, str]] = []
    lines = body.split("\n")
    buffer: list[str] = []

    def flush_markdown() -> None:
        text = "\n".join(buffer).strip("\n")
        buffer.clear()
        if text.strip():
            cells.append(("markdown", rewrite_links(text)))

    index = 0
    while index < len(lines):
        line = lines[index]
        if line.strip() == "```{python}":
            flush_markdown()
            index += 1
            code: list[str] = []
            while index < len(lines) and lines[index].strip() != "```":
                code.append(lines[index])
                index += 1
            cells.append(("code", "\n".join(code).strip("\n")))
        else:
            buffer.append(line)
        index += 1
    flush_markdown()
    return cells


def colab_url(stem: str) -> str:
    """Return the Colab link that opens the committed notebook for a tutorial."""
    path = f"docs/_notebooks/{stem}.ipynb"
    return f"https://colab.research.google.com/github/{GH_USER}/{GH_REPO}/blob/{GH_REF}/{path}"


def badge_markdown(stem: str) -> str:
    """Return the badge image link that opens the tutorial in Colab."""
    return f"[![Open In Colab]({BADGE_IMG})]({colab_url(stem)})"


def build_notebook(stem: str, title: str, cells: list[tuple[str, str]]) -> nbformat.NotebookNode:
    """Assemble the notebook: title and badge, setup cell, then tutorial cells."""
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    ordered: list[tuple[str, str]] = [
        ("markdown", f"# {title}\n\n{badge_markdown(stem)}"),
        ("code", setup_code(stem)),
        *cells,
    ]
    for index, (kind, text) in enumerate(ordered):
        make = nbformat.v4.new_code_cell if kind == "code" else nbformat.v4.new_markdown_cell
        cell = make(text)
        # Deterministic ids keep re-runs byte-identical; nbformat assigns random
        # ones otherwise, which would make every build produce a spurious diff.
        cell["id"] = f"cell-{index:02d}"
        nb.cells.append(cell)
    return nb


def insert_badge(header_text: str, body: str, stem: str) -> str:
    """Return the page with the badge region placed above the tutorial body."""
    region = f"{BADGE_START}\n\n{badge_markdown(stem)}\n\n{BADGE_END}"
    return f"{header_text}\n\n{region}\n\n{body}"


def process(page: Path) -> bool:
    """Rebuild one tutorial's notebook and badge. Return True if it has code."""
    text = page.read_text()
    header, raw_body = split_front_matter(text)
    body = strip_badge_region(raw_body)
    cells = parse_cells(body)
    stem = page.stem

    if not any(kind == "code" for kind, _ in cells):
        # No runnable code, so no notebook and no badge (for example an index page).
        if raw_body != body:
            end = text.index("\n---", 4) + 4
            page.write_text(text[:end] + "\n\n" + body)
        return False

    end = text.index("\n---", 4) + 4
    header_text = text[:end]
    page.write_text(insert_badge(header_text, body, stem))

    nb = build_notebook(stem, str(header.get("title", stem)), cells)
    NOTEBOOKS.mkdir(exist_ok=True)
    nbformat.write(nb, NOTEBOOKS / f"{stem}.ipynb")
    return True


def main() -> None:
    built = []
    for page in sorted(TUTORIALS.glob("*.qmd")):
        if process(page):
            built.append(page.stem)
    print(f"Wrote {len(built)} notebooks to {NOTEBOOKS.relative_to(DOCS.parent)}:")
    for stem in built:
        print(f"  {stem}.ipynb")


if __name__ == "__main__":
    main()
