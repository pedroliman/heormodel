"""Copy the changelog and roadmap into the site before rendering.

Runs as a Quarto pre-render script. The repository files stay canonical;
these copies exist only so the site can render them, so they are
gitignored. Links that point outside the site are rewritten to GitHub.
"""

from __future__ import annotations

from pathlib import Path

DOCS = Path(__file__).parent
ROOT = DOCS.parent
BLOB = "https://github.com/pedroliman/heval/blob/main/"


def rewrite_links(text: str) -> str:
    """Point links that leave the site at the GitHub repository."""
    text = text.replace("](../guidance/", f"]({BLOB}guidance/")
    text = text.replace("](RELEASING.md)", f"]({BLOB}RELEASING.md)")
    return text


def promote_title(text: str) -> str:
    """Lift a leading level-1 heading into YAML front matter."""
    lines = text.split("\n")
    if not lines or not lines[0].startswith("# "):
        return text
    title = lines[0][2:].strip()
    body = "\n".join(lines[1:]).lstrip("\n")
    return f'---\ntitle: "{title}"\n---\n\n{body}'


def convert(src: Path, dest: Path) -> None:
    dest.write_text(promote_title(rewrite_links(src.read_text())))


def main() -> None:
    convert(ROOT / "CHANGELOG.md", DOCS / "changelog.md")

    out = DOCS / "roadmap"
    out.mkdir(exist_ok=True)
    for src in sorted((ROOT / "roadmap").glob("*.md")):
        convert(src, out / ("index.md" if src.name == "README.md" else src.name))

    done_out = out / "done"
    done_out.mkdir(exist_ok=True)
    for src in sorted((ROOT / "roadmap" / "done").glob("*.md")):
        convert(src, done_out / src.name)


if __name__ == "__main__":
    main()
