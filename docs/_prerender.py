"""Copy the changelog into the site before rendering.

Runs as a Quarto pre-render script. The repository file stays canonical;
the copy exists only so the site can render it, so it is gitignored.
Links that point outside the site are rewritten to GitHub.
"""

from __future__ import annotations

from pathlib import Path

DOCS = Path(__file__).parent
ROOT = DOCS.parent
BLOB = "https://github.com/pedroliman/heormodel/blob/main/"


def rewrite_links(text: str) -> str:
    """Point links that leave the site at the GitHub repository."""
    text = text.replace("](RELEASING.md)", f"]({BLOB}RELEASING.md)")
    return text


def promote_title(text: str) -> str:
    """Lift a leading level-1 heading into YAML front matter.

    The changelog is a standalone page in no sidebar, so `sidebar: false` keeps a
    section's sidebar from showing beside it.
    """
    lines = text.split("\n")
    if not lines or not lines[0].startswith("# "):
        return text
    title = lines[0][2:].strip()
    body = "\n".join(lines[1:]).lstrip("\n")
    return f'---\ntitle: "{title}"\nsidebar: false\n---\n\n{body}'


def convert(src: Path, dest: Path) -> None:
    dest.write_text(promote_title(rewrite_links(src.read_text())))


def main() -> None:
    convert(ROOT / "CHANGELOG.md", DOCS / "changelog.md")


if __name__ == "__main__":
    main()
