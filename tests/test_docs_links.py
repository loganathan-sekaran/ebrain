"""Every relative Markdown link and #anchor in the docs must resolve (GitHub-style slugs)."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted([*ROOT.glob("*.md"), *(ROOT / "docs").rglob("*.md")])
FENCE = re.compile(r"^```.*?^```", re.S | re.M)
LINK = re.compile(r"\]\(([^)\s]+)\)")


def _slug(heading: str) -> str:
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for line in FENCE.sub("", path.read_text()).splitlines():
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            s = _slug(m.group(1))
            n = counts.get(s, 0)
            anchors.add(s if n == 0 else f"{s}-{n}")  # GitHub suffixes repeats with -1, -2, ...
            counts[s] = n + 1
    return anchors


@pytest.mark.parametrize("md", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_links_resolve(md: Path) -> None:
    broken = []
    for url in LINK.findall(FENCE.sub("", md.read_text())):
        if url.startswith(("http://", "https://", "mailto:")):
            continue
        path, _, anchor = url.partition("#")
        target = (md.parent / path).resolve() if path else md
        if not target.exists():
            broken.append(f"{url} (missing file)")
        elif anchor and target.suffix == ".md" and anchor not in _anchors(target):
            broken.append(f"{url} (missing anchor)")
    assert not broken, f"{md.name}: {broken}"
