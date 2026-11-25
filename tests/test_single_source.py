"""PLAN.md is the only place that holds gate thresholds and horizon month ranges.

A maintained doc that restates one fails this test. Legacy docs are exempt until they are
rewritten (TRK-17 to TRK-23) or archived; remove a doc from EXEMPT once it is rewritten.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

EXEMPT = {
    "docs/PLAN.md",  # the source of truth itself
    "docs/12-project-plan-review-and-recommendations.md",  # historical tracker
    "docs/13-design-review-2026-11.md",  # review record; quotes old thresholds
    "docs/01-feasibility-analysis.md",
    "docs/03-developmental-phases.md",
    "docs/04-implementation-roadmap.md",
    "docs/05-technical-stack.md",
    "docs/06-challenges-and-solutions.md",
    "docs/07-model-cloning-and-knowledge-transfer.md",
    "docs/08-tech-stack-decision-framework.md",
    "docs/10-computational-advantages-and-tool-use.md",
    "docs/11-philosophy-best-of-both-worlds.md",
    "docs/appendix-a-bio-inspired-neuron-research-branch.md",
    "docs/appendix-b-architecture-v1-unscheduled.md",
}

FENCE = re.compile(r"^```.*?^```", re.S | re.M)
COMPARATOR = r"(?:≥|≤|>=|<=|>|<)\s*\d"
RULES = {
    "horizon month range": re.compile(r"\bMonths?\s+\d+\s*[\u2013-]\s*\d+"),
    "margin restated": re.compile(r"\bby\s*(?:\*\*)?\s*(?:≥|>=|>)\s*\d+(?:\.\d+)?\s*points?"),
    "AUROC threshold restated": re.compile(r"AUROC\s*(?:≥|>=)\s*0?\.\d"),
}
GATE = re.compile(r"\bGate\s*[123]\b")


def _maintained() -> list[Path]:
    files = [*ROOT.glob("*.md"), *DOCS.rglob("*.md")]
    return sorted(
        f
        for f in files
        if str(f.relative_to(ROOT)) not in EXEMPT and "archive" not in f.relative_to(ROOT).parts
    )


def _violations(text: str) -> list[str]:
    found = []
    for raw in FENCE.sub("", text).splitlines():
        line = raw.lstrip("> ")
        for name, rule in RULES.items():
            if rule.search(line):
                found.append(f"{name}: {raw.strip()[:100]}")
        if GATE.search(line) and re.search(r"(?<![\w>-])" + COMPARATOR, line):
            found.append(f"gate threshold restated: {raw.strip()[:100]}")
    return found


@pytest.mark.parametrize("md", _maintained(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_restated_plan_numbers(md: Path) -> None:
    found = _violations(md.read_text())
    assert not found, f"{md.name} restates PLAN.md numbers; link to PLAN.md instead: {found}"


def test_exempt_files_exist() -> None:
    missing = [f for f in EXEMPT if not (ROOT / f).exists()]
    assert not missing, f"stale EXEMPT entries: {missing}"


@pytest.mark.parametrize(
    "line",
    [
        "Gate 1 needs ACC ≥ 2 points",
        "H1 runs in Months 1\u20133",
        "beats the control by ≥ 2 points",
        "AUROC ≥ 0.90",
    ],
)
def test_rules_catch_restatements(line: str) -> None:
    assert _violations(line)
