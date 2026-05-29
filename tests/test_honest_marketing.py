"""Honest-marketing enforcement (ship-and-yank lesson).

1. No affirmative overclaim phrase appears in README.md or the shipped source.
2. The ban patterns are proven NON-dead by a positive fixture each one must match.
3. README carries no un-measured placeholder numbers before the S6 measurement step.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# ERE-style, case-insensitive. These describe AFFIRMATIVE overclaims demoforge must not make.
BANNED_PATTERNS = {
    "improves_policy_or_success": r"improves?\s+\w*\s*(policy|success|behaviou?r|task)",
    "success_rate": r"success\s+rate",
    "policy_success": r"policy[\s-]success",
    "fully_automatic": r"fully\s+automatic",
    "permanent": r"\bpermanent",
    "worlds_first": r"world.?s\s+first",
    "state_of_the_art": r"state[\s-]of[\s-]the[\s-]art",
    "guarantees_sim2real": r"guarantees?\s+(sim2real|real[\s-]robot)",
}


# Files whose marketing/claims surface must stay clean.
def _scan_targets() -> list[Path]:
    targets = [ROOT / "README.md"]
    targets += sorted((ROOT / "src" / "demoforge").rglob("*.py"))
    return [p for p in targets if p.is_file()]


@pytest.mark.parametrize("name,pattern", list(BANNED_PATTERNS.items()))
def test_no_banned_phrase_in_shipped_text(name, pattern):
    rx = re.compile(pattern, re.IGNORECASE)
    hits = []
    for path in _scan_targets():
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if rx.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not hits, f"banned phrase {name!r} present:\n" + "\n".join(hits)


# Positive fixtures proving each pattern actually fires (no dead grep / BRE-vs-ERE scar).
POSITIVE_FIXTURES = {
    "improves_policy_or_success": "this tool improves policy learning",
    "success_rate": "raises the success rate by 30%",
    "policy_success": "guaranteed policy-success",
    "fully_automatic": "a fully automatic pipeline",
    "permanent": "a permanent solution",
    "worlds_first": "the world's first re-timer",
    "state_of_the_art": "state-of-the-art results",
    "guarantees_sim2real": "guarantees sim2real transfer",
}


@pytest.mark.parametrize("name,pattern", list(BANNED_PATTERNS.items()))
def test_ban_patterns_are_not_dead(name, pattern):
    rx = re.compile(pattern, re.IGNORECASE)
    assert rx.search(POSITIVE_FIXTURES[name]), f"pattern {name!r} is dead (matched nothing)"


def test_readme_has_no_unmeasured_placeholder_marker_after_s6():
    """Before S6 the README carries a <!--MEASURED@S6--> marker. After S6, generated numbers
    replace it. This test documents the gate: if the marker is gone, a results file must exist.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    results = list((ROOT / "results").glob("*.json")) if (ROOT / "results").is_dir() else []
    if "<!--MEASURED@S6-->" in readme:
        return  # pre-measurement state is acceptable
    assert results, "README claims measured numbers but results/*.json is missing"
