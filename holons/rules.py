"""Catalogue of Life-like (outer-totalistic) rules, tagged by regime.

Each rule is a birth/survive mask pair (see life.step_rule) with a `regime`
label taken from the established behaviour of the rule in the Life-like
cellular-automata literature:

  - "trivial"  : the dynamics collapses to a constant field (closure anchor);
  - "ordered"  : growth freezes into static structure (mazes, flakes);
  - "complex"  : Life-class rules with gliders, oscillators, long transients;
  - "chaotic"  : explosive / turbulent rules with no stable coarse structure.

The regime labels are PRIORS from the literature, not ground truth. The
closure experiment (closure.py) measures an intrinsic defect per rule; the
question is whether that defect separates these labelled groups. The labels
are inputs to that test, not its conclusion.

B0 rules are excluded: with an open (zero-padded) boundary they birth the
infinite dead exterior (see life.step_rule).
"""
from __future__ import annotations

from typing import NamedTuple


class LifeRule(NamedTuple):
    name: str
    birth: frozenset[int]
    survive: frozenset[int]
    regime: str


def format_rule(birth: frozenset[int], survive: frozenset[int]) -> str:
    """Return the standard `Bxxx/Sxxx` string for a birth/survive mask pair."""
    b = "".join(str(k) for k in sorted(birth))
    s = "".join(str(k) for k in sorted(survive))
    return "B" + b + "/S" + s


def _rule(name, birth, survive, regime) -> LifeRule:
    return LifeRule(name, frozenset(birth), frozenset(survive), regime)


CATALOGUE: list[LifeRule] = [
    # Trivial: every live cell dies and nothing is born -> constant 0 field.
    _rule("Vanish", [], [], "trivial"),

    # Ordered: growth freezes into static maze / flake structure.
    _rule("LifeWithoutDeath", [3], [0, 1, 2, 3, 4, 5, 6, 7, 8], "ordered"),
    _rule("Maze", [3], [1, 2, 3, 4, 5], "ordered"),
    _rule("Mazectric", [3], [1, 2, 3, 4], "ordered"),

    # Complex: Life-class rules with gliders and long transients.
    _rule("Life", [3], [2, 3], "complex"),
    _rule("HighLife", [3, 6], [2, 3], "complex"),
    _rule("DayAndNight", [3, 6, 7, 8], [3, 4, 6, 7, 8], "complex"),
    _rule("Morley", [3, 6, 8], [2, 4, 5], "complex"),

    # Chaotic: explosive / turbulent, no persistent coarse structure.
    _rule("Seeds", [2], [], "chaotic"),
    _rule("Replicator", [1, 3, 5, 7], [1, 3, 5, 7], "chaotic"),
    _rule("ThreeFourLife", [3, 4], [3, 4], "chaotic"),
]
