"""Shared helpers for the reference (positive-control) designs.

These are deliberately NOT exposed to the agent: they automate scaffold routing
and stapling, which is exactly the skill the benchmark measures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import scadnano as sc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from synbio_bench.origami.geometry import crossover_positions  # noqa: E402


def pick_near(options: list[int], target: int) -> int:
    return min(options, key=lambda o: abs(o - target))


def periodic(options: list[int], period: int, lo: int, hi: int) -> list[int]:
    """Choose the longest arithmetic run with the given period from candidate offsets inside [lo, hi]."""
    inside = [o for o in options if lo <= o <= hi]
    best: list[int] = []
    for start in inside:
        run = [o for o in range(start, hi + 1, period) if o in inside]
        if len(run) > len(best):
            best = run
    return best


def add_staple_crossovers(design: sc.Design, pairs: list[tuple[int, int]], scaffold_forward: dict[int, bool],
                          period: int, margin: int = 6) -> None:
    for a, b in pairs:
        staple_fwd = not scaffold_forward[a]
        pos = crossover_positions(design, a, b)[f"strand_on_helix_{'forward' if staple_fwd else 'reverse'}"]
        da = [d for d in _domains_on(design, a) if d.forward == staple_fwd]
        db = [d for d in _domains_on(design, b) if d.forward != staple_fwd]
        lo = max(min(d.start for d in da), min(d.start for d in db)) + margin
        hi = min(max(d.end for d in da), max(d.end for d in db)) - margin
        for o in periodic(pos["full_crossover_offsets"], period, lo, hi):
            design.add_full_crossover(helix=a, helix2=b, offset=o, forward=staple_fwd)


def _domains_on(design: sc.Design, helix: int) -> list[sc.Domain]:
    return [d for s in design.strands for d in s.domains if isinstance(d, sc.Domain) and d.helix == helix]


def break_long_staples(design: sc.Design, max_len: int = 49, min_domain: int = 14) -> None:
    """Nick staples in the middle of their longest domain until every staple is <= max_len."""
    while True:
        long = [s for s in design.strands if not s.is_scaffold and s.dna_length() > max_len]
        if not long:
            return
        strand = long[0]
        doms = [d for d in strand.domains if isinstance(d, sc.Domain) and d.dna_length() >= min_domain]
        if not doms:
            return
        d = max(doms, key=lambda d: d.dna_length())
        design.add_nick(helix=d.helix, offset=(d.start + d.end) // 2, forward=d.forward)


def merge_short_staples(design: sc.Design, min_len: int = 18, max_len: int = 60) -> None:
    """Ligate staples shorter than min_len onto an end-to-end neighbour on the same helix."""
    changed = True
    while changed:
        changed = False
        for s in [s for s in design.strands if not s.is_scaffold and s.dna_length() < min_len]:
            for d, at_5p in ((s.first_domain(), True), (s.last_domain(), False)):
                # the nick sits at this staple's 5' or 3' boundary on its helix
                if d.forward:
                    nick = d.start if at_5p else d.end
                else:
                    nick = d.end if at_5p else d.start
                other = [t for t in design.strands if t is not s and not t.is_scaffold
                         for e in (t.first_domain(), t.last_domain())
                         if e.helix == d.helix and e.forward == d.forward and nick in (e.start, e.end)]
                if other and other[0].dna_length() + s.dna_length() <= max_len:
                    design.ligate(helix=d.helix, offset=nick, forward=d.forward)
                    changed = True
                    break
            if changed:
                break
