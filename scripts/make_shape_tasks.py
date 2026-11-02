"""Generate the shape x scaffold-size task matrix and the Rothemund literature tasks.

Run: uv run python scripts/make_shape_tasks.py
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SIZES = [500, 1000, 2000, 4000, 7000]
SHAPES = {
    "rectangle": ({"type": "rectangle", "aspect": 1.5}, "a rectangle with a 3:2 aspect ratio", SIZES),
    "triangle": ({"type": "triangle"}, "an equilateral triangle", SIZES),
    "disk": ({"type": "disk"}, "a disk (circle)", SIZES),
    "star": ({"type": "star", "points": 5, "inner_ratio": 0.5},
             "a five-pointed star whose inner vertices sit at half the radius of its tips", [2000, 4000, 7000]),
}
WEIGHTS = {"spec": 1 / 3, "sim": 1 / 3, "shape": 1 / 3}
REFERENCES = {("triangle", 4000): "reference_designs/sharp_triangle.py"}
MIN_IOU = 0.85


class _Literal(str):
    pass


yaml.SafeDumper.add_representer(_Literal, lambda d, s: d.represent_scalar("tag:yaml.org,2002:str", s, style="|"))


def shape_prompt(desc: str, lo: int, hi: int, extra: str = "") -> str:
    return _Literal(f"""\
Design a flat, single-layer DNA origami shaped like {desc}, built like Rothemund's scaffolded origami from
parallel helices joined by periodic crossovers. The sheet can be one raster of parallel helices, or several
domains whose helices run in different directions within the sheet (one scadnano helix group per domain),
joined by the scaffold and by bridging staples where they meet.
{extra}
Requirements:
- square lattice; a single flat layer: every helix lies in one plane (one lattice row per helix group, and
  groups rotated only within the plane of the sheet)
- a single scaffold strand that is a contiguous fragment of M13mp18, {lo}-{hi} nt long
- at least 95% of the scaffold base-paired to staples
- every staple 16-60 nt long
- neighbouring helices held together by staple crossovers along their whole length, and domains that meet
  at a seam joined across it
- shape fidelity is scored as the intersection-over-union (IoU) between your design's footprint and the ideal
  shape of the same area, at the best rotation and position. For the footprint each base pair is 0.34 nm long
  along its helix and neighbouring helices are 2.6 nm apart; a design with several helix groups is drawn from
  the groups' actual positions and orientations. Holes enclosed by the sheet count as filled, so the outline
  is what is judged. Aim for IoU >= {MIN_IOU}. The simulated oxDNA structure is also scored for shape: the
  sheet is unrolled along its own surface (following the helices, their crossovers and the seams between
  domains) and its outline compared with the target the same way. Bending or rolling up in solution does not
  lower this score, but helices or domains that pull apart or tear away from their neighbours do, so the
  crossovers and seams must hold the whole shape together.

Save your design under /designs/ and submit it with submit_design.
""")


def shape_tasks() -> None:
    out = ROOT / "tasks" / "origami-shapes"
    out.mkdir(parents=True, exist_ok=True)
    for name, (shape, desc, sizes) in SHAPES.items():
        for n in sizes:
            lo, hi = round(0.97 * n), min(round(1.03 * n), 7249)
            task = {
                "id": f"origami-shapes/{name}-{n}",
                "title": f"Flat {name}, ~{n}-nt scaffold",
                "difficulty": 1 + SIZES.index(n) // 2 + (name == "star"),
                "family": "shape-scaling",
                "params": {"shape": name, "scaffold_nt": n},
                "prompt": shape_prompt(desc, lo, hi),
                "spec": {
                    "grid": "square",
                    "planar": True,
                    "scaffold": {"count": 1, "source": "m13", "length": [lo, hi], "min_paired_fraction": 0.95},
                    "staple_length": [16, 60],
                    "shape": {**shape, "min_iou": MIN_IOU},
                },
                "weights": WEIGHTS,
            }
            if (name, n) in REFERENCES:
                task["reference"] = REFERENCES[(name, n)]
            (out / f"{name}-{n}.yaml").write_text(yaml.safe_dump(task, sort_keys=False, width=110))


ROTHEMUND = {
    "rectangle": dict(helices=24, bp=288, ref="rect", title="Rothemund rectangle (24 helices x 288 bp)"),
    "tall-rectangle": dict(helices=32, bp=224, ref="tallrect", title="Rothemund tall rectangle (32 helices x 224 bp)"),
}


def rothemund_tasks() -> None:
    out = ROOT / "tasks" / "origami-rothemund"
    out.mkdir(parents=True, exist_ok=True)
    for name, r in ROTHEMUND.items():
        w, h = r["bp"] * 0.34, r["helices"] * 2.6
        extra = (f"\nThis is a recreation of the {r['title']} from Rothemund, Nature 2006: {r['helices']} parallel helices of "
                 f"~{r['bp']} bp each (about {w:.0f} nm x {h:.0f} nm), folded from nearly the whole M13mp18 genome.\n")
        task = {
            "id": f"origami-rothemund/{name}",
            "title": r["title"],
            "difficulty": 4,
            "family": "literature",
            "reference": f"reference_designs/rothemund.py {r['ref']}",
            "prompt": shape_prompt(f"a {w:.0f} nm x {h:.0f} nm rectangle", 6600, 7249, extra),
            "spec": {
                "grid": "square",
                "planar": True,
                "num_helices": r["helices"],
                "helix_bp": [r["bp"] - 12, r["bp"] + 12],
                "scaffold": {"count": 1, "source": "m13", "length": [6600, 7249], "min_paired_fraction": 0.95},
                "staple_length": [16, 60],
                "shape": {"type": "rectangle", "aspect": round(w / h, 4), "min_iou": 0.9},
            },
            "weights": WEIGHTS,
        }
        (out / f"{name}.yaml").write_text(yaml.safe_dump(task, sort_keys=False, width=110))


if __name__ == "__main__":
    shape_tasks()
    rothemund_tasks()
    print(sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "tasks").rglob("*.yaml")))
