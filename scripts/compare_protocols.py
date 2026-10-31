"""Check that a shorter grading protocol ranks designs like a longer one.

Simulates each design under each protocol and prints the grading metrics side by side.
Run: uv run python scripts/compare_protocols.py --task origami-rothemund/rectangle \
         --design reference_designs/out/rothemund_rect.sc --design reference_designs/out/rothemund_rect_twistfix.sc \
         --protocol shape --protocol standard
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from synbio_bench.origami.oxdna import OxDNAConfig, config_from_env, simulate

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--design", action="append", required=True)
    p.add_argument("--protocol", action="append", required=True)
    p.add_argument("--out", default=str(ROOT / "runs" / "_calibration" / "protocols"))
    args = p.parse_args()

    cfg = config_from_env(OxDNAConfig.from_dict(yaml.safe_load((ROOT / "configs" / "oxdna.yaml").read_text())))
    shape = yaml.safe_load((ROOT / "tasks" / f"{args.task}.yaml").read_text())["spec"].get("shape")
    rows = []
    for protocol in args.protocol:
        for design in args.design:
            work = Path(args.out) / f"{Path(design).stem}__{protocol}"
            t0 = time.time()
            r = simulate(ROOT / design, work, cfg, protocol=protocol, seed=1, shape=shape)
            m = r.get("metrics", {})
            row = {
                "design": Path(design).stem,
                "protocol": protocol,
                "ok": r.get("ok"),
                "minutes": round((time.time() - t0) / 60, 1),
                "bp_fraction": m.get("bp_fraction_mean"),
                "shape_iou": (m.get("shape_fidelity") or {}).get("iou"),
                "flatness": (m.get("shape_fidelity") or {}).get("flatness"),
                "rmsf_nm": m.get("rmsf_mean_nm"),
                "error": r.get("error"),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    (Path(args.out) / "comparison.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
