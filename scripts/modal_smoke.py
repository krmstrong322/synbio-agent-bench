"""First check of the Modal oxDNA backend: simulate the 6HB reference design on Modal.

Prereqs (docs/modal.md): `uv sync --extra modal`, `uv run modal token new`,
`uv run modal deploy src/synbio_bench/origami/modal_app.py`.

Run:  uv run python scripts/modal_smoke.py                    # quick protocol, deployed GPU
      uv run python scripts/modal_smoke.py --protocol shape --gpu L4
      uv run python scripts/modal_smoke.py --compare runs/_smoke/local   # compare with a local/WSL run

Prints per-stage wall time (including transfer), the GPU and oxDNA version used, and the grading metrics.
With --compare, also prints the same metrics from a previous `simulate()` work dir (e.g. one made with
OXDNA_MODE=wsl), so you can check that Modal and the local GTX 1660 agree before switching graders.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

from synbio_bench.origami.modal_runner import ModalUnavailable
from synbio_bench.origami.oxdna import OxDNAConfig, config_from_env, simulate

ROOT = Path(__file__).resolve().parents[1]


def summary(r: dict) -> dict:
    m = r.get("metrics", {})
    return {
        "ok": r.get("ok"),
        "error": r.get("error"),
        "failed_stage": r.get("failed_stage"),
        "log_tail": r.get("log_tail"),
        "stage_seconds": r.get("stage_seconds"),
        "bp_fraction": m.get("bp_fraction_mean"),
        "rmsf_nm": m.get("rmsf_mean_nm"),
        "rmsd_to_design_nm": m.get("rmsd_mean_structure_to_design_nm"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", default="quick")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gpu", help="Modal GPU type for this run (default: the deployed one)")
    p.add_argument("--out", default=str(ROOT / "runs" / "_smoke" / "modal"))
    p.add_argument("--compare", help="work dir of an earlier simulate() run to compare against")
    args = p.parse_args()

    cfg = config_from_env(OxDNAConfig.from_dict(yaml.safe_load((ROOT / "configs" / "oxdna.yaml").read_text())))
    cfg.mode = "modal"
    if args.gpu:
        cfg.modal = {**cfg.modal, "gpu": args.gpu}

    design = Path(args.out).parent / "hb6.sc"
    design.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "hb6_honeycomb.py", str(design)], cwd=ROOT / "reference_designs", check=True)

    t0 = time.time()
    try:
        r = simulate(design, Path(args.out), cfg, protocol=args.protocol, seed=args.seed)
    except ModalUnavailable as e:
        sys.exit(f"Modal is not ready: {e}")
    print(f"total {round(time.time() - t0)} s")
    for log in sorted(Path(args.out).glob("log_*.txt")):
        header = log.read_text().splitlines()[0] if log.stat().st_size else ""
        if m := re.match(r"\[modal\] (.*)", header):
            print(f"{log.stem:>18}: {m.group(1)}")
    print(json.dumps({"modal": summary(r)}, indent=1))
    if args.compare:
        other = json.loads((Path(args.compare) / "result.json").read_text())
        print(json.dumps({"compare": summary(other)}, indent=1))
    sys.exit(0 if r.get("ok") else 1)


if __name__ == "__main__":
    main()
