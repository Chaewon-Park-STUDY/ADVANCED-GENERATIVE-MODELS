"""
Runs the XSum selection grid in priority order; finished runs are skipped.

    py run_all_text.py --quick     # 1 round with a small pool, to measure time per round first
    py run_all_text.py                 # full grid with seed 0
    py run_all_text.py --seeds 1 2     # the same grid with seeds 1 and 2

Stopping is safe: run the same command again and finished runs are skipped,
while an interrupted run continues from the last finished round.
"""
import subprocess
import sys
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--only", default=None)
ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="seed 0 keeps the original run names")
a = ap.parse_args()
HERE = Path(__file__).resolve().parent
OUT = HERE / "text_runs"

GRID = [
    # Priority 1: replace, the setting of the original XSum experiment
    ("rouge_top", "none", "replace"),
    ("disc_top", "none", "replace"),
    ("ratio", "none", "replace"),
    ("conf_ratio", "mcd", "replace"),
    ("random", "none", "replace"),
    # Priority 2: accumulate
    ("disc_top", "none", "accumulate"),
    ("ratio", "none", "accumulate"),
    ("conf_ratio", "mcd", "accumulate"),
    # Priority 3: conformal without uncertainty
    ("conf_ratio", "none", "replace"),
]

JOBS = [(seed, g) for seed in a.seeds for g in GRID]
for seed, (method, unc, regime) in JOBS:
    tag = f"{method}_{unc}_{regime}" + ("" if seed == 0 else f"_s{seed}")
    if a.quick:
        tag = "quick_" + tag
    if a.only and a.only not in tag:
        continue
    log = OUT / tag / "log.txt"
    if log.exists() and "done" in log.read_text(encoding="utf-8"):
        print(f"[skip] {tag}")
        continue
    cmd = [sys.executable, str(HERE / "xsum_selection.py"), "--method", method, "--unc", unc,
           "--regime", regime, "--out", str(OUT), "--tag", tag,
           "--grad_accum", "2", "--seed", str(seed)]  # same effective batch of 16, half the peak GPU memory
    if a.quick:
        cmd += ["--rounds", "1", "--n_init", "500", "--n_pool", "500", "--n_test", "100"]
    print("[run]", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=HERE, check=False).returncode
    if rc != 0:
        print(f"[fail] {tag} exited with code {rc}; stopping here. Run the same command again to resume.")
        sys.exit(rc)
    if a.quick:
        break  # one quick run is enough to read the time per round

print("all requested runs finished")
