"""
Final proposal experiment: long-tailed MNIST, label-free (pooled) selection, accumulate.
Digits 7-9 are rare in the initial real data and play the role of marginal modes.

RQ1  Can the verifier identify marginal samples?            selection rule x verifier uncertainty
RQ2  How strongly should marginal samples be used?          ratio**lam, lam = 0.5, 1, 2, 5
RQ3  Can the verifier steer retraining toward the tail?     importance-weighted retraining, train_lam
RQ4  What happens when the verifier itself is biased?       the discriminator and calibration data are
                                                            long-tailed like the generator (ratio 0.1),
                                                            evaluation stays on the balanced test set

Finished runs are skipped, so the script can be stopped and restarted.

    cd ...\\MNIST
    py scripts\\run_proposal.py --seeds 0            # one seed first
    py scripts\\run_proposal.py --seeds 1 2          # remaining seeds
    py scripts\\run_proposal.py --seeds 0 --quick    # 2 rounds per run, timing check
    py scripts\\run_proposal.py --seeds 0 1 2 --core --gpu   # RQ1 core + RQ4 on a GPU, with FID
"""
import subprocess
import sys
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--quick", action="store_true")
ap.add_argument("--only", nargs="+", default=None, help="run only tags containing any of these texts, e.g. RQ4")
ap.add_argument("--core", action="store_true",
                help="RQ1 random/topk/ratio/conf_ratio-gp and the three RQ4 runs only")
ap.add_argument("--gpu", action="store_true", help="compute FID as well (slow on CPU)")
ap.add_argument("--lite", action="store_true",
                help="slow laptop: 5 core configs, 5 rounds, smaller pool and shorter training")
a = ap.parse_args()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "result" / "proposal"

COMMON = ["--regime", "accumulate", "--pooled", "1", "--longtail", "0.1", "--rounds", "10",
          "--pool", "5000", "--keep", "0.1", "--disc_real", "5000", "--disc_epochs", "10",
          "--calib", "1000", "--cvae_epochs", "100"]
if not a.gpu:
    COMMON += ["--no_fid", "1"]

# (rq, method, unc, lam, train_lam, disc_longtail)
GRID = [
    ("RQ1", "random", "none", 1.0, 0.0, 0.0),
    ("RQ1", "topk", "none", 1.0, 0.0, 0.0),
    ("RQ1", "ratio", "none", 1.0, 0.0, 0.0),
    ("RQ1", "conf_ratio", "none", 1.0, 0.0, 0.0),
    ("RQ1", "conf_ratio", "mcd", 1.0, 0.0, 0.0),
    ("RQ1", "conf_ratio", "gp", 1.0, 0.0, 0.0),
    ("RQ2", "ratio", "none", 0.5, 0.0, 0.0),
    ("RQ2", "ratio", "none", 2.0, 0.0, 0.0),
    ("RQ2", "ratio", "none", 5.0, 0.0, 0.0),
    ("RQ3", "conf_ratio", "gp", 1.0, 0.5, 0.0),
    ("RQ3", "conf_ratio", "gp", 1.0, 1.0, 0.0),
    ("RQ4", "topk", "none", 1.0, 0.0, 0.1),
    ("RQ4", "ratio", "none", 1.0, 0.0, 0.1),
    ("RQ4", "conf_ratio", "gp", 1.0, 0.0, 0.1),
]
CORE = {("RQ1", "random"), ("RQ1", "topk"), ("RQ1", "ratio"), ("RQ1", "conf_ratio-gp"),
        ("RQ4", "topk"), ("RQ4", "ratio"), ("RQ4", "conf_ratio-gp")}


def core_key(g):
    return (g[0], g[1] + ("-" + g[2] if g[1] == "conf_ratio" else ""))


if a.core:
    GRID = [g for g in GRID if core_key(g) in CORE and g[3] == 1.0 and g[4] == 0.0]

if a.lite:
    COMMON = [x for x in COMMON]  # copy, then override
    def setv(k, v):
        i = COMMON.index(k)
        COMMON[i + 1] = v
    setv("--rounds", "5")
    setv("--pool", "2000")
    setv("--disc_real", "2000")
    setv("--disc_epochs", "5")
    setv("--calib", "500")
    setv("--cvae_epochs", "100")  # 30 epochs left recall near 0
    GRID = [g for g in GRID if g[0] == "RQ4" or (g[1], g[2], g[3], g[4]) in {
        ("random", "none", 1.0, 0.0), ("topk", "none", 1.0, 0.0), ("ratio", "none", 1.0, 0.0),
        ("conf_ratio", "gp", 1.0, 0.0), ("conf_ratio", "gp", 1.0, 0.5)}]

for seed in a.seeds:
    for rq, method, unc, lam, tl, dlt in GRID:
        tag = f"{rq}_{method}_{unc}_lam{lam}_tl{tl}_s{seed}" + ("_lite" if a.lite else "")
        if a.quick:
            tag = "quick_" + tag
        if a.only and not any(o in tag for o in a.only):
            continue
        log = ROOT / tag / "log.txt"
        if log.exists() and "done" in log.read_text(encoding="utf-8"):
            print(f"[skip] {tag}")
            continue
        cmd = [sys.executable, str(HERE / "run_selection.py"), "--method", method, "--unc", unc,
               "--lam", str(lam), "--train_lam", str(tl), "--disc_longtail", str(dlt), "--seed", str(seed),
               "--tag", tag, "--out_root", str(ROOT)] + COMMON
        if a.quick:
            cmd += ["--rounds", "2"]
        print("[run]", tag, flush=True)
        ret = subprocess.run(cmd, cwd=HERE.parent).returncode
        if ret != 0:
            sys.exit(f"[stop] {tag} failed with exit code {ret}. Fix the error above and rerun the same command; "
                     f"finished runs are skipped and interrupted runs resume.")
print("all requested runs finished")
