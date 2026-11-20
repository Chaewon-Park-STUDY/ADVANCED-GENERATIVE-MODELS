"""
Summarizes result/proposal across seeds.

    py scripts\\summarize_proposal.py

Writes to result/proposal/summary/
    table_final.csv      last-round metrics, mean and sd over seeds
    table_rq.md          the same as markdown tables per RQ
    traj_<RQ>.png        mean trajectories over rounds
    rq2_lambda.png       final metrics against lam, random = 0 and top-k = right end
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent / "result" / "proposal"
OUT = ROOT / ("summary_quick" if "--quick" in sys.argv else "summary")
OUT.mkdir(parents=True, exist_ok=True)

frames = []
QUICK = "--quick" in sys.argv
for csv in ROOT.glob("*RQ*_s*/results.csv"):
    if csv.parent.name.startswith("quick_") != QUICK:
        continue
    m = re.match(r"(?:quick_)?(RQ\d)_(.+)_s(\d+)(_lite)?$", csv.parent.name)
    if not m:
        continue
    df = pd.read_csv(csv)
    # lite runs use different settings, so they are kept as separate configs
    df["rq"], df["config"], df["seed"] = m.group(1), m.group(2) + (m.group(4) or ""), int(m.group(3))
    frames.append(df)
if not frames:
    sys.exit(f"no results under {ROOT}")
df = pd.concat(frames, ignore_index=True)
if "rare_share_selected" in df:
    df["rare_lift"] = df["rare_share_selected"] - df["rare_share_pool"]
df.to_csv(OUT / "all_rounds.csv", index=False)

final_cols = ["recall", "min_class_recall", "class_tv", "precision", "val_loss", "fid"]
avg_cols = ["rare_lift", "artifact_in_selected", "gate_rate"]
rows = []
for (rq, cfg), g in df.groupby(["rq", "config"]):
    last = g[g["round"] == g.groupby("seed")["round"].transform("max")]
    sel = g[g["round"] > 0]
    row = {"rq": rq, "config": cfg, "n_seeds": g["seed"].nunique(), "rounds": int(g["round"].max())}
    for c in final_cols:
        if c in last and last[c].notna().any():
            row[c] = f"{last[c].mean():.3f} ± {last[c].std(ddof=0):.3f}"
    for c in avg_cols:
        if c in sel and sel[c].notna().any():
            per_seed = sel.groupby("seed")[c].mean()
            row[c + "_avg"] = f"{per_seed.mean():.3f} ± {per_seed.std(ddof=0):.3f}"
    rows.append(row)
tab = pd.DataFrame(rows).sort_values(["rq", "config"])
tab.to_csv(OUT / "table_final.csv", index=False)
with open(OUT / "table_rq.md", "w", encoding="utf-8") as f:
    for rq, g in tab.groupby("rq"):
        g = g.drop(columns="rq").fillna("")
        head = "| " + " | ".join(g.columns) + " |\n| " + " | ".join("---" for _ in g.columns) + " |\n"
        body = "".join("| " + " | ".join(str(v) for v in r) + " |\n" for r in g.values)
        f.write(f"## {rq}\n\n" + head + body + "\n")
print(tab.to_string(index=False))

# trajectories
for rq in sorted(df["rq"].unique()):
    sub = df[df["rq"].isin([rq, "RQ1"])] if rq != "RQ1" else df[df["rq"] == "RQ1"]
    if rq != "RQ1":  # compare against the RQ1 anchors
        anchors = [c + suf for c in ["topk_none_lam1.0_tl0.0", "ratio_none_lam1.0_tl0.0", "conf_ratio_gp_lam1.0_tl0.0"]
                   for suf in ["", "_lite"]]
        sub = sub[(sub["rq"] == rq) | sub["config"].isin(anchors)]
    metrics = [c for c in ["recall", "min_class_recall", "class_tv", "precision", "rare_lift"] if c in sub]
    fig, ax = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6))
    for (rq_, cfg), g in sub.groupby(["rq", "config"]):
        mean = g.groupby("round")[metrics].mean()
        for i, c in enumerate(metrics):
            ax[i].plot(mean.index, mean[c], marker=".", label=f"{rq_} {cfg}")
    for i, c in enumerate(metrics):
        ax[i].set_title(c)
        ax[i].set_xlabel("round")
        ax[i].grid(alpha=0.3)
    ax[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / f"traj_{rq}.png", dpi=130)
    plt.close(fig)

# RQ2: lambda curve
lam_rows = []
for cfg, g in df[df["rq"] != "RQ4"].groupby("config"):
    m = re.match(r"(random|ratio|topk)_none_lam([\d.]+)_tl0.0$", cfg)
    if not m:
        continue
    lam = {"random": 0.0, "topk": np.inf}.get(m.group(1), float(m.group(2)))
    last = g[g["round"] == g["round"].max()]
    lam_rows.append(dict(lam=lam, **{c: last[c].mean() for c in ["recall", "min_class_recall", "class_tv", "precision"]}))
if lam_rows:
    L = pd.DataFrame(lam_rows).sort_values("lam")
    xs = np.arange(len(L))
    labels = ["random" if v == 0 else ("top-k" if np.isinf(v) else f"λ={v:g}") for v in L["lam"]]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
    for i, c in enumerate(["recall", "min_class_recall", "class_tv", "precision"]):
        ax[i].plot(xs, L[c], marker="o")
        ax[i].set_xticks(xs, labels)
        ax[i].set_title(c)
        ax[i].grid(alpha=0.3)
    fig.suptitle("RQ2: how strongly marginal samples are used (final round)")
    fig.tight_layout()
    fig.savefig(OUT / "rq2_lambda.png", dpi=130)
    plt.close(fig)
print("saved to", OUT)
