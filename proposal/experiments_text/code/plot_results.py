"""
Collects results.csv files and draws the quality-diversity summary.

    py plot_results.py <result_dir> [--text]

MNIST : <result_dir> = MNIST/result/selection     quality = FID (lower better), diversity = recall
XSum  : <result_dir> = text_runs                  quality = ROUGE-1, diversity = distinct-2
"""
import sys
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
text = "--text" in sys.argv
rows = []
for csv in sorted(root.glob("*/results.csv")):
    if csv.parent.name.startswith(("smoke", "quick")) and "--all" not in sys.argv:
        continue
    df = pd.read_csv(csv)
    df["run"] = csv.parent.name
    rows.append(df)
if not rows:
    sys.exit("no results.csv found")
df = pd.concat(rows, ignore_index=True)
df["seed"] = df["run"].str.extract(r"_s(\d+)$", expand=False).fillna("0").astype(int)
df["run"] = df["run"].str.replace(r"_s\d+$", "", regex=True)
df.to_csv(root / "summary_all_rounds.csv", index=False)

if text:
    q, qlab, d, dlab = "rouge1", "ROUGE-1 (higher is better)", "distinct2", "distinct-2 (higher is more diverse)"
    extra = ["self_bleu", "len_std", "hallu_in_selected"]
    if "eval_green" in df.columns:  # watermark grid
        extra = ["self_bleu", "eval_green", "sel_green", "hallu_in_selected"]
else:
    q, qlab, d, dlab = "fid", "FID (lower is better)", "recall", "Recall (higher is more diverse)"
    extra = ["val_loss", "min_class_recall", "class_tv", "artifact_in_selected"]

cols = [c for c in [q, d] + extra if c in df.columns]
last = df.sort_values("round").groupby(["run", "seed"]).tail(1)
agg = last.groupby("run")[cols].agg(["mean", "std"])
out = pd.DataFrame({c: agg[(c, "mean")].map("{:.4f}".format) + " ± " + agg[(c, "std")].fillna(0).map("{:.4f}".format)
                    for c in cols})
out.insert(0, "n_seeds", last.groupby("run").size())
out.to_csv(root / "summary_last_round.csv")
print(out.to_string())
df = df.groupby(["run", "round"], as_index=False)[[c for c in df.columns if c not in ("run", "round", "seed") and pd.api.types.is_numeric_dtype(df[c])]].mean()

groups = {}
for run in df["run"].unique():
    nk = 1 if text else 3
    key = "_".join(run.split("_")[-nk:])
    groups.setdefault(key, []).append(run)

for key, runs in groups.items():
    fig, ax = plt.subplots(1, 2 + len(extra), figsize=(5 * (2 + len(extra)), 4))
    for run in runs:
        r = df[df.run == run].sort_values("round")
        lab = run[: -len(key) - 1]
        ax[0].plot(r[d], r[q], marker="o", label=lab)
        ax[0].annotate("start", (r[d].iloc[0], r[q].iloc[0]), fontsize=7)
        ax[1].plot(r["round"], r[q], marker=".", label=lab)
        for j, e in enumerate(extra):
            if e in r:
                ax[2 + j].plot(r["round"], r[e], marker=".", label=lab)
                ax[2 + j].set_title(e)
    ax[0].set_xlabel(dlab)
    ax[0].set_ylabel(qlab)
    ax[0].set_title("quality-diversity trajectory")
    ax[1].set_title(q)
    for x in ax:
        x.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle(key)
    fig.tight_layout()
    fig.savefig(root / f"plot_{key}.png", dpi=130)
    plt.close(fig)
print("saved plots to", root)
