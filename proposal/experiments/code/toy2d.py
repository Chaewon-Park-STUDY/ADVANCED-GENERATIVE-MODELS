"""Toy 2-D illustration: does top-k cut the tails, and does the conformal threshold keep them?

Real data: 8 Gaussians on a ring (sd 0.15); one of them is rare (3% of the mass).
Generator: a Gaussian mixture refit every generation (Accumulate: initial real + all selected data).
Each generation: draw 10,000 candidates, replace 5% with uniform noise (errors), keep 1,000.
Verifier: an MLP discriminator, real (held-out) vs. generated, refit every generation.
"""
import numpy as np, json, sys
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier

K, R, SD, RARE = 8, 2.0, 0.15, 0.03
ANG = np.arange(K) * 2 * np.pi / K
CENT = np.c_[R * np.cos(ANG), R * np.sin(ANG)]
W = np.r_[np.full(K - 1, (1 - RARE) / (K - 1)), RARE]  # last mode is rare


def real(n, rng):
    k = rng.choice(K, n, p=W)
    return CENT[k] + rng.normal(0, SD, (n, 2)), k


def mode_of(x):
    d = ((x[:, None, :] - CENT[None]) ** 2).sum(-1)
    return d.argmin(1), np.sqrt(d.min(1))


def run(method, seed, G=10, n_init=2000, pool=10000, keep=1000, err=0.05, alpha=0.01):
    rng = np.random.default_rng(seed)
    train, _ = real(n_init, rng)
    hist = []
    for g in range(G + 1):
        gmm = GaussianMixture(K, covariance_type="full", random_state=seed + g, max_iter=300).fit(train)
        s, _ = gmm.sample(20000)
        m, dist = mode_of(s)
        ok = dist < 4 * SD
        spread = float(np.mean([s[ok & (m == k)].std(0).mean() for k in range(K) if (ok & (m == k)).sum() > 20]))
        hist.append({"gen": g, "rare_share": float(np.mean(ok & (m == K - 1))), "spread": spread,
                     "err_share": float(np.mean(~ok))})
        if g == G:
            return hist, s[:3000]
        cand, _ = gmm.sample(pool)
        cand = cand[rng.permutation(pool)]
        n_err = int(err * pool)
        cand[:n_err] = rng.uniform(-3.2, 3.2, (n_err, 2))
        is_err = np.zeros(pool, bool); is_err[:n_err] = True
        if method == "random":
            idx = rng.choice(pool, keep, replace=False)
        else:
            xr, _ = real(4000, rng)
            xg, _ = gmm.sample(4000)
            clf = MLPClassifier((64, 64), max_iter=400, random_state=seed + g).fit(np.r_[xr, xg], np.r_[np.ones(4000), np.zeros(4000)])
            D = np.clip(clf.predict_proba(cand)[:, 1], 1e-4, 1 - 1e-4)
            if method == "topk":
                idx = np.argsort(-D)[:keep]
            else:  # conformal threshold + ratio sampling
                xc, _ = real(2000, rng)
                Dc = clf.predict_proba(xc)[:, 1]
                tau = np.quantile(Dc, alpha)
                passed = np.where(D >= tau)[0]
                ratio = D[passed] / (1 - D[passed])
                ratio = np.minimum(ratio, np.quantile(ratio, 0.99))
                idx = rng.choice(passed, min(keep, len(passed)), replace=False, p=ratio / ratio.sum())
        hist[-1]["sel_err"] = float(is_err[idx].mean())
        train = np.r_[train, cand[idx]]
    return hist, None


if __name__ == "__main__":
    out = {}
    for method in ["random", "topk", "ours"]:
        for seed in range(3):
            h, s = run(method, seed)
            out[f"{method}_{seed}"] = {"hist": h, "samples": s.tolist() if seed == 0 else None}
            print(method, seed, h[-1], flush=True)
    json.dump(out, open("/home/claude/proposal_deck/toy2d.json", "w"))
