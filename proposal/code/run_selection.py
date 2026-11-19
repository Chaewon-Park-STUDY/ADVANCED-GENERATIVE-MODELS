"""
Selection-rule experiment on top of the Escaping Model Collapse MNIST pipeline.

Place this file in  Verified-Synthetic-Data-main/MNIST/scripts/  and run from the MNIST folder:
    py scripts/run_selection.py --method conf_ratio --unc mcd --regime accumulate --tag my_run

Selection rules (--method)
    random      : uniform subsample, no verifier
    topk        : keep top-q by discriminator logit  (paper baseline)
    ratio       : sample proportional to D/(1-D) = exp(logit)  (density-ratio calibration)
    conf_ratio  : conformal gate on held-out real scores, then ratio sampling inside the gate

Verifier uncertainty (--unc)
    none  : single discriminator
    mcd   : MC dropout, T forward passes
    ens   : deep ensemble of M discriminators
    gp    : random-Fourier-feature GP head with Laplace posterior on the discriminator
            features (SNGP-style approximation of a GP classifier)

Pooling (--pooled 1)
    Selection is done jointly over all classes and the next generator samples labels with the
    class frequencies of its own training set. This removes the per-digit guard of the original
    paper, which is the label-free setting discussed in the review.
"""
import os
import sys
import json
import math
import time
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torchvision import datasets, transforms

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent / "src"
sys.path.append(str(SRC_DIR))
import models  # noqa: E402
import train_helper  # noqa: E402
import utils  # noqa: E402
try:  # FID needs torchmetrics; CPU runs with --no_fid 1 do not
    from fid import calculate_fid_score  # noqa: E402
except ImportError:
    calculate_fid_score = None

# ----------------------------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--method", default="conf_ratio", choices=["random", "topk", "ratio", "conf_ratio"])
p.add_argument("--unc", default="none", choices=["none", "mcd", "ens", "gp"])
p.add_argument("--regime", default="replace", choices=["replace", "accumulate"])
p.add_argument("--pooled", type=int, default=1, help="1: select jointly over classes, 0: per class")
p.add_argument("--rounds", type=int, default=10)
p.add_argument("--init_size", type=int, default=500)
p.add_argument("--longtail", type=float, default=0.0,
               help="0: balanced init. r>0: class c gets weight r**(c/9), e.g. 0.1 makes digit 9 ten times rarer than 0")
p.add_argument("--disc_longtail", type=float, default=0.0,
               help="RQ4 biased verifier: 0 = balanced real reference, r>0 = long-tailed reference like --longtail")
p.add_argument("--pool", type=int, default=20000, help="synthetic candidates per round")
p.add_argument("--keep", type=float, default=0.1, help="fraction of the pool kept")
p.add_argument("--alpha", type=float, default=0.01, help="conformal miscoverage for real samples")
p.add_argument("--conf_by_class", type=int, default=1, help="Mondrian conformal threshold per class")
p.add_argument("--temp_scale", type=int, default=1, help="temperature-scale the discriminator on held-out data")
p.add_argument("--lam", type=float, default=1.0,
               help="ratio strength: weight = ratio**lam. 0 = random, 1 = exact density ratio, large = close to top-k")
p.add_argument("--train_lam", type=float, default=0.0,
               help="RQ3: importance-weighted retraining. Selected samples are replayed in training with "
                    "probability proportional to ratio**train_lam. 0 = plain retraining")
p.add_argument("--kappa", type=float, default=1.0, help="uncertainty multiplier")
p.add_argument("--clip_q", type=float, default=0.99, help="clip density ratio at this quantile")
p.add_argument("--art_frac", type=float, default=0.05, help="fraction of injected artifacts in the pool")
p.add_argument("--disc_real", type=int, default=10000)
p.add_argument("--disc_epochs", type=int, default=15)
p.add_argument("--calib", type=int, default=2000)
p.add_argument("--mc_T", type=int, default=20)
p.add_argument("--ens_M", type=int, default=5)
p.add_argument("--gp_D", type=int, default=1024)
p.add_argument("--cvae_epochs", type=int, default=200)
p.add_argument("--fid_n", type=int, default=6000)
p.add_argument("--fid_real", type=int, default=10000)
p.add_argument("--no_fid", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", default=None)
p.add_argument("--resume", type=int, default=1, help="continue an interrupted run from its last finished round")
p.add_argument("--out_root", default=None, help="parent folder of the run folder")
p.add_argument("--data_root", default="./data")
args = p.parse_args()

tag = args.tag or (f"{args.method}_{args.unc}_{args.regime}_p{args.pooled}_lt{args.longtail}"
                   f"_lam{args.lam}_tl{args.train_lam}_s{args.seed}")
OUT = (Path(args.out_root) if args.out_root else THIS_DIR.parent / "result" / "selection") / tag
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "config.json", "w") as f:
    json.dump(vars(args), f, indent=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
NC = 10


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT / "log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ----------------------------------------------------------------------------------------------
# Data: disjoint real splits for init / discriminator / calibration
# ----------------------------------------------------------------------------------------------
full = datasets.MNIST(args.data_root, train=True, download=True, transform=transforms.ToTensor())
test = datasets.MNIST(args.data_root, train=False, download=True, transform=transforms.ToTensor())
X_full = full.data.float().div(255.).unsqueeze(1)  # [60000,1,28,28]
Y_full = full.targets.clone()
X_test = test.data.float().div(255.).unsqueeze(1)
Y_test = test.targets.clone()
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=256, shuffle=False)

g = torch.Generator().manual_seed(args.seed)
perm_by_class = {c: torch.nonzero(Y_full == c).squeeze(1)[torch.randperm(int((Y_full == c).sum()), generator=g)]
                 for c in range(NC)}


def take(per_class_counts, offset):
    idx = []
    for c in range(NC):
        idx.append(perm_by_class[c][offset[c]: offset[c] + per_class_counts[c]])
        offset[c] += per_class_counts[c]
    return torch.cat(idx)


offset = [0] * NC
if args.longtail > 0:
    w = np.array([args.longtail ** (c / (NC - 1)) for c in range(NC)])
    init_counts = np.maximum(1, np.round(args.init_size * w / w.sum())).astype(int).tolist()
else:
    init_counts = [args.init_size // NC] * NC
idx_init = take(init_counts, offset)


def ref_counts(total):
    """Class counts of the verifier's real reference. Balanced by default; with --disc_longtail r the
    verifier only sees a long-tailed slice of the real distribution (biased verifier, RQ4)."""
    if args.disc_longtail > 0:
        w = np.array([args.disc_longtail ** (c / (NC - 1)) for c in range(NC)])
        return np.maximum(2, np.round(total * w / w.sum())).astype(int).tolist()
    return [total // NC] * NC


idx_disc = take(ref_counts(args.disc_real), offset)  # real reference used to train the discriminator
idx_cal = take(ref_counts(args.calib), offset)       # conformal calibration, same reference distribution
idx_T = take(ref_counts(args.calib), offset)         # separate split for temperature scaling

X_init, Y_init = X_full[idx_init], Y_full[idx_init]
X_disc_real = X_full[idx_disc]
X_cal, Y_cal = X_full[idx_cal], Y_full[idx_cal]
X_T = X_full[idx_T]
log(f"init class counts: {init_counts}")
log(f"verifier reference class counts: {ref_counts(args.disc_real)}")


# ----------------------------------------------------------------------------------------------
# Generator helpers
# ----------------------------------------------------------------------------------------------
def train_cvae(X, Y, name, weights=None):
    m = models.CVAE(input_dim=784, label_dim=NC, latent_dim=20, name=name, arch="conv").to(device)
    if weights is None:
        loader = DataLoader(TensorDataset(X, Y), batch_size=128, shuffle=True)
    else:  # importance-weighted retraining through weighted resampling every epoch
        sampler = torch.utils.data.WeightedRandomSampler(weights.double(), num_samples=len(X), replacement=True)
        loader = DataLoader(TensorDataset(X, Y), batch_size=128, sampler=sampler)
    train_helper.train_model(m, loader, device, epochs=args.cvae_epochs, lr=1e-3, patience=5, verbose=False)
    m.eval()
    return m


@torch.no_grad()
def generate(model, n, class_probs, binary=True):
    labels = torch.multinomial(torch.as_tensor(class_probs, dtype=torch.float), n, replacement=True)
    xs = []
    for c in range(NC):
        k = int((labels == c).sum())
        if k == 0:
            continue
        xs.append((model.sample_x_given_y(c, k, binary_format=binary).view(-1, 1, 28, 28).cpu(), torch.full((k,), c)))
    X = torch.cat([a for a, _ in xs])
    Y = torch.cat([b for _, b in xs])
    perm = torch.randperm(len(X))
    return X[perm], Y[perm]


def make_artifacts(X, Y, frac):
    """Overlay two digits of different classes: stroke-like images that are not real digits."""
    n = int(len(X) * frac)
    if n == 0:
        return X, Y, torch.zeros(len(X), dtype=torch.bool)
    X = X.clone()
    idx = torch.randperm(len(X))[:n]
    partner = torch.randint(0, len(X), (n,))
    shift = torch.randint(-4, 5, (n, 2))
    for i in range(n):
        other = torch.roll(X[partner[i]], shifts=(int(shift[i, 0]), int(shift[i, 1])), dims=(1, 2))
        X[idx[i]] = torch.maximum(X[idx[i]], other)
    is_art = torch.zeros(len(X), dtype=torch.bool)
    is_art[idx] = True
    return X, Y, is_art


# ----------------------------------------------------------------------------------------------
# Verifier: discriminator with optional dropout, ensemble, GP head
# ----------------------------------------------------------------------------------------------
class Disc(nn.Module):
    def __init__(self, p_drop=0.0):
        super().__init__()
        d = lambda: nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()  # noqa: E731
        self.body = nn.Sequential(
            nn.Linear(784, 512), nn.LeakyReLU(0.2), d(),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), d(),
            nn.Linear(256, 128), nn.LeakyReLU(0.2), d(),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
        )
        self.head = nn.Linear(64, 1)

    def features(self, x):
        return self.body(x.view(x.size(0), -1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def binarize(x):
    return torch.bernoulli(x.clamp(0, 1))


def train_disc(X_real, X_fake, p_drop=0.0, seed=0):
    torch.manual_seed(seed)
    X = torch.cat([binarize(X_real), X_fake]).to(device)
    y = torch.cat([torch.ones(len(X_real)), torch.zeros(len(X_fake))]).to(device)
    d = Disc(p_drop).to(device)
    opt = torch.optim.Adam(d.parameters(), lr=1e-3)
    n = len(X)
    for ep in range(args.disc_epochs):
        d.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            loss = F.binary_cross_entropy_with_logits(d(X[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    d.eval()
    return d, X, y


class RFFLaplaceGP:
    """GP classifier approximation: random Fourier features on frozen discriminator features,
    Bayesian logistic regression with Laplace posterior. Predictive std comes from phi' Sigma phi."""

    def __init__(self, disc, X, y, D=1024, lengthscale=None, ridge=1.0):
        self.disc = disc
        with torch.no_grad():
            H = torch.cat([disc.features(X[i:i + 4096]) for i in range(0, len(X), 4096)])
        self.mu, self.sd = H.mean(0), H.std(0) + 1e-6
        H = (H - self.mu) / self.sd
        if lengthscale is None:
            sub = H[torch.randperm(len(H), device=H.device)[:1000]]
            lengthscale = torch.cdist(sub, sub).median().item()
        self.W = torch.randn(H.shape[1], D, device=H.device) / lengthscale
        self.b = torch.rand(D, device=H.device) * 2 * math.pi
        Phi = self._phi_from_h(H)
        beta = torch.zeros(D, device=H.device, requires_grad=True)
        opt = torch.optim.LBFGS([beta], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            l = F.binary_cross_entropy_with_logits(Phi @ beta, y, reduction="sum") + 0.5 * ridge * beta.pow(2).sum()
            l.backward()
            return l
        opt.step(closure)
        self.beta = beta.detach()
        with torch.no_grad():
            pr = torch.sigmoid(Phi @ self.beta)
            prec = ridge * torch.eye(D, device=H.device) + (Phi * (pr * (1 - pr)).unsqueeze(1)).T @ Phi
            self.Sigma = torch.linalg.inv(prec)

    def _phi_from_h(self, H):
        return math.sqrt(2.0 / self.W.shape[1]) * torch.cos(H @ self.W + self.b)

    @torch.no_grad()
    def predict(self, X):
        H = (self.disc.features(X) - self.mu) / self.sd
        Phi = self._phi_from_h(H)
        m = Phi @ self.beta
        var = ((Phi @ self.Sigma) * Phi).sum(1)
        return m, var.clamp_min(0).sqrt()


def fit_temperature(logits, labels):
    """Temperature scaling on held-out real vs fresh synthetic samples (Guo et al. 2017)."""
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        l = F.binary_cross_entropy_with_logits(logits / logT.exp(), labels)
        l.backward()
        return l
    opt.step(closure)
    return float(logT.detach().exp())


class Verifier:
    def calibrate(self, X_real_cal, X_fake_cal):
        self.T = 1.0
        if not args.temp_scale:
            return
        m_r, _ = self.score(X_real_cal)
        m_f, _ = self.score(X_fake_cal)
        self.T = fit_temperature(torch.cat([m_r, m_f]),
                                 torch.cat([torch.ones(len(m_r)), torch.zeros(len(m_f))]))

    def __init__(self, X_real, X_fake):
        self.T = 1.0
        if args.unc == "none":
            self.discs = [train_disc(X_real, X_fake)[0]]
        elif args.unc == "mcd":
            self.discs = [train_disc(X_real, X_fake, p_drop=0.2)[0]]
        elif args.unc == "ens":
            self.discs = [train_disc(X_real, X_fake, seed=s)[0] for s in range(args.ens_M)]
        elif args.unc == "gp":
            d, X, y = train_disc(X_real, X_fake)
            self.discs = [d]
            self.gp = RFFLaplaceGP(d, X, y, D=args.gp_D)

    @torch.no_grad()
    def score(self, X):
        """Returns mean logit m and its std s. logit = log D/(1-D) estimates log p_data/p_gen."""
        X = X.to(device)
        ms, ss = [], []
        for i in range(0, len(X), 4096):
            xb = X[i:i + 4096]
            if args.unc == "none":
                m, s = self.discs[0](xb), torch.zeros(len(xb), device=device)
            elif args.unc == "mcd":
                self.discs[0].train()
                L = torch.stack([self.discs[0](xb) for _ in range(args.mc_T)])
                self.discs[0].eval()
                m, s = L.mean(0), L.std(0)
            elif args.unc == "ens":
                L = torch.stack([d(xb) for d in self.discs])
                m, s = L.mean(0), L.std(0)
            else:
                m, s = self.gp.predict(xb)
            ms.append(m)
            ss.append(s)
        return torch.cat(ms).cpu() / self.T, torch.cat(ss).cpu() / self.T


# ----------------------------------------------------------------------------------------------
# Selection rules
# ----------------------------------------------------------------------------------------------
def ratio_weights(m, s):
    """Density-ratio weights exp(logit), shrunk by uncertainty and clipped."""
    logw = args.lam * (m - args.kappa * s)
    w = torch.exp(logw - logw.max())
    cap = torch.quantile(w, args.clip_q)
    return torch.minimum(w, cap).clamp_min(1e-12)


def weighted_wor(w, k):
    k = min(k, len(w))
    return torch.multinomial(w, k, replacement=False)


def select_block(m, s, k, cal_m, cal_s):
    """Select k indices from one block (all classes when pooled, one class otherwise)."""
    n = len(m)
    info = {}
    if args.method == "random":
        return torch.randperm(n)[:k], info
    if args.method == "topk":
        return torch.topk(m, min(k, n)).indices, info
    if args.method == "ratio":
        return weighted_wor(ratio_weights(m, s), k), info
    # conf_ratio: optimistic score for the gate so uncertain rare samples are not rejected
    ucb, cal_ucb = m + args.kappa * s, cal_m + args.kappa * cal_s
    tau = torch.quantile(cal_ucb, args.alpha)
    gate = torch.nonzero(ucb >= tau).squeeze(1)
    info["gate_rate"] = len(gate) / max(n, 1)
    if len(gate) <= k:
        return fill(gate, m, s, k), info
    return gate[weighted_wor(ratio_weights(m[gate], s[gate]), k)], info


def fill(gate, m, s, k):
    """If fewer than k samples pass the gate, keep all of them and top up the budget by ratio sampling
    among the rejected samples, so every method trains on the same number of samples.
    The shortfall is logged through gate_rate."""
    if len(gate) >= k:
        return gate
    mask = torch.ones(len(m), dtype=torch.bool)
    mask[gate] = False
    rest = torch.nonzero(mask).squeeze(1)
    extra = rest[weighted_wor(ratio_weights(m[rest], s[rest]), min(k - len(gate), len(rest)))]
    return torch.cat([gate, extra])


def select(Xp, Yp, m, s, k, verifier):
    cal_m, cal_s = verifier.score(binarize(X_cal))
    infos = {}
    if args.pooled:
        if args.method == "conf_ratio" and args.conf_by_class:
            # Mondrian gate per class, then a single pooled ratio sampling step
            ucb = m + args.kappa * s
            gate_mask = torch.zeros(len(m), dtype=torch.bool)
            for c in range(NC):
                cm = Y_cal == c
                tau = torch.quantile(cal_m[cm] + args.kappa * cal_s[cm], args.alpha)
                gate_mask |= (Yp == c) & (ucb >= tau)
            gate = torch.nonzero(gate_mask).squeeze(1)
            infos["gate_rate"] = len(gate) / len(m)
            sel = fill(gate, m, s, k) if len(gate) <= k else gate[weighted_wor(ratio_weights(m[gate], s[gate]), k)]
        else:
            sel, infos = select_block(m, s, k, cal_m, cal_s)
    else:
        sel = []
        for c in range(NC):
            ic = torch.nonzero(Yp == c).squeeze(1)
            cm = Y_cal == c if args.conf_by_class else torch.ones_like(Y_cal, dtype=torch.bool)
            sc, _ = select_block(m[ic], s[ic], k // NC, cal_m[cm], cal_s[cm])
            sel.append(ic[sc])
        sel = torch.cat(sel)
    return sel, infos


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3), nn.ReLU(),
                               nn.MaxPool2d(2), nn.Flatten(), nn.Linear(1600, 128), nn.ReLU())
        self.out = nn.Linear(128, NC)

    def forward(self, x):
        return self.out(self.f(x))


def get_feature_net():
    """Small MNIST classifier trained once on the full training set; its 128-d features are the
    embedding for precision/recall, as LeNet features are commonly used for MNIST evaluation."""
    path = THIS_DIR.parent / "result" / "lenet_features.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    net = LeNet().to(device)
    if path.exists():
        net.load_state_dict(torch.load(path, map_location=device))
    else:
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        Xd, Yd = X_full.to(device), Y_full.to(device)
        for ep in range(3):
            perm = torch.randperm(len(Xd), device=device)
            for i in range(0, len(Xd), 256):
                b = perm[i:i + 256]
                loss = F.cross_entropy(net(Xd[b]), Yd[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        torch.save(net.state_dict(), path)
    net.eval()
    return net


_feat = None
ROWS = []


@torch.no_grad()
def embed(X):
    out = [_feat.f(X[i:i + 2048].to(device)).cpu() for i in range(0, len(X), 2048)]
    return torch.cat(out)


def pr_recall(X_real, X_gen, k=3, n=2000):
    """Improved precision/recall (Kynkaanniemi et al. 2019) in LeNet feature space."""
    global _feat
    if _feat is None:
        _feat = get_feature_net()
    R = embed(X_real[torch.randperm(len(X_real))[:n]])
    G = embed(X_gen[torch.randperm(len(X_gen))[:n]])
    dRR, dGG, dRG = torch.cdist(R, R), torch.cdist(G, G), torch.cdist(R, G)
    rR = dRR.kthvalue(k + 1, dim=1).values
    rG = dGG.kthvalue(k + 1, dim=1).values
    precision = (dRG <= rR.unsqueeze(1)).any(0).float().mean().item()
    recall = (dRG.T <= rG.unsqueeze(1)).any(0).float().mean().item()
    return precision, recall


@torch.no_grad()
def evaluate(model, class_probs, round_id, extra):
    val_loss, val_recon, val_kl = train_helper.calculate_validation_loss(model, test_loader, device)
    Xg, Yg = generate(model, args.fid_n, class_probs, binary=False)
    Xg_soft = Xg.float()
    if args.no_fid:
        fid = float("nan")
    else:
        real_ds = TensorDataset(X_test[: args.fid_real], Y_test[: args.fid_real])
        fid = calculate_fid_score(real_ds, TensorDataset(Xg_soft, Yg), batch_size=50, device=device)
    prec, rec = pr_recall(X_test, Xg_soft)
    rec_c = []
    for c in range(NC):
        rc, gc = X_test[Y_test == c], Xg_soft[Yg == c]
        rec_c.append(pr_recall(rc, gc, n=min(500, len(rc), len(gc)))[1] if len(gc) > 10 else 0.0)
    tv = 0.5 * np.abs(np.asarray(class_probs) - 1.0 / NC).sum()
    row = dict(round=round_id, val_loss=val_loss, val_recon=val_recon, val_kl=val_kl, fid=fid,
               precision=prec, recall=rec, min_class_recall=min(rec_c), class_tv=tv,
               class_probs=json.dumps([round(float(x), 4) for x in class_probs]),
               class_recall=json.dumps([round(r, 3) for r in rec_c]), **extra)
    ROWS.append(row)
    pd.DataFrame(ROWS).to_csv(OUT / "results.csv", index=False)  # rewritten each round, columns stay aligned
    log(f"round {round_id}: ELBO-loss {val_loss:.2f}  FID {fid:.2f}  P {prec:.3f}  R {rec:.3f}  "
        f"minR {min(rec_c):.3f}  classTV {tv:.3f}  {extra}")


# ----------------------------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------------------------
log(f"config: {vars(args)}")
_feat = get_feature_net()
CKPT = OUT / "checkpoint.pt"


def new_cvae_shell():
    return models.CVAE(input_dim=784, label_dim=NC, latent_dim=20, name="resumed", arch="conv").to(device)


def save_checkpoint(r):
    """Saved after every finished round so an interrupted run continues from round r + 1."""
    torch.save(dict(round=r, gen=gen.state_dict(), class_probs=class_probs, rows=ROWS,
                    acc_X=acc_X, acc_Y=acc_Y, acc_W=acc_W,
                    rng=(torch.get_rng_state(), np.random.get_state(), random.getstate())), CKPT)


k_keep = int(args.pool * args.keep)
start = 1
if args.resume and CKPT.exists():
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    gen = new_cvae_shell()
    gen.load_state_dict(ck["gen"])
    gen.eval()
    class_probs, acc_X, acc_Y, acc_W = ck["class_probs"], ck["acc_X"], ck["acc_Y"], ck["acc_W"]
    ROWS.extend(ck["rows"])
    torch.set_rng_state(ck["rng"][0])
    np.random.set_state(ck["rng"][1])
    random.setstate(ck["rng"][2])
    start = ck["round"] + 1
    log(f"resumed from checkpoint after round {ck['round']}")
else:
    class_probs = np.bincount(Y_init.numpy(), minlength=NC) / len(Y_init)
    gen = train_cvae(X_init, Y_init, "round0")
    evaluate(gen, class_probs, 0, {})
    acc_X, acc_Y, acc_W = [X_init], [Y_init], [torch.ones(len(X_init))]
    save_checkpoint(0)

for r in range(start, args.rounds + 1):
    t0 = time.time()
    # verifier for the current generator, real reference = balanced held-out MNIST
    X_fake_disc, _ = generate(gen, len(X_disc_real), class_probs)
    verifier = Verifier(X_disc_real, X_fake_disc)
    X_fake_T, _ = generate(gen, len(X_T), class_probs)
    verifier.calibrate(binarize(X_T), X_fake_T)

    # candidate pool with injected artifacts
    Xp, Yp = generate(gen, args.pool, class_probs)
    Xp, Yp, is_art = make_artifacts(Xp, Yp, args.art_frac)
    m, s = verifier.score(Xp)
    sel, info = select(Xp, Yp, m, s, k_keep, verifier)

    X_sel, Y_sel = Xp[sel], Yp[sel]
    pool_cls = np.bincount(Yp.numpy(), minlength=NC) / len(Yp)
    sel_cls = np.bincount(Y_sel.numpy(), minlength=NC) / max(len(Y_sel), 1)
    extra = dict(n_selected=len(sel),
                 pool_class=json.dumps([round(float(x), 3) for x in pool_cls]),
                 selected_class=json.dumps([round(float(x), 3) for x in sel_cls]),
                 rare_share_pool=round(float(pool_cls[7:].sum()), 4),
                 rare_share_selected=round(float(sel_cls[7:].sum()), 4),
                 artifact_in_pool=float(is_art.float().mean()),
                 artifact_in_selected=float(is_art[sel].float().mean()),
                 mean_unc=float(s.mean()), temperature=round(verifier.T, 3),
                 **{k: round(v, 4) for k, v in info.items()})

    # training weight of each selected sample: ratio**train_lam, normalized to mean 1
    lw = args.train_lam * (m[sel] - args.kappa * s[sel])
    W_sel = torch.exp(lw - lw.max())
    W_sel = torch.minimum(W_sel, torch.quantile(W_sel, args.clip_q))
    W_sel = W_sel / W_sel.mean()
    if args.train_lam > 0:
        extra["train_rare_share"] = round(float((W_sel * (Y_sel >= 7).float()).sum() / W_sel.sum()), 4)

    if args.regime == "replace":
        X_train, Y_train, W_train = X_sel, Y_sel, W_sel
    else:
        acc_X.append(X_sel)
        acc_Y.append(Y_sel)
        acc_W.append(W_sel)
        X_train, Y_train, W_train = torch.cat(acc_X), torch.cat(acc_Y), torch.cat(acc_W)

    # the next generator samples labels with the (weighted) class frequencies of its training data
    class_probs = np.bincount(Y_train.numpy(), weights=W_train.numpy(), minlength=NC)
    class_probs = class_probs / class_probs.sum()
    gen = train_cvae(X_train, Y_train, f"round{r}", W_train if args.train_lam > 0 else None)
    extra["sec"] = round(time.time() - t0, 1)
    evaluate(gen, class_probs, r, extra)
    save_checkpoint(r)
    del verifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

CKPT.unlink(missing_ok=True)  # finished runs keep only results.csv and log.txt
log("done")
