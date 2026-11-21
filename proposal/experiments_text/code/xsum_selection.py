"""
Selection-rule experiment on XSum summarization with SmolLM2-135M, following the text setting of
Escaping Model Collapse (Sections 5.3) at a reduced scale so that one run takes about an hour.

    py xsum_selection.py --method conf_ratio --unc mcd --regime accumulate

Each round
  1. the current model writes one summary for every article in a fixed pool,
  2. a fraction of summaries is replaced by a summary of another article (fluent but unfaithful,
     a proxy for hallucination),
  3. a selection rule keeps `keep` of the pool,
  4. a fresh copy of the base model is fine-tuned on the selected data (replace) or on the initial
     real data plus every selected batch so far (accumulate).

Selection rules (--method)
  random      : uniform subsample
  rouge_top   : top-k by ROUGE-1 against the reference summary (oracle verifier of the paper)
  disc_top    : top-k by a learned discriminator that sees only article + summary, no reference
  ratio       : sampling proportional to D/(1-D) of that discriminator (density-ratio calibration)
  conf_ratio  : conformal gate on held-out real summaries, then ratio sampling inside the gate

The discriminator is cross-fitted: the pool is split in two folds, a classifier trained on one fold
(real reference summary vs generated summary of the same article) scores the other fold.
"""
import os
import re
import sys
import json
import math
import time
import random
import argparse
import gc
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--method", default="conf_ratio", choices=["random", "rouge_top", "disc_top", "ratio", "conf_ratio"])
p.add_argument("--unc", default="none", choices=["none", "mcd"])
p.add_argument("--regime", default="replace", choices=["replace", "accumulate"])
p.add_argument("--rounds", type=int, default=5)
p.add_argument("--n_init", type=int, default=2000, help="real article-summary pairs for round 0")
p.add_argument("--n_pool", type=int, default=3000, help="articles summarized every round")
p.add_argument("--keep", type=float, default=0.25)
p.add_argument("--hallu_frac", type=float, default=0.05)
p.add_argument("--alpha", type=float, default=0.05)
p.add_argument("--kappa", type=float, default=1.0)
p.add_argument("--clip_q", type=float, default=0.99)
p.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
p.add_argument("--disc_model", default="distilroberta-base")
p.add_argument("--article_tokens", type=int, default=384)
p.add_argument("--max_new_tokens", type=int, default=48)
p.add_argument("--temperature", type=float, default=1.0)
p.add_argument("--top_p", type=float, default=1.0)
p.add_argument("--train_bs", type=int, default=16)
p.add_argument("--grad_accum", type=int, default=1, help="micro-batches per optimizer step; train_bs is split into this many")
p.add_argument("--gen_bs", type=int, default=64)
p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--disc_epochs", type=int, default=2)
p.add_argument("--disc_bs", type=int, default=32)
p.add_argument("--disc_len", type=int, default=320)
p.add_argument("--mc_T", type=int, default=10)
p.add_argument("--n_test", type=int, default=300)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="text_runs")
p.add_argument("--tag", default=None)
p.add_argument("--local_test", action="store_true", help="tiny random models and fake data, for checking the code")
args = p.parse_args()

tag = args.tag or f"{args.method}_{args.unc}_{args.regime}"
OUT = Path(args.out) / tag
OUT.mkdir(parents=True, exist_ok=True)
json.dump(vars(args), open(OUT / "config.json", "w"), indent=2)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT / "log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,  # noqa: E402
                          get_linear_schedule_with_warmup)
from rouge_score import rouge_scorer  # noqa: E402

ROUGE = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------


def load_xsum():
    if args.local_test:
        words = "the a government said on new year people police minister club company report city".split()
        def doc(i):
            rng = random.Random(i)
            return " ".join(rng.choice(words) for _ in range(rng.randint(40, 80)))
        tr = [{"document": doc(i), "summary": doc(i + 10 ** 6)[:60]} for i in range(args.n_init + args.n_pool + 50)]
        va = [{"document": doc(-i), "summary": doc(-i - 10 ** 6)[:60]} for i in range(1, args.n_test + 1)]
        return tr, va
    from datasets import load_dataset
    for name in ["EdinburghNLP/xsum", "xsum"]:
        try:
            ds = load_dataset(name)
            return list(ds["train"]), list(ds["validation"])
        except Exception as e:  # newer datasets versions refuse script-based repos
            log(f"load_dataset({name}) failed: {type(e).__name__}: {str(e)[:120]}")
    base = "hf://datasets/EdinburghNLP/xsum@refs/convert/parquet/default"
    ds = load_dataset("parquet", data_files={"train": f"{base}/train/*.parquet",
                                             "validation": f"{base}/validation/*.parquet"})
    return list(ds["train"]), list(ds["validation"])


train_raw, val_raw = load_xsum()
rng = random.Random(args.seed)
rng.shuffle(train_raw)
init = train_raw[: args.n_init]
pool = train_raw[args.n_init: args.n_init + args.n_pool]
test = val_raw[: args.n_test]
log(f"init {len(init)}  pool {len(pool)}  test {len(test)}")

# ----------------------------------------------------------------------------------------------
# Tokenizer / generator
# ----------------------------------------------------------------------------------------------


def build_tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    texts = [r["document"] + " " + r["summary"] for r in train_raw[:500]] + ["Article: Summary:"]
    tok.train_from_iterator(texts, trainers.WordLevelTrainer(special_tokens=["<unk>", "<pad>", "<eos>", "<s>", "</s>"]))
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>",
                                   bos_token="<s>", cls_token="<s>", sep_token="</s>")


if args.local_test:
    TOK = build_tiny_tokenizer()
    DTOK = TOK
else:
    TOK = AutoTokenizer.from_pretrained(args.model)
    DTOK = AutoTokenizer.from_pretrained(args.disc_model)
if TOK.pad_token is None:
    TOK.pad_token = TOK.eos_token


def new_generator():
    if args.local_test:
        from transformers import GPT2Config, GPT2LMHeadModel
        cfg = GPT2Config(vocab_size=len(TOK), n_positions=1024, n_embd=64, n_layer=2, n_head=2,
                         pad_token_id=TOK.pad_token_id, eos_token_id=TOK.eos_token_id, bos_token_id=TOK.bos_token_id)
        return GPT2LMHeadModel(cfg).to(device)
    return AutoModelForCausalLM.from_pretrained(args.model).to(device)


def prompt_ids(article):
    a = TOK(article, add_special_tokens=False, truncation=True, max_length=args.article_tokens)["input_ids"]
    return TOK("Article: ", add_special_tokens=False)["input_ids"] + a + TOK("\n\nSummary:", add_special_tokens=False)["input_ids"]


def encode_pair(article, summary):
    pi = prompt_ids(article)
    si = TOK(" " + summary.strip(), add_special_tokens=False, truncation=True, max_length=96)["input_ids"] + [TOK.eos_token_id]
    return pi + si, [-100] * len(pi) + si  # loss only on summary tokens


def fine_tune(pairs):
    model = new_generator()
    model.train()
    data = [encode_pair(a, s) for a, s in pairs]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(data) / args.train_bs) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    mb = math.ceil(args.train_bs / args.grad_accum)
    for ep in range(args.epochs):
        random.shuffle(data)
        for i in range(0, len(data), args.train_bs):
            batch = data[i:i + args.train_bs]
            n_tok = sum(sum(1 for t in y if t != -100) for _, y in batch)
            opt.zero_grad()
            for k0 in range(0, len(batch), mb):  # micro-batches; the loss is summed over summary tokens
                sub = batch[k0:k0 + mb]
                L = max(len(x) for x, _ in sub)
                ids = torch.full((len(sub), L), TOK.pad_token_id)
                lab = torch.full((len(sub), L), -100)
                att = torch.zeros((len(sub), L), dtype=torch.long)
                for j, (x, y) in enumerate(sub):
                    ids[j, :len(x)], lab[j, :len(y)], att[j, :len(x)] = torch.tensor(x), torch.tensor(y), 1
                sub_tok = sum(sum(1 for t in y if t != -100) for _, y in sub)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                    loss = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device)).loss
                (loss * sub_tok / n_tok).backward()
                del loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
    model.eval()
    return model


@torch.no_grad()
def summarize(model, articles, temperature, top_p):
    outs = []
    for i in range(0, len(articles), args.gen_bs):
        prompts = [prompt_ids(a) for a in articles[i:i + args.gen_bs]]
        L = max(len(x) for x in prompts)
        ids = torch.full((len(prompts), L), TOK.pad_token_id)
        att = torch.zeros((len(prompts), L), dtype=torch.long)
        for j, x in enumerate(prompts):  # left padding
            ids[j, L - len(x):], att[j, L - len(x):] = torch.tensor(x), 1
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            gen = model.generate(input_ids=ids.to(device), attention_mask=att.to(device), do_sample=True,
                                 temperature=temperature, top_p=top_p, max_new_tokens=args.max_new_tokens,
                                 pad_token_id=TOK.pad_token_id, eos_token_id=TOK.eos_token_id)
        for row in gen[:, L:]:
            text = TOK.decode(row, skip_special_tokens=True).strip().split("\n")[0].strip()
            outs.append(text if text else ".")
    return outs


# ----------------------------------------------------------------------------------------------
# Discriminator (cross-fitted), temperature scaling, MC dropout
# ----------------------------------------------------------------------------------------------


def new_disc():
    if args.local_test:
        from transformers import RobertaConfig, RobertaForSequenceClassification
        cfg = RobertaConfig(vocab_size=len(DTOK), hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                            intermediate_size=128, max_position_embeddings=600, num_labels=1,
                            pad_token_id=DTOK.pad_token_id)
        return RobertaForSequenceClassification(cfg).to(device)
    return AutoModelForSequenceClassification.from_pretrained(args.disc_model, num_labels=1).to(device)


def disc_batch(summaries, articles):
    enc = DTOK(summaries, articles, truncation="only_second", max_length=args.disc_len, padding=True, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}


def train_disc(real_s, fake_s, arts):
    model = new_disc()
    X = [(s, a, 1.0) for s, a in zip(real_s, arts)] + [(s, a, 0.0) for s, a in zip(fake_s, arts)]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    steps = math.ceil(len(X) / args.disc_bs) * args.disc_epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    model.train()
    for ep in range(args.disc_epochs):
        random.shuffle(X)
        for i in range(0, len(X), args.disc_bs):
            b = X[i:i + args.disc_bs]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logit = model(**disc_batch([x[0] for x in b], [x[1] for x in b])).logits.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit.float(), torch.tensor([x[2] for x in b], device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
    model.eval()
    return model


@torch.no_grad()
def disc_score(model, summaries, articles):
    ms, ss = [], []
    for i in range(0, len(summaries), 64):
        b = disc_batch(summaries[i:i + 64], articles[i:i + 64])
        if args.unc == "mcd":
            model.train()  # dropout on
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                L = torch.stack([model(**b).logits.squeeze(-1).float() for _ in range(args.mc_T)])
            model.eval()
            ms.append(L.mean(0).cpu())
            ss.append(L.std(0).cpu())
        else:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                m = model(**b).logits.squeeze(-1).float().cpu()
            ms.append(m)
            ss.append(torch.zeros_like(m))
    return torch.cat(ms), torch.cat(ss)


def fit_temperature(logits, labels):
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        l = F.binary_cross_entropy_with_logits(logits / logT.exp(), labels)
        l.backward()
        return l
    opt.step(closure)
    return float(logT.detach().exp())


def crossfit_scores(gen_s, arts, refs):
    """Returns per-candidate (m, s) and real calibration scores, all temperature-scaled."""
    n = len(arts)
    idx = list(range(n))
    random.Random(0).shuffle(idx)
    folds = [idx[: n // 2], idx[n // 2:]]
    m, s = torch.zeros(n), torch.zeros(n)
    cal_m, cal_s, temps = [], [], []
    for a, b in [(0, 1), (1, 0)]:
        tr, te = folds[a], folds[b]
        d = train_disc([refs[i] for i in tr], [gen_s[i] for i in tr], [arts[i] for i in tr])
        fm, fs = disc_score(d, [gen_s[i] for i in te], [arts[i] for i in te])
        rm, rs = disc_score(d, [refs[i] for i in te], [arts[i] for i in te])
        half = len(te) // 2  # first half of the held-out fold fits T, second half calibrates the gate
        T = fit_temperature(torch.cat([rm[:half], fm[:half]]),
                            torch.cat([torch.ones(half), torch.zeros(half)]))
        temps.append(T)
        m[te], s[te] = fm / T, fs / T
        cal_m.append(rm[half:] / T)
        cal_s.append(rs[half:] / T)
        del d
        torch.cuda.empty_cache() if device.type == "cuda" else None
    return m, s, torch.cat(cal_m), torch.cat(cal_s), float(np.mean(temps))


# ----------------------------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------------------------


def ratio_weights(m, s):
    logw = m - args.kappa * s
    w = torch.exp(logw - logw.max())
    return torch.minimum(w, torch.quantile(w, args.clip_q)).clamp_min(1e-12)


def wor(w, k):
    return torch.multinomial(w, min(k, len(w)), replacement=False)


def select(gen_s, arts, refs, k):
    info = {}
    n = len(gen_s)
    if args.method == "random":
        return torch.randperm(n)[:k], info
    if args.method == "rouge_top":
        r = torch.tensor([ROUGE.score(refs[i], gen_s[i])["rouge1"].fmeasure for i in range(n)])
        return torch.topk(r, k).indices, info
    m, s, cal_m, cal_s, T = crossfit_scores(gen_s, arts, refs)
    info.update(temperature=round(T, 3), mean_unc=round(float(s.mean()), 4))
    if args.method == "disc_top":
        return torch.topk(m, k).indices, info
    if args.method == "ratio":
        return wor(ratio_weights(m, s), k), info
    ucb = m + args.kappa * s
    tau = torch.quantile(cal_m + args.kappa * cal_s, args.alpha)
    gate = torch.nonzero(ucb >= tau).squeeze(1)
    info["gate_rate"] = round(len(gate) / n, 4)
    if len(gate) >= k:
        return gate[wor(ratio_weights(m[gate], s[gate]), k)], info
    rest_mask = torch.ones(n, dtype=torch.bool)
    rest_mask[gate] = False
    rest = torch.nonzero(rest_mask).squeeze(1)
    extra = rest[wor(ratio_weights(m[rest], s[rest]), k - len(gate))]
    return torch.cat([gate, extra]), info


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------


def toks(t):
    return re.findall(r"\w+|[^\w\s]", t.lower())


def distinct_n(texts, n):
    grams = [tuple(t[i:i + n]) for t in map(toks, texts) for i in range(len(t) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def bleu2(hyp, refs_tok):
    h = toks(hyp)
    if len(h) < 2:
        return 0.0
    score = 0.0
    for n in (1, 2):
        hg = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
        maxref = Counter()
        for r in refs_tok:
            maxref |= Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        clip = sum(min(c, maxref[g]) for g, c in hg.items())
        score += math.log(max(clip, 1e-9) / max(sum(hg.values()), 1)) / 2
    return math.exp(score)


def self_bleu(texts, n=200, n_ref=50):
    r = random.Random(0)
    sample = r.sample(texts, min(n, len(texts)))
    tok_all = [toks(t) for t in texts]
    vals = []
    for i, t in enumerate(sample):
        refs_tok = r.sample(tok_all, min(n_ref, len(tok_all)))
        vals.append(bleu2(t, [x for x in refs_tok if x != toks(t)]))
    return float(np.mean(vals))


def unigram_entropy(texts):
    c = Counter(w for t in texts for w in toks(t))
    tot = sum(c.values())
    return float(-sum(v / tot * math.log(v / tot) for v in c.values()))


def text_stats(texts):
    lens = [len(toks(t)) for t in texts]
    return dict(distinct1=round(distinct_n(texts, 1), 4), distinct2=round(distinct_n(texts, 2), 4),
                self_bleu=round(self_bleu(texts), 4), entropy=round(unigram_entropy(texts), 4),
                len_mean=round(float(np.mean(lens)), 2), len_std=round(float(np.std(lens)), 2))


ROWS = []


def evaluate(model, round_id, extra):
    outs = summarize(model, [r["document"] for r in test], temperature=0.7, top_p=0.9)
    r1 = float(np.mean([ROUGE.score(r["summary"], o)["rouge1"].fmeasure for r, o in zip(test, outs)]))
    row = dict(round=round_id, rouge1=round(r1, 4), **text_stats(outs), **extra)
    ROWS.append(row)
    pd.DataFrame(ROWS).to_csv(OUT / "results.csv", index=False)
    with open(OUT / f"samples_round{round_id}.jsonl", "w", encoding="utf-8") as f:
        for r, o in list(zip(test, outs))[:50]:
            f.write(json.dumps({"reference": r["summary"], "generated": o}, ensure_ascii=False) + "\n")
    log(f"round {round_id}: {row}")


# ----------------------------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------------------------
log(f"config: {vars(args)}")
ref_stats = text_stats([r["summary"] for r in test])
json.dump(ref_stats, open(OUT / "reference_stats.json", "w"), indent=2)
log(f"reference summary stats: {ref_stats}")

init_pairs = [(r["document"], r["summary"]) for r in init]
pool_arts = [r["document"] for r in pool]
pool_refs = [r["summary"] for r in pool]
k = int(args.keep * len(pool))

# Resume: a checkpoint is written after every round and removed when the run finishes.
CKPT = OUT / "checkpoint.pt"


def save_ckpt(rd, model, acc):
    state = {"round": rd, "model": {n: t.detach().cpu() for n, t in model.state_dict().items()},
             "acc": acc, "rows": ROWS, "py": random.getstate(), "np": np.random.get_state(),
             "torch": torch.get_rng_state(),
             "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
    tmp = OUT / "checkpoint.tmp"
    torch.save(state, tmp)
    os.replace(tmp, CKPT)


if CKPT.exists():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)  # RNG state must stay on CPU
    model = new_generator()
    model.load_state_dict(ck["model"])
    model.eval()
    acc = ck["acc"]
    ROWS[:] = ck["rows"]
    pd.DataFrame(ROWS).to_csv(OUT / "results.csv", index=False)
    random.setstate(ck["py"])
    np.random.set_state(ck["np"])
    torch.set_rng_state(ck["torch"])
    if ck["cuda"] is not None and device.type == "cuda":
        torch.cuda.set_rng_state_all(ck["cuda"])
    start = ck["round"] + 1
    del ck
    log(f"resumed after round {start - 1}")
else:
    model = fine_tune(init_pairs)
    evaluate(model, 0, {})
    acc = list(init_pairs)
    save_ckpt(0, model, acc)
    start = 1

for rd in range(start, args.rounds + 1):
    t0 = time.time()
    gen_s = summarize(model, pool_arts, args.temperature, args.top_p)
    # hallucination proxy: replace some summaries by the generated summary of another article
    n_h = int(args.hallu_frac * len(gen_s))
    h_idx = random.sample(range(len(gen_s)), n_h)
    donors = random.sample(range(len(gen_s)), n_h)
    is_h = torch.zeros(len(gen_s), dtype=torch.bool)
    for i, j in zip(h_idx, donors):
        if i != j:
            gen_s[i] = gen_s[j]
            is_h[i] = True
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    sel, info = select(gen_s, pool_arts, pool_refs, k)
    sel_pairs = [(pool_arts[i], gen_s[i]) for i in sel.tolist()]
    sel_r1 = float(np.mean([ROUGE.score(pool_refs[i], gen_s[i])["rouge1"].fmeasure for i in sel.tolist()]))
    extra = dict(n_selected=len(sel), hallu_in_pool=round(float(is_h.float().mean()), 4),
                 hallu_in_selected=round(float(is_h[sel].float().mean()), 4), selected_rouge1=round(sel_r1, 4),
                 **{f"sel_{k_}": v for k_, v in text_stats([g for _, g in sel_pairs]).items()}, **info)

    if args.regime == "replace":
        train_pairs = sel_pairs
    else:
        acc += sel_pairs
        train_pairs = acc
    gc.collect()
    torch.cuda.empty_cache() if device.type == "cuda" else None
    model = fine_tune(train_pairs)
    extra["sec"] = round(time.time() - t0, 1)
    evaluate(model, rd, extra)
    save_ckpt(rd, model, acc)

CKPT.unlink(missing_ok=True)
log("done")
