"""
lm-sim CAPA toy 실험
- 두 모델(정확도 80%, 2,000문항)이 어떻게 틀리느냐에 따라 CAPA와 error consistency(EC)가 어떻게 달라지는지 확인
- 실행: pip install lm-sim numpy  →  python capa_demo.py
"""
import numpy as np
from lmsim.metrics import CAPA, EC

rng = np.random.default_rng(0)
N = 2000          # 문항 수
ACC = 0.8         # 두 모델의 정확도


def one_hot_model(gt, correct_mask, wrong_choice, k):
    """correct_mask가 True인 문항은 정답, False인 문항은 wrong_choice를 고르는 모델 (one-hot 출력)"""
    out = []
    for g, c, w in zip(gt, correct_mask, wrong_choice):
        v = np.zeros(k)
        v[g if c else w] = 1.0
        out.append(v)
    return out


def random_wrong(gt, k):
    """각 문항에서 정답이 아닌 선택지 하나를 무작위로 고름"""
    return np.array([rng.choice([j for j in range(k) if j != g]) for g in gt])


def run(k, title):
    gt = rng.integers(0, k, N)
    print(f"--- {title} (선택지 {k}개)")

    # 1) 서로 독립적으로 틀림
    a = one_hot_model(gt, rng.random(N) < ACC, random_wrong(gt, k), k)
    b = one_hot_model(gt, rng.random(N) < ACC, random_wrong(gt, k), k)
    print(f"독립적으로 틀림       CAPA {CAPA(prob=False).compute_k(a, b, gt):6.3f} | EC {EC().compute_k(a, b, gt):6.3f}")

    # 2) 같은 문항에서 틀리지만 고른 오답이 다름
    mask = rng.random(N) < ACC
    a = one_hot_model(gt, mask, random_wrong(gt, k), k)
    b = one_hot_model(gt, mask, random_wrong(gt, k), k)
    print(f"같은 문항, 다른 오답  CAPA {CAPA(prob=False).compute_k(a, b, gt):6.3f} | EC {EC().compute_k(a, b, gt):6.3f}")

    # 3) 같은 문항에서 같은 오답
    w = random_wrong(gt, k)
    a = one_hot_model(gt, mask, w, k)
    b = one_hot_model(gt, mask, w, k)
    print(f"같은 문항, 같은 오답  CAPA {CAPA(prob=False).compute_k(a, b, gt):6.3f} | EC {EC().compute_k(a, b, gt):6.3f}")


def run_confidence():
    """통과/탈락(2지선다)에 확신도를 함께 줄 때 CAPA(확률 버전)가 어떻게 달라지는지"""
    gt = rng.integers(0, 2, N)
    mask = rng.random(N) < ACC          # 두 모델이 같은 문항에서 틀림

    def soft(conf):
        out = []
        for g, c in zip(gt, mask):
            p = np.zeros(2)
            pick = g if c else 1 - g
            p[pick], p[1 - pick] = conf, 1 - conf
            out.append(p)
        return out

    print("--- 통과/탈락 + 확신도 (같은 문항에서 틀림)")
    for conf in [0.99, 0.8, 0.6]:
        print(f"확신도 {conf}: CAPA(prob) {CAPA().compute_k(soft(conf), soft(conf), gt):.3f}")


if __name__ == "__main__":
    run(4, "객관식 4지선다")
    run(2, "verifier 통과/탈락")
    run_confidence()
