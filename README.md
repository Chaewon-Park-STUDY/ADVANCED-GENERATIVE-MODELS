# Advanced Generative Models

고급생성모델 프로젝트

## 주제

생성 모델이 자기 생성 데이터로 반복 재학습될 때 발생하는 model collapse와, verifier 기반 합성 데이터 선별이 collapse를 완화하거나 가속하는 조건

## 선정 논문

| # | 논문 | 학회 | 링크 | 리뷰 |
| --- | --- | --- | --- | --- |
| 1 | Escaping Model Collapse via Synthetic Data Verification | ICLR 2026 | [arXiv](https://arxiv.org/abs/2510.16657) · [code](https://github.com/liuqiyuanhhh/Verified-Synthetic-Data) | [리뷰](papers/01_Escaping_Model_Collapse/README.md) |
| 2 | When Sample Selection Bias Precipitates Model Collapse | ICML 2026 | [arXiv](https://arxiv.org/abs/2606.13732) | [리뷰](papers/02_Sample_Selection_Bias/README.md) |

두 논문은 같은 질문에 대해 서로 대척점에 있는 결론을 내린다. 첫 번째 논문은 verifier 기반 선별이 collapse를 완화한다고 보고, 두 번째 논문은 편향된 verifier 기반 선별이 collapse를 가속한다고 본다.

## Proposal

**문제.** 합성 데이터로 반복 학습하면 model collapse가 생긴다. Paper 1은 verifier로 걸러도 verifier의 편향 수준에서 성능이 멈춤을, Paper 2는 real 데이터를 쌓아도 상위 점수만 고르면 분산이 줄어 collapse가 일어남을 보였다. Paper 2의 해결책은 전체 데이터 분포가 필요해 현실적으로 쓰기 어렵다.

**제안.** 상위 몇 %를 남기는 대신, 소량의 held-out real 데이터로 기준선 τ를 정해(real의 99%가 통과) 그 아래만 버리고, 나머지는 D/(1−D)에 비례해 뽑는다(conformal threshold + ratio sampling). 목표 지표는 verifier 점수가 아니라 실제 분포와의 거리이며(Vision은 FID, Text는 ROUGE-1과 다양성), calibration용 real 데이터가 test 분포를 대표한다고 가정한다.

**연구 질문.**
- RQ1: real로 보정한 선택 규칙이 분포를 자르지 않고 오류만 제거하는가, 드문 영역을 유지하는가?
- RQ2: 워터마크가 들어간 synthetic 데이터로 재학습하면 다양성이 줄어드는가, 워터마크 점수로 이를 되살리는 데이터를 고를 수 있는가?

**현재 결과.**
- RQ1 Vision(long-tail MNIST, CVAE, 10세대): 우리 방법만 ELBO loss가 유지되었다(126 vs top-k 135, FID 119.7 vs 127.5). 편향 verifier에서도 같았다. 다만 CPU 재실행에서 top-k와의 차이가 재현되지 않아 GPU로 확인 중이다. 드문 영역을 더 고르는 효과는 확인되지 않았다(rare lift ≈ 0).
- RQ1 Text(XSum, SmolLM2-135M, 5세대): verifier는 random보다 나았지만 우리 방법의 추가 이점은 없었다. discriminator 기반 선택은 세대가 지날수록 hallucination을 다시 통과시켰다.
- RQ2 워터마크: 워터마크가 강할수록 다양성이 줄었고(distinct-2 0.753 → 0.720), 학습된 모델에서 워터마크는 검출되지 않았다. 길이를 맞춘 low-watermark 선택으로 감소폭의 약 2/3를 회복했다(0.743). Accumulate에서는 효과가 작았다.

**다음 단계.** 드물지만 타당한 sample과 오류를 구분하는 신호, 텍스트 verifier의 내용 일치 확인, calibration과 test 분포 차이를 보정하는 weighted conformal, 세대 수와 seed 확대.

자세한 내용은 `proposal/experiments/README.md`(Vision)와 `proposal/experiments_text/README.md`(Text, 워터마크)에 있다.

## 구조

```
papers/
├── 01_Escaping_Model_Collapse/
│   ├── README.md          # 논문 리뷰, 재현 결과
│   └── code/              # 실험 스크립트, 결과 CSV, 그래프
└── 02_Sample_Selection_Bias/
    └── README.md          # 논문 리뷰
proposal/
├── design_draft.md        # 프로젝트 설계 초안
├── experiments/           # 실험1 Vision (RQ1)
└── experiments_text/      # 실험2 Text (RQ1), 워터마크 (RQ2)


```
