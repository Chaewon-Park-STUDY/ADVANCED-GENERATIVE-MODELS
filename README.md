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

## 구조

```
papers/
├── 01_Escaping_Model_Collapse/
│   ├── README.md          # 논문 리뷰, 재현 결과
│   └── code/              # 실험 스크립트, 결과 CSV, 그래프
└── 02_Sample_Selection_Bias/
    └── README.md          # 논문 리뷰
proposal/
└── design_draft.md        # 프로젝트 설계 초안


```
