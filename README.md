# Advanced Generative Models

고급생성모델 프로젝트

## 주제

생성 모델이 자기 생성 데이터로 반복 재학습될 때 발생하는 model collapse와, verifier 기반 합성 데이터 선별을 통한 완화 방안

## 선정 논문

| # | 논문 | 학회 | 링크 | 리뷰 |
| --- | --- | --- | --- | --- |
| 1 | Escaping Model Collapse via Synthetic Data Verification | ICLR 2026 | [arXiv](https://arxiv.org/abs/2510.16657) · [code](https://github.com/liuqiyuanhhh/Verified-Synthetic-Data) | [리뷰](papers/01_Escaping_Model_Collapse/README.md) |
| 2 | Great Models Think Alike and this Undermines AI Oversight | ICML 2025 Spotlight | [arXiv](https://arxiv.org/abs/2502.04313) · [code](https://github.com/model-similarity/lm-similarity) | [리뷰](papers/02_Great_Models_Think_Alike/README.md) |
| 3 | When Sample Selection Bias Precipitates Model Collapse | ICML 2026 | [ICML](https://icml.cc/virtual/2026/poster/65263) | |
| 4 | Machine-generated text detection prevents language model collapse | EMNLP 2025 | [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1506/) · [code](https://github.com/GeorgeDrayson/model_collapse) | |

## 구조

```
papers/
├── 01_Escaping_Model_Collapse/
│   ├── README.md          # 논문 리뷰, 재현 결과
│   └── code/              # 실험 스크립트, 결과 CSV, 그래프
└── 02_Great_Models_Think_Alike/
    ├── README.md          # 논문 리뷰, CAPA 예시
    └── code/              # CAPA 예시 코드, 실행 결과
```
