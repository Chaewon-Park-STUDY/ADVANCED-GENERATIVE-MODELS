# When Sample Selection Bias Precipitates Model Collapse

- 학회: ICML 2026
- 논문: https://arxiv.org/abs/2606.13732
- 원본 코드: 공개되지 않음

## 선정 이유

Escaping Model Collapse via Synthetic Data Verification은 verifier 기반 선별이 model collapse를 완화할 수 있음을 보였다. 그러나 verifier 자체가 편향되어 있다면 선별 과정이 오히려 collapse를 앞당길 수 있다. 본 논문은 이러한 반대 방향의 결론을 이론과 실험으로 제시하므로, 두 논문을 대비하면 verifier의 질에 따라 model collapse가 완화될 수도, 가속될 수도 있다는 점을 보일 수 있다. 또한 공정한 verifier를 탐구하는 연구 방향의 출발점이 된다는 점에서 선정하였다.

## 문제의식

합성 데이터 선별은 model collapse의 해결책으로 널리 여겨지지만, 그 신뢰성은 verifier가 사용하는 reference distribution에 크게 의존한다.

의료 컨소시엄이나 금융 기관처럼 raw 데이터를 한곳에 모을 수 없는 data silo 환경에서는 각 verifier가 전체 분포 중 작고 파편화된, 편향된 일부만을 관찰한다.

저자들은 이러한 low-resource verification 환경에서 선별 자체가 편향되며, 그 결과 분포의 꼬리에 해당하는 드문 샘플이 제거되어 collapse가 가속된다고 본다.

## 방법

### 이론 설정

- 데이터 분포를 multivariate Gaussian으로 두고, 선별을 목표점에 가까운 상위 일부 샘플만 남기는 truncated sampling으로 모델링한다.
- 재학습 방식으로 데이터를 교체하는 replace와 누적하는 accumulate를 모두 분석한다.

### 완화 방안

raw 데이터를 공유하지 않고 여러 silo의 정보를 결합해 proxy reference를 구성한다.

- Scheme I: 각 silo가 합성 데이터와 자신의 데이터 사이를 Wasserstein geodesic으로 보간하여 proxy reference를 만든다.
- Scheme II: 서버가 각 silo의 보간 결과를 Wasserstein barycenter로 결합하여 전체 분포를 추정한다.
- 두 방식 모두 Sinkhorn optimal transport의 dual potential로 샘플별 점수를 계산하고, 분포 간 차이를 키우는 샘플을 제거한다.

## 핵심 결과

- 편향된 선별 하에서 평균은 목표점으로 수렴하지만 variance는 0으로 수렴하여 한 점으로 collapse한다. 이 현상은 일반적으로 안정적이라고 알려진 accumulate 설정에서도 발생한다.
- variance는 power-law 속도로 감소하며, 이는 diversity의 power-law 감소로 이어진다.
- 실제 분포에서의 downstream risk는 선별된 분포와 실제 분포 사이의 Wasserstein distance로 상한이 주어진다. 이 결과가 Wasserstein 기반 완화 방안의 근거가 된다.

## 논문 실험

- DDPM을 CIFAR-10, STL-10, CelebA에 적용하였다. real 데이터를 10개 silo에 non-IID로 분할하고 10세대 동안 재학습하였다.

| 방법 | CIFAR-10 FID ↓ | STL-10 FID ↓ | CelebA FID ↓ |
| --- | --- | --- | --- |
| Random 선별 | 106 | 95 | 96 |
| CovMatch | 115 | 131 | 92 |
| Scheme II, barycenter | 85 | 69 | 75 |
| Scheme I, geodesic | 71 | 65 | 69 |

- 기존 선별 방법 일부는 무작위 선별보다도 나쁜 성능을 보였다. 이는 local reference에 기반한 선별이 skewed distribution에서 실패함을 보여준다.
- 제안한 proxy reference는 diversity 저하를 완화하였다. CIFAR-10에서 recall은 Random 선별의 0.48에서 Scheme I의 0.58로 향상되었다.

## 한계

- 핵심 정리가 Gaussian 분포와 local concavity 가정에 의존한다.
- 이미지 DDPM 실험만 수행하였으며 언어 모델 실험은 포함하지 않았다.
- 다수의 silo가 악의적인 경우 공유 proxy reference가 오염될 수 있다.
- 코드가 공개되지 않아 재현이 어렵다.

## Escaping Model Collapse와의 관계

| | Escaping Model Collapse | When Sample Selection Bias |
| --- | --- | --- |
| 결론 | verifier 기반 선별은 collapse를 완화한다 | 편향된 verifier 기반 선별은 collapse를 가속한다 |
| verifier 가정 | 전체 real 데이터에 접근 가능한 verifier | 일부 real 데이터만 관찰하는 verifier |
| 수렴 결과 | verifier의 knowledge center로 수렴 | variance가 0으로 수렴하며 diversity 소실 |
| 공통점 | 재학습의 결과는 verifier가 무엇을 보고 판단하는가에 의해 결정된다 | |

## 프로젝트 연결

- 여러 silo의 reference를 결합하는 본 논문의 완화 방안은, 여러 verifier를 결합하는 multi-verifier 접근과 같은 발상이다.
- 프로젝트 실험에서 각 verifier가 real 데이터의 일부만 관찰하도록 설정하여 본 논문의 상황을 재현하고, multi-verifier가 이 편향을 완화하는지 검증한다.
- 본 논문이 성능보다 diversity가 먼저 무너진다고 보고하므로, FID와 함께 recall 등 diversity 지표를 기록한다.
