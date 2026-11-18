# Project Design Draft

> 초안. 팀 회의에서 확정 전까지 변경될 수 있음.

## 1. 배경

- Escaping Model Collapse는 verifier 기반 선별이 collapse를 완화하며, 결과는 verifier의 knowledge center로 수렴한다고 본다.
- When Sample Selection Bias는 verifier가 real 데이터의 일부만 볼 때 선별이 편향되어 variance가 0으로 수렴하고 collapse가 가속된다고 본다.
- 두 논문의 차이는 verifier가 관찰하는 reference의 범위에서 비롯된다.

## 2. 문제 제기: 재현 결과에서 출발

- Escaping 코드의 선별은 discriminator 점수 상위 10%만 남기는 방식으로, Sample Selection Bias가 분석한 truncated sampling과 같은 연산이다.
- 재현 실험에서 verifier가 real 6만 장 전체를 관찰했음에도 FID는 개선되고 negative ELBO는 3회 이후 악화되었다.
- 가설: reference가 넓더라도 상위만 자르는 선별은 sample quality를 높이는 대신 coverage를 떨어뜨린다.

## 3. 연구 질문

1. 상위 일부만 자르는 선별은 reference가 넓어도 coverage를 떨어뜨리는가?
2. real 데이터의 일부만 보는 verifier들을 결합하면 전체를 보는 verifier에 가까운 결과를 얻는가?
3. 자르지 않는 선별 방식이 sample quality와 coverage를 함께 유지할 수 있는가?

## 4. Harness 구조

```
Generator ─▶ Verifier(s) ─▶ Aggregator ─▶ Selector ─▶ Retrain ─▶ 다음 세대
                                                         │
                                  Monitor: 세대별 quality, coverage 기록
```

| 모듈 | 역할 | 비교할 선택지 |
| --- | --- | --- |
| Verifier | 샘플별 점수 산출 | full-reference discriminator / silo discriminator |
| Aggregator | 여러 verifier 점수 결합 | 단일 / 평균 / 최소값 |
| Selector | 학습에 쓸 샘플 선택 | top-k 절단 / 점수 비례 확률 샘플링 / 숫자별 비율 유지 |
| Monitor | 세대별 지표 기록 | 아래 5절 |

## 5. 평가 지표

| 구분 | 지표 |
| --- | --- |
| sample quality | FID |
| coverage | negative ELBO, recall, 숫자별 생성 비율 |
| verifier 상태 | accept rate, verifier 간 판정 불일치율 |

## 6. 실험 계획

공통: MNIST conditional VAE, real 500장 시작, 세대당 합성 2만 장, 8세대

| 실험 | 비교 | 연구 질문 |
| --- | --- | --- |
| E1 | 필터 없음 / full-reference + top-k | 1, 재현 기준선 |
| E2 | full-reference에서 top-k / 확률 샘플링 / 숫자별 비율 유지 | 1, 3 |
| E3 | silo verifier 단독 / silo verifier 여러 개 결합 / full-reference | 2 |
| E4 | silo 결합 + 확률 샘플링 | 2, 3 |

- silo 설정: real 6만 장을 숫자 분포가 치우치도록 non-IID로 나누고, 각 verifier는 하나의 silo만 학습한다.
- 예상 실행 수 약 9회, 기존 속도 기준 약 16시간 GPU.

## 7. 일정

| 주차 | 할 일 |
| --- | --- |
| 1 | harness 뼈대, E1로 기존 결과 재현 확인 |
| 2 | Selector 3종 구현, E2 |
| 3 | silo verifier와 aggregator 구현, E3 |
| 4 | E4, 결과 정리 |

## 8. 미정 사항

- 실험 도메인: vision 유지 여부, text 확장 여부
- silo 개수와 편향 정도
- 역할 분담
- 원문 확인: Sample Selection Bias 실험 수치, Escaping 논문의 ELBO 역전 시점
