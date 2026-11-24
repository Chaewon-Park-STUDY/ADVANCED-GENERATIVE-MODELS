# Proposal 예비 실험 2: XSum 요약에서의 synthetic data selection

## 가설

실험1에서 MNIST로 확인한 conformal gate의 효과가 텍스트 요약에서도 나타나는지 본다. 텍스트에서는 verifier가 걸러야 할 대상이 hallucination으로 분명하다. verifier가 이를 세대가 지나도 계속 걸러내는지, 그리고 verifier가 고른 데이터의 특성이 모델 출력에 나타나는지를 함께 본다.

## 검증 방법

Escaping Model Collapse의 XSum 실험과 같은 SmolLM2-135M-Instruct를 쓴다. 0세대는 real 기사-요약 2,000쌍으로 학습한다. 매 세대 기사 3,000개를 요약하고 그중 25%인 750개를 골라 다시 학습하며, 5세대를 돌린다. 후보의 5%는 다른 기사의 요약으로 바꿔치기해 hallucination을 인위적으로 넣는다.

verifier는 두 종류이다. 하나는 정답 요약과의 ROUGE-1이고, 다른 하나는 (요약, 기사) 쌍이 real인지 생성인지 가르는 discriminator(distilroberta)이다. discriminator는 후보를 둘로 나눠 서로 다른 절반으로 학습하고 점수를 매긴다. 비교한 방법은 random, ROUGE 상위, discriminator 상위, ratio, conformal gate만, conformal gate + MC dropout이다. 매 세대 최신 synthetic 데이터만 쓰는 replace와 real과 이전 데이터를 쌓는 accumulate를 나눠 본다. 성능은 validation 300개의 ROUGE-1로, 다양성은 distinct-2와 self-BLEU로 본다.

Escaping Model Collapse의 XSum 실험과 데이터셋, 모델은 같지만 규모를 줄이고 verifier를 바꿨다.

| 항목 | 우리 실험 | 저자 실험 | 근거 |
| --- | --- | --- | --- |
| 0세대 학습 데이터 | real 2,000쌍 | XSum train의 12.5% (약 2.55만 개) | 논문 Section 5.3, 공식 코드 |
| 세대마다 요약 생성 | 기사 3,000개 | train 전체 기사 | 논문 Section 5.3 |
| 선택 비율 | 25% (750개) | 상위 12.5% (공식 코드는 세대 t마다 12.5% × (1 + t/10)로 증가) | 논문 Section 5.3, 공식 코드 |
| 세대 수 | 5 | 15 | 논문 Figure 5 |
| 평가 | validation 300개 | test ROUGE-1 (공식 코드는 validation 전체 11,332개) | 논문 Section 5.3, 공식 코드 |
| verifier | ROUGE, discriminator, ratio, conformal gate 등 | 정답과의 ROUGE-1 (oracle) vs random | 논문 Section 5.3 |
| hallucination 주입 | 후보의 5% | 없음 | |

seed 3개로 반복했고, 표의 값은 평균 ± 표준편차이다. 코드는 `code/`, 결과 원본은 `results_3seeds/`에 있다. seed 0만 돌린 첫 결과는 `results_seed0/`에 남겨 둔다.

## 결과

5세대 시점 값이다. 0세대 ROUGE-1은 0.234~0.243이고, 후보 전체의 hallucination 비율은 0.05이다.

| 방식 | 방법 | ROUGE-1 ↑ | distinct-2 ↑ | self-BLEU ↓ | 선택된 데이터의 hallucination |
| --- | --- | --- | --- | --- | --- |
| replace | random | 0.209 ± 0.004 | 0.755 ± 0.004 | 0.290 ± 0.007 | 0.064 ± 0.002 |
| replace | ROUGE 상위 | 0.220 ± 0.006 | 0.766 ± 0.004 | 0.275 ± 0.007 | 0.004 ± 0.001 |
| replace | discriminator 상위 | 0.222 ± 0.007 | 0.770 ± 0.002 | 0.268 ± 0.003 | 0.052 ± 0.008 |
| replace | ratio | 0.216 ± 0.006 | 0.765 ± 0.000 | 0.273 ± 0.005 | 0.047 ± 0.004 |
| replace | conformal gate | 0.223 ± 0.010 | 0.771 ± 0.008 | 0.276 ± 0.018 | 0.044 ± 0.006 |
| replace | conformal gate + MC dropout | 0.219 ± 0.005 | 0.767 ± 0.002 | 0.275 ± 0.003 | 0.051 ± 0.004 |
| accumulate | discriminator 상위 | 0.252 ± 0.002 | 0.716 ± 0.004 | 0.343 ± 0.004 | 0.017 ± 0.017 |
| accumulate | ratio | 0.246 ± 0.001 | 0.709 ± 0.005 | 0.350 ± 0.007 | 0.039 ± 0.008 |
| accumulate | conformal gate + MC dropout | 0.248 ± 0.004 | 0.706 ± 0.005 | 0.341 ± 0.007 | 0.043 ± 0.005 |

![replace](results_3seeds/plot_replace.png)

![accumulate](results_3seeds/plot_accumulate.png)

replace에서는 모든 방법의 ROUGE-1이 1세대에 0.219~0.227로 떨어졌다. 5세대에는 verifier를 쓴 방법이 0.216~0.223, random이 0.209로, 다섯 verifier의 평균이 모두 random보다 0.007~0.014 높았다. 다만 seed 2에서는 ratio와 conformal gate가 random보다 낮아, seed 3개로는 차이가 확정적이지 않다. 1세대의 하락에는 학습 데이터가 real 2,000쌍에서 synthetic 750쌍으로 바뀐 효과가 섞여 있을 수 있다.

replace의 다양성 지표는 0세대 대비 5세대에 오히려 좋아졌다. 6개 방법 평균으로 distinct-2가 0.72에서 0.77로 오르고 self-BLEU가 0.33에서 0.28로 내려갔다. 5세대 안에서는 다양성 감소 없이 ROUGE-1만 저하되었다.

accumulate에서는 ROUGE-1이 0.246~0.252로 0세대보다 올랐지만, self-BLEU가 0.33에서 0.34~0.35로 오르고 distinct-2가 0.72에서 0.71~0.72로 내려갔다. 성능은 오르는 대신 출력끼리 비슷해졌다.

verifier를 쓴 방법끼리의 ROUGE-1 차이는 replace에서 최대 0.006으로 seed 표준편차(0.005~0.010)와 비슷한 크기이고, accumulate에서는 최대 0.005로 discriminator 상위가 표준편차(0.001~0.004)보다 크게 높았다. 어느 방식에서도 conformal gate가 다른 verifier보다 낫다는 근거는 없었다.

verifier가 고른 데이터에서는 다음이 관찰된다.

- **replace에서 discriminator 계열 verifier는 세대가 지날수록 hallucination을 덜 걸러낸다.** 선택된 데이터의 hallucination 비율(seed 평균)은 1세대 0.006~0.025에서 5세대 0.044~0.052로 올라, 후보 전체 비율(0.05)에 가까워졌다. accumulate의 discriminator 상위는 5세대에도 0.017이었다. 생성 모델이 바뀌면서 discriminator가 내용의 일치보다 문체 단서에 의존하게 된 것으로 추정되나, 이 지표만으로는 확인할 수 없다.
- **정답을 참조하는 ROUGE verifier는 hallucination 비율을 가장 낮게 유지했다.** 1~5세대 모두 0.006 이하였으나, ROUGE-1은 discriminator 계열과 비슷했다.
- **verifier마다 고르는 요약 길이가 다르다.** random이 고른 요약(후보 길이의 대용값, 약 30단어) 대비 ROUGE verifier는 긴 요약(31.5~31.9단어)을, discriminator 계열은 짧은 요약(24.5~26.4단어)을 선택했다. ROUGE 상위로 학습한 모델의 출력은 26.1단어에서 27.5단어로 세 seed 모두 길어졌으나, discriminator 계열 replace에서는 출력 길이 변화의 방향이 seed마다 달랐다.

## 정리

- 텍스트에서는 verifier를 쓴 방법의 평균 ROUGE-1이 모두 random보다 높았지만, seed 3개로는 차이가 확정적이지 않다. conformal gate의 이점은 확인되지 않았다.
- replace에서 discriminator 계열이 고른 데이터의 hallucination 비율은 세대마다 높아졌다. verifier를 매 세대 held-out real 데이터로 보정하는 방향을 proposal에서 검토할 근거가 된다.
- 이 실험은 calibration과 평가가 모두 XSum에서 나온 데이터라, reference와 test가 같은 분포인 설정이다. 둘의 분포가 다른 경우(예: 일부 주제로 calibration하고 다른 주제로 평가)는 아직 확인하지 않았다.
- discriminator가 hallucination을 덜 걸러내게 되는 원인과, verifier의 길이 선호가 모델로 전이되는지는 아직 확인되지 않았다. 선택 데이터의 길이만 바꾼 대조 실험과 discriminator의 판단 근거 분석이 필요하다.

## 실행

```powershell
py run_all_text.py --quick         # 시간 측정용, 1세대
py run_all_text.py                 # seed 0, 9개 run
py run_all_text.py --seeds 1 2     # seed 1, 2 추가
py plot_results.py text_runs --text
```
