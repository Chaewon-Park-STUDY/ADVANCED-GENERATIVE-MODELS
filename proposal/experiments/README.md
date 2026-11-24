# Proposal 예비 실험: long-tail MNIST에서의 synthetic data selection

## 설정

- **데이터:** 초기 real 500장을 long-tail로 구성한다. 숫자 9가 숫자 0의 1/10이며, 숫자 7~9가 드문 mode 역할을 한다.
- **생성 모델:** Escaping Model Collapse via Synthetic Data Verification의 MNIST CVAE를 사용한다.
- **선택 방식:** class label 없이 모든 클래스를 한 번에 고르며, 세대마다 선택한 sample을 누적한다.
- **artifact 주입:** 후보의 5%를 두 숫자를 겹친 이미지로 바꿔 넣는다.
- **평가:**
  - FID는 MNIST test set 10,000장 기준으로 계산한다.
  - recall과 precision은 MNIST로 학습한 LeNet feature 공간에서 계산한다.
- **원래 설정, GPU:** 10세대, 세대당 후보 5,000장 중 10% 선택, discriminator는 real 5,000장으로 학습, seed 0·1·2.
- **verifier 조건:**
  - **RQ1, 균형 기준:** discriminator와 calibration이 균형 잡힌 real 데이터를 기준으로 학습한다.
  - **RQ4, 편향 기준:** discriminator와 calibration의 real 데이터도 생성 모델과 같은 long-tail로 구성한다. verifier가 편향된 일부만 보는 Sample Selection Bias의 상황에 해당한다. 평가는 균형 잡힌 test set으로 하므로, reference와 test의 분포가 다른 경우이기도 하다.

| 방법 | 설명 |
| --- | --- |
| random | verifier 없이 무작위 선택 |
| top-k | discriminator logit 상위 10% |
| ratio | D/(1−D)에 비례해 sampling. discriminator는 temperature scaling 적용 |
| conformal + ratio | held-out real 기준 conformal gate를 통과한 후보 안에서 ratio sampling |
| conformal + ratio + MC dropout | 위에 MC dropout 불확실성을 더함 |
| conformal + ratio + GP | 위에 GP posterior variance를 더함. gate는 넓히고 weight는 줄임 |

## 목표 지표와 분포 가정

**목표 지표.** 이 방법은 verifier 점수가 높은 sample을 최대한 많이 고르는 것을 목표로 하지 않는다. density ratio D/(1−D)에 비례해 고르면 선택된 sample이 reference real 분포를 따르게 되므로, 목표는 선택된 데이터와 모델 출력이 real 분포에 가까워지는 것이다. 따라서 주 지표는 real 분포와의 거리인 FID이며, 드문 영역이 빠지지 않았는지는 최소 class recall로 함께 확인한다. ELBO loss, precision, recall, class TV는 collapse의 양상을 진단하는 보조 지표로 쓴다.

**분포 가정.** conformal gate의 통과 보장(held-out real의 99%가 통과)은 calibration에 쓴 real 데이터와 목표 분포가 exchangeable하다는 가정에서만 성립한다. 기본 설정(RQ1)은 이 가정을 따른다. test 분포를 미리 알 수 없는 상황은 RQ4로 본다. RQ4에서는 discriminator와 calibration이 long-tail real만 보지만 FID는 균형 잡힌 MNIST test set으로 계산하므로, reference와 test의 분포가 다른 경우에 해당한다. 이 조건에서도 conformal 계열은 ELBO loss 126 근처를 유지했고, 최소 class recall은 top-k보다 높았다(0.148~0.150 대 0.093). 다만 분포 차이가 이보다 크면 통과 보장은 깨지며, covariate shift를 보정하는 weighted conformal prediction을 후속 방향으로 둔다.

## 결과: 10세대 시점, 3 seed 평균 ± 표준편차

**RQ1, 균형 기준 verifier**

| 방법 | ELBO loss ↓ | FID ↓ | precision ↑ | recall ↑ | 최소 class recall ↑ | class TV ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| random | 145.0 ± 0.6 | 191.5 ± 3.6 | 0.630 ± 0.010 | 0.236 ± 0.005 | 0.106 ± 0.023 | 0.280 ± 0.000 |
| top-k | 135.1 ± 0.4 | 127.5 ± 0.4 | 0.679 ± 0.009 | 0.288 ± 0.033 | 0.114 ± 0.054 | 0.256 ± 0.010 |
| ratio | 135.3 ± 0.8 | 127.6 ± 1.1 | 0.742 ± 0.008 | 0.278 ± 0.015 | 0.079 ± 0.037 | 0.239 ± 0.015 |
| conformal + ratio | 126.1 ± 0.1 | 119.7 ± 0.7 | 0.673 ± 0.008 | 0.311 ± 0.030 | 0.100 ± 0.018 | 0.242 ± 0.014 |
| conformal + ratio + MC dropout | 126.2 ± 0.5 | 122.5 ± 1.0 | 0.775 ± 0.026 | 0.260 ± 0.018 | 0.104 ± 0.021 | 0.264 ± 0.017 |
| conformal + ratio + GP | 126.0 ± 0.5 | 122.1 ± 1.6 | 0.718 ± 0.047 | 0.288 ± 0.079 | 0.129 ± 0.036 | 0.262 ± 0.023 |

**RQ4, 편향 기준 verifier**

| 방법 | ELBO loss ↓ | FID ↓ | precision ↑ | recall ↑ | 최소 class recall ↑ | class TV ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| top-k | 137.0 ± 0.9 | 125.9 ± 2.3 | 0.699 ± 0.006 | 0.277 ± 0.010 | 0.093 ± 0.028 | 0.284 ± 0.008 |
| ratio | 136.6 ± 0.9 | 125.8 ± 3.0 | 0.766 ± 0.014 | 0.274 ± 0.032 | 0.150 ± 0.031 | 0.273 ± 0.023 |
| conformal + ratio | 126.1 ± 0.4 | 118.3 ± 2.7 | 0.728 ± 0.018 | 0.292 ± 0.018 | 0.150 ± 0.030 | 0.268 ± 0.022 |
| conformal + ratio + GP | 126.2 ± 0.1 | 120.3 ± 2.1 | 0.681 ± 0.016 | 0.286 ± 0.041 | 0.148 ± 0.014 | 0.276 ± 0.024 |

- 0세대의 ELBO loss는 123~124, FID는 142~152, class TV는 0.295이다.
- 세대별 값은 `results_full_3seeds/all_rounds.csv`, 궤적은 `traj_RQ1.png`, `traj_RQ4.png`에 있다.

## 해석

**차이가 표준편차보다 확실히 큰 결과**

- **verifier가 없으면 collapse가 뚜렷하다.** random은 10세대에서 ELBO loss 145, FID 191로 가장 크게 악화되었다.
- **conformal gate가 collapse를 크게 늦췄다.**
  - gate를 쓴 모든 방법이 두 verifier 조건 모두에서 ELBO loss 126 근처에 머물렀다. top-k와 ratio는 135~137까지 상승했다.
  - 세대별로 보면 gate를 쓴 방법은 4세대 이후 ELBO loss가 거의 일정했지만, top-k와 ratio는 10세대까지 계속 증가했다.
  - FID도 gate를 쓴 방법이 118~122로 가장 낮았다.
- **GP는 collapse 완화에 추가로 기여하지 않았다.**
  - gate만 쓴 경우와 gate + GP의 ELBO loss는 균형 기준 126.1 대 126.0, 편향 기준 126.1 대 126.2로 같았다.
  - FID는 오히려 gate만 쓴 경우가 낮았다(119.7 대 122.1, 118.3 대 120.3). 다만 편향 기준은 표준편차가 2 이상이라 차이라고 보기 어렵다.
  - GP가 기여할 여지가 있다고 본 편향 기준에서도 최소 class recall이 0.150 대 0.148로 같았다.
- **ratio 단독은 top-k와 collapse 정도가 같았다.** ELBO loss와 FID 모두 차이가 없었다. density ratio 보정만으로는 collapse를 늦추지 못하며, conformal gate와의 결합이 필요하다.

**경향 수준의 결과**

- **verifier가 편향되면 top-k가 tail을 잃는다.**
  - 균형 기준에서 편향 기준으로 바뀌면 top-k의 class TV가 0.256에서 0.284로 악화되었다.
  - 편향 기준에서 top-k의 최소 class recall은 0.093으로, ratio와 conformal 계열 0.148~0.150보다 낮았다. 차이는 표준편차의 약 2배이다.
  - 반면 ratio 계열은 편향 기준에서도 최소 class recall을 유지했다.
- **recall은 verifier를 쓴 방법 사이에 차이가 없었다.** 모두 0.27~0.29로, random 0.236보다 높았다.

- **불확실성 항은 precision과 recall의 균형을 바꾼다.** 균형 기준에서 GP나 MC dropout을 붙이면 precision이 0.673에서 0.718~0.775로 오르고 recall은 0.311에서 0.260~0.288로 내려갔다. 편향 기준에서는 gate만 쓴 경우의 precision이 0.728로 GP(0.681)보다 높아, 방향이 조건에 따라 다르다.

## marginal 영역 선택 (rare lift)

숫자 7~9를 data의 marginal 영역으로 두고, verifier가 이 영역의 sample을 실제로 골랐는지 본다. rare lift는 선택된 데이터의 7~9 비율에서 후보 전체의 7~9 비율을 뺀 값이며, 1~10세대 평균이다. 0보다 크면 7~9를 후보보다 많이 고른 것이다.

| verifier | 방법 | rare lift |
| --- | --- | --- |
| 균형 | random | 0.012 ± 0.000 |
| 균형 | top-k | 0.003 ± 0.003 |
| 균형 | ratio | 0.013 ± 0.003 |
| 균형 | conformal + ratio | 0.009 ± 0.003 |
| 균형 | conformal + ratio + MC dropout | 0.005 ± 0.012 |
| 균형 | conformal + ratio + GP | 0.009 ± 0.011 |
| 편향 | top-k | 0.002 ± 0.001 |
| 편향 | ratio | 0.002 ± 0.007 |
| 편향 | conformal + ratio | −0.001 ± 0.003 |
| 편향 | conformal + ratio + GP | −0.001 ± 0.002 |

- 모든 방법의 rare lift가 ±0.013 이내로, 7~9를 후보보다 뚜렷하게 더 고른 방법은 없었다.
- 균형 기준에서는 random도 0.012로 0이 아니어서 이를 기준으로 비교한다. top-k는 0.003으로 random보다 marginal sample을 덜 골랐고, ratio는 0.013으로 random 수준을 유지했다.
- 편향 기준에서는 모든 방법이 0 근처였다. 따라서 편향 기준의 최소 class recall 차이(0.093 대 0.150)는 선택 비율의 차이로는 설명되지 않으며, verifier가 marginal sample을 판별한다는 근거는 아직 없다.

## 축소 설정 결과와의 차이

노트북 CPU용 축소 설정(5세대, 후보 2,000장)의 결과는 `results_lite_3seeds/`에 남겨 둔다. 축소 설정에서는 ratio의 precision이 가장 낮았으나, 원래 설정에서는 ratio의 precision이 가장 높았다. 따라서 "ratio 단독은 precision을 떨어뜨린다"는 축소 설정의 해석은 철회한다. 원래 설정에서 확인된 결론은 "ratio 단독은 collapse를 늦추지 못한다"이다.

## 실행

```powershell
py scripts\run_proposal.py --seeds 0 1 2 --core --gpu   # RQ1 핵심과 RQ4
py scripts\run_proposal.py --seeds 0 1 2 --gpu          # 전체 grid
py scripts\run_proposal.py --seeds 0 1 2 --lite         # 노트북용 축소 설정
py scripts\summarize_proposal.py
```

코드는 `code/` 폴더에 있다. 이 폴더의 파일은 원 레포 `Verified-Synthetic-Data`의 `MNIST/scripts`에 넣어 실행한다.
