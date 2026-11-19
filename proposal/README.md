# Proposal 예비 실험: long-tail MNIST에서의 synthetic data selection

## 설정

- **데이터:** 초기 real 500장을 long-tail로 구성한다. 숫자 9가 숫자 0의 1/10이며, 숫자 7~9가 드문 mode 역할을 한다.
- **생성 모델:** Escaping Model Collapse via Synthetic Data Verification의 MNIST CVAE를 사용한다.
- **선택 방식:** class label 없이 모든 클래스를 한 번에 고른다. 매 세대 선택한 sample을 누적한다.
- **artifact 주입:** 매 세대 후보의 5%를 두 숫자를 겹친 이미지로 바꿔 넣는다.
- **평가 지표:** recall과 precision은 MNIST로 학습한 LeNet feature 공간에서 계산한다.
- **축소 설정 `--lite`:** 노트북 CPU로 돌리기 위한 설정이다.
  - 5세대, 세대당 후보 2,000장 중 10% 선택
  - discriminator는 real 2,000장, 5 epoch로 학습
  - CVAE는 100 epoch로 학습
  - seed 0, 1, 2 반복

| 방법 | 설명 |
| --- | --- |
| random | verifier 없이 무작위 선택 |
| top-k | discriminator logit 상위 10% |
| ratio | D/(1−D)에 비례해 sampling. discriminator는 temperature scaling 적용 |
| conformal + ratio + GP | held-out real 기준 conformal gate를 통과한 후보 안에서 ratio sampling. GP posterior variance로 gate는 넓히고 weight는 줄임 |
| RQ3 재학습 가중치 | conformal + ratio + GP로 고른 sample을 재학습할 때 ratio^0.5로 가중. 구현은 복원 추출 |

## 결과: 5세대 시점, 3 seed 평균 ± 표준편차

| 방법 | ELBO loss ↓ | class TV ↓ | precision ↑ | recall ↑ | 최소 class recall ↑ | 드문 숫자 추가 선택률 | artifact 통과율 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 128.9 ± 0.6 | 0.310 ± 0.000 | 0.734 ± 0.017 | 0.310 ± 0.016 | 0.109 ± 0.030 | −0.003 | 0.070 |
| top-k | 125.8 ± 0.3 | 0.270 ± 0.012 | 0.698 ± 0.023 | 0.318 ± 0.030 | 0.135 ± 0.018 | 0.016 | 0.049 |
| ratio | 125.5 ± 0.1 | 0.286 ± 0.008 | 0.676 ± 0.020 | 0.294 ± 0.056 | 0.112 ± 0.033 | 0.008 | 0.045 |
| conformal + ratio + GP | 125.3 ± 0.5 | 0.250 ± 0.006 | 0.757 ± 0.031 | 0.296 ± 0.022 | 0.167 ± 0.040 | 0.013 | 0.046 |
| RQ3 재학습 가중치 | 125.6 ± 0.4 | 0.242 ± 0.012 | 0.688 ± 0.103 | 0.223 ± 0.025 | 0.055 ± 0.016 | 0.013 | 0.023 |

- 시작 시점(0세대)의 class TV는 0.295이다.
- 드문 숫자 추가 선택률은 선택된 sample의 숫자 7~9 비율에서 후보 전체의 해당 비율을 뺀 값이다.
- artifact 통과율의 기준값은 후보 전체 비율 0.05이다.
- 세대별 값은 `results_lite_3seeds/all_rounds.csv`, 그림은 `traj_RQ1.png`, `traj_RQ3.png`에 있다.

## 해석

**차이가 표준편차보다 확실히 큰 결과**

- verifier를 쓰면 collapse가 줄었다. random은 ELBO loss가 128.9까지 악화되었고, verifier를 쓴 방법은 모두 125대에 머물렀다. random은 class 비율도 다수 클래스 쪽으로 계속 쏠렸다(class TV 0.295 → 0.310).
- conformal + ratio + GP가 class 균형과 precision 모두에서 가장 좋았다.
  - class TV는 0.250으로, ratio 0.286과 top-k 0.270보다 낮았다.
  - precision은 0.757로, ratio 0.676보다 높았다.
- ratio만 쓰면 precision이 가장 낮았고 recall도 개선되지 않았다. 따라서 calibration은 conformal gate와 결합해야 효과가 있다.
- RQ3의 복원 추출 방식 재학습 가중치는 recall과 최소 class recall을 모두 떨어뜨렸다. 현재 구현으로는 가설이 지지되지 않는다. 중복 sample로 유효 데이터가 줄어든 것이 원인으로 추정된다.

**아직 결론 내리기 어려운 결과**

- recall은 네 방법 모두 0.29~0.32로, 표준편차 범위에서 차이가 없다.
- 최소 class recall은 conformal + ratio + GP가 가장 높지만, 표준편차가 커서 경향 수준이다.
- top-k도 드문 숫자를 우대했다. discriminator가 균형 잡힌 real 기준으로 학습되어, 드문 숫자의 p_data/p_gen이 크게 추정되기 때문이다. 따라서 verifier 자체가 편향된 Sample Selection Bias의 조건은 아직 재현되지 않았다.

## 진행 중

**RQ4, 편향된 verifier:** discriminator와 calibration의 real 기준도 long-tail로 둔다. 원래 설정으로 GPU에서 실행 중이다.
- 10세대, 후보 5,000장, FID 포함
- top-k, ratio, conformal + ratio + GP 3개 설정
- RQ1 기준 4개 설정도 같은 설정으로 다시 실행해 비교한다.

## 실행

```powershell
py scripts\run_proposal.py --seeds 0 1 2 --lite      # 노트북용 축소 설정
py scripts\run_proposal.py --seeds 0 1 2 --core --gpu # GPU, RQ1 핵심과 RQ4
py scripts\summarize_proposal.py
```

코드는 `code/` 폴더에 있다. 이 폴더의 파일은 원 레포 `Verified-Synthetic-Data`의 `MNIST/scripts`에 넣어 실행한다.
