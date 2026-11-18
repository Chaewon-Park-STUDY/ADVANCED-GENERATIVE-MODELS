"""
verified(필터 O)와 nofilter(필터 X) 실험을 차례로 실행하는 스크립트.
MNIST 폴더에서 `python scripts/run_both.py` 한 줄로 실행하면 됨.
아래 SETTINGS 숫자만 바꿔서 쓰면 됨.
"""
import os
import subprocess
import sys
from pathlib import Path

SETTINGS = {
    "K": "10",          # 라운드 수 (논문 원본은 40)
    "SYN": "20000",     # 라운드당 합성 데이터 수
    "DEP": "80",        # discriminator 최대 epoch
    "FID_N": "6000",    # FID 계산용 생성 이미지 수
    "FID_REAL": "10000" # FID 계산용 real test 이미지 수
}

RUNS = [
    {"TAG": "verified", "TH": "0.1"},  # 상위 10%만 통과
    {"TAG": "nofilter", "TH": "1.0"},  # 필터 없음 (baseline)
]

script = Path(__file__).resolve().parent / "run_small.py"
mnist_dir = script.parent.parent

for run in RUNS:
    env = os.environ.copy()
    env.update(SETTINGS)
    env.update(run)
    print(f"\n===== {run['TAG']} 실행 시작 =====", flush=True)
    subprocess.run([sys.executable, str(script)], env=env, cwd=mnist_dir, check=True)

print("\n완료. 결과는 MNIST/result/verified, MNIST/result/nofilter 폴더에 있음.")
