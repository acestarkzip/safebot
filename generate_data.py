
import numpy as np
import pandas as pd
from pathlib import Path

# 재현 가능한 데이터 생성
np.random.seed(42)

# 데이터 개수
NUM_SAMPLES = 1000

# 가상 로봇 작업 환경 생성
data = pd.DataFrame({
    "robot_speed": np.random.uniform(0.1, 2.0, NUM_SAMPLES),
    "human_distance": np.random.uniform(0.2, 5.0, NUM_SAMPLES),
    "obstacle_detected": np.random.randint(0, 2, NUM_SAMPLES),
    "workspace_congestion": np.random.randint(1, 6, NUM_SAMPLES),
    "emergency_stop": np.random.randint(0, 2, NUM_SAMPLES)
})

# 위험도 점수 계산 (교육용 가상 규칙)
score = (
    (data["robot_speed"] > 1.2) * 2
    + (data["human_distance"] < 1.5) * 3
    + (data["obstacle_detected"] == 1) * 2
    + (data["workspace_congestion"] >= 4) * 2
    + (data["emergency_stop"] == 0) * 2
)

# 위험 등급 분류
data["risk_level"] = np.select(
    [score <= 2, score <= 5],
    ["Low", "Medium"],
    default="High"
)

# CSV 저장
output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

data.to_csv(output_dir / "safety_data.csv", index=False)

print("Safety dataset generated successfully!")
print(data.head())
print("\nRisk distribution:")
print(data["risk_level"].value_counts())
