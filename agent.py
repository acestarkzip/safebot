
import os
from dotenv import load_dotenv
from openai import OpenAI

# 환경변수 불러오기
load_dotenv()

# API 클라이언트 생성
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def analyze_safety(risk_level, robot_data):

    # 입력값을 사람이 이해하기 쉬운 형태로 변환
    obstacle_text = (
        "장애물 감지됨"
        if robot_data["obstacle_detected"] == 1
        else "장애물 감지 안 됨"
    )

    emergency_text = (
        "비상 정지 기능 활성화"
        if robot_data["emergency_stop"] == 1
        else "비상 정지 기능 비활성화"
    )

    risk_labels = {
        "Low": "낮은 위험도",
        "Medium": "중간 위험도",
        "High": "높은 위험도"
    }

    risk_text = risk_labels.get(risk_level, risk_level)

    prompt = f"""
당신은 로봇 작업 환경의 안전 분석을 돕는 전문가입니다.

다음 데이터를 바탕으로 로봇 작업 환경을 분석해 주세요.

[위험도 예측 결과]
- 머신러닝 예측 등급: {risk_text}

[작업 환경 데이터]
- 로봇 이동 속도: {robot_data["robot_speed"]} m/s
- 작업자와의 거리: {robot_data["human_distance"]} m
- 장애물 감지: {obstacle_text}
- 작업 공간 혼잡도: {robot_data["workspace_congestion"]} / 5
- 비상 정지 기능: {emergency_text}

[중요한 해석 기준]
1. emergency_stop = 1은 비상 정지 기능이 활성화된 상태입니다.
2. emergency_stop = 0은 비상 정지 기능이 비활성화된 상태입니다.
3. 비상 정지 기능의 활성화 여부만으로 실제 비상 정지 작동 횟수나 빈도를 추측하지 마세요.
4. 비상 정지 기능이 활성화되어 있다면 안전 기능이 활성화된 상태라고만 설명하세요.
5. 비상 정지 기능이 비활성화되어 있다면 비상 상황에 대비한 안전 기능 점검이 필요하다고 설명하세요.
6. 입력 데이터에 없는 사실이나 상황을 임의로 만들어내지 마세요.
7. 머신러닝 예측 결과와 실제 현장의 안전 상태는 동일하지 않습니다.

[답변 형식]
다음 세 가지 항목으로 한국어 분석 결과를 작성해 주세요.

### 1. 위험 원인
입력 데이터를 바탕으로 주요 위험 요소를 설명하세요.

### 2. 안전 개선 방안
각 위험 요소에 대응할 수 있는 구체적인 개선 방법을 제안하세요.

### 3. 추가 주의사항
작업자가 추가로 확인해야 할 안전 사항을 설명하세요.

교육용 프로토타입의 분석 결과이므로 실제 산업 현장의 안전 인증이나 전문적인 위험성 평가를 대체할 수 없다는 점을 명시하세요.
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    return response.output_text
